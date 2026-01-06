import os
import sys
import glob
import pandas as pd
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# ADK 및 GenAI 관련
try:
    from google.adk.agents import Agent  # type: ignore[import-not-found]
    from google.adk.tools import google_search  # type: ignore[import-not-found]
except ImportError:
    print("❌ google-adk 패키지가 설치되어 있지 않습니다.")
    print("   다음 명령어로 설치해주세요: pip install google-adk")
    sys.exit(1)

from adk_utils import AdkAgentRunner, print_runtime_llm_config, _ensure_google_api_key

try:
    # 기존 근거 데이터 생성 로직(동일 동작 유지)
    # NOTE: stock_analyzer.py는 google-genai/yfinance/ta-lib 의존성이 있으므로 설치 필요
    from google import genai
    from google.genai import types
    
    from stock_analyzer import (
        API_DELAY,
        DEFAULT_MODEL,
        MARKET_DATA_OUTPUT_DIR,
        MARKET_INFO,
        STRATEGY_INFO,
        create_analyzer_output_dir,
        fetch_and_save_market_data_for_stock,
        get_latest_screener_dir,
    )
except Exception as e:
    print("❌ 기존 stock_analyzer 모듈 임포트 실패:", str(e))
    print("   requirements.txt의 의존성(google-genai, yfinance, ta-lib 등)을 설치했는지 확인해주세요.")
    sys.exit(1)


# 사용자가 gemini-2.5-flash로 설정했으나, 반복 생성 문제가 있다면 1.5로 롤백 권장
ADK_DEFAULT_MODEL = os.environ.get("MARKET_LENS_ADK_MODEL", "gemini-2.5-flash")


class StockAnalyzerADK:
    def __init__(self, model: str = ADK_DEFAULT_MODEL, use_tools: bool = False):
        # ADK는 tools를 function calling으로 실행하므로, tool 지원 모델 사용을 권장
        self.model = model or ADK_DEFAULT_MODEL
        self.use_tools = bool(use_tools)
        self.market_data_dir: Optional[str] = None
        
        # 파일 업로드를 위한 GenAI 클라이언트 초기화
        _ensure_google_api_key()
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        if self.use_tools:
            instruction = (
                "너는 월스트리트에서 일하는 시니어 애널리스트다. "
                "사용자가 제공한 종목을 10단계 목차에 따라 분석 보고서 형태로 한국어로 작성한다. "
                "필요하면 google_search 도구로 최신 정보를 확인하고, 첨부된 CSV 파일(가격/지표, 재무)을 우선 근거로 사용한다."
            )
            tools = [google_search]
        else:
            # Tool-less 모드 (사용 안함)
            instruction = (
                "너는 월스트리트에서 일하는 시니어 애널리스트다. "
                "사용자가 제공한 종목을 10단계 목차에 따라 분석 보고서 형태로 한국어로 작성한다. "
                "프롬프트에 포함된 텍스트 근거 데이터를 최우선으로 참고하여 분석하라."
            )
            tools = []

        self._agent = Agent(
            name="market_lens_stock_analyst",
            model=self.model,
            instruction=instruction,
            description="Market Lens AI - Stock Analysis Agent (ADK)",
            tools=tools,
        )
        self._runner = AdkAgentRunner(self._agent)

    def prepare_evidence_for_row(self, row: pd.Series, market: str = "us") -> tuple[str, str]:
        """
        한 종목에 대해 yfinance/ta-lib로 데이터를 가져와 CSV로 저장하고 경로 반환.
        순차 실행 환경에서 호출됨.
        """
        ticker = row.get("ticker")
        if not ticker:
            return "", ""

        # 시장 데이터 디렉토리 설정
        output_date = pd.Timestamp.now().strftime("%Y%m%d")
        self.market_data_dir = os.path.join(MARKET_DATA_OUTPUT_DIR, output_date)
        os.makedirs(self.market_data_dir, exist_ok=True)

        _, price_csv, fin_csv = fetch_and_save_market_data_for_stock(
            ticker, market, self.market_data_dir
        )
        return price_csv or "", fin_csv or ""

    def build_evidence_text(self, price_csv: str, fin_csv: str, max_chars: int = 15000) -> str:
        """
        로컬 CSV 파일 내용을 텍스트로 읽어 요약/헤드만 반환.
        ADK Function Calling이 안 될 때 프롬프트에 텍스트로 주입하기 위함.
        (현재 파일 첨부 방식으로 변경되어 사용하지 않을 수 있음)
        """
        parts = []
        if price_csv and os.path.exists(price_csv):
            try:
                dfp = pd.read_csv(price_csv)
                if not dfp.empty:
                    parts.append("[PRICE_INDICATOR_HEAD_30]")
                    parts.append(dfp.tail(30).to_csv(index=False))  # 최근 30일
                    parts.append("\n[PRICE_SUMMARY]")
                    parts.append(dfp.describe().to_string())
            except Exception as e:
                parts.append(f"[PRICE_CSV_ERROR] {e}")

        if fin_csv and os.path.exists(fin_csv):
            try:
                dff = pd.read_csv(fin_csv)
                if not dff.empty:
                    parts.append("[FIN_HEAD_30]")
                    parts.append(dff.head(30).to_csv(index=False))
            except Exception as e:
                parts.append(f"[FIN_CSV_ERROR] {e}")

        text = "\n".join([p for p in parts if p])
        if len(text) > max_chars:
            return text[: max_chars - 200] + "\n...[TRUNCATED]..."
        return text

    def analyze_strategy(
        self,
        df: pd.DataFrame,
        strategy: str,
        max_stocks: int = 10,
        market: Optional[str] = None,
        concurrency: int = 1,
    ) -> List[Dict]:
        results: List[Dict] = []
        strategy_info = STRATEGY_INFO.get(strategy, {})

        print(f"\n📊 {strategy_info.get('name', strategy)} 전략 분석 시작...")
        print(f"   총 {len(df)}개 종목 중 상위 {min(len(df), max_stocks)}개 분석")

        # 병렬 실행 (asyncio.gather + 개별 Runner)
        # - ParallelAgent는 개별 파일 첨부가 어려워, 독립 세션 병렬 실행 방식으로 전환함.
        # - yfinance 데이터 준비는 순차 처리 (Rate Limit)
        if int(concurrency) > 1:
            head_df = df.head(max_stocks).reset_index(drop=True)
            batch_size = int(concurrency)

            final_results: List[Dict] = []

            # 1) 순차적으로 Evidence(CSV) 준비
            prepared_data: List[Dict] = []
            for i in range(len(head_df)):
                row = head_df.iloc[i]
                ticker = row.get("ticker", "Unknown")
                name = row.get("name", "Unknown")
                print(f"   [{i+1}/{len(head_df)}] {ticker} ({name}) 근거 데이터 생성 중...(순차)")

                price_csv, fin_csv = self.prepare_evidence_for_row(row, market=market)

                prompt = self._create_analysis_prompt(
                    row=row,
                    strategy=strategy,
                    market=market,
                    evidence_price_csv=price_csv,
                    evidence_fin_csv=fin_csv,
                )

                prepared_data.append({
                    "idx": i,
                    "ticker": ticker,
                    "name": name,
                    "row": row,
                    "prompt": prompt,
                    "price_csv": price_csv,
                    "fin_csv": fin_csv,
                })

                if i < len(head_df) - 1:
                    time.sleep(API_DELAY)

            # 2) 배치 단위로 병렬 실행 (asyncio.gather)
            for i in range(0, len(prepared_data), batch_size):
                batch = prepared_data[i : i + batch_size]
                print(f"\n   🚀 Batch [{i+1}~{min(i+batch_size, len(prepared_data))}/{len(prepared_data)}] 병렬 분석 실행 중...")

                # 내부 async 함수 정의 (동기 메서드 내에서 실행하기 위함)
                async def _run_batch_async(batch_items):
                    async def _analyze_one(item):
                        # 파일 업로드 -> Content 생성 -> Runner 실행 -> 파일 삭제
                        ticker = item["ticker"]
                        uploaded_files = []
                        try:
                            # 파일 업로드 (스레드풀에서 실행하여 이벤트 루프 블로킹 방지)
                            if item["price_csv"]:
                                f1 = await asyncio.to_thread(
                                    self.client.files.upload,
                                    file=item["price_csv"],
                                    config=types.UploadFileConfig(mime_type='text/csv')
                                )
                                uploaded_files.append(f1)
                            if item["fin_csv"]:
                                f2 = await asyncio.to_thread(
                                    self.client.files.upload,
                                    file=item["fin_csv"],
                                    config=types.UploadFileConfig(mime_type='text/csv')
                                )
                                uploaded_files.append(f2)
                            
                            # Content 구성 (프롬프트 + 파일)
                            # types.Content 생성 시 str이나 File 객체를 직접 넣으면 Pydantic 검증 오류 발생 가능
                            # 명시적으로 types.Part 객체로 변환하여 구성함
                            parts = [types.Part(text=item["prompt"])]
                            for f in uploaded_files:
                                # File 객체 -> Part(file_data=...) 변환
                                parts.append(types.Part(
                                    file_data=types.FileData(
                                        mime_type=f.mime_type, 
                                        file_uri=f.uri
                                    )
                                ))
                            
                            new_message = types.Content(role="user", parts=parts)
                            
                            # Runner 실행 (독립 세션)
                            result_text = await self._runner.run_text_async(
                                prompt="", # new_message로 전달하므로 빈 문자열
                                new_message=new_message,
                                session_id=str(uuid4()) # 독립 세션
                            )
                            
                            # 파일 정리
                            for f in uploaded_files:
                                try:
                                    await asyncio.to_thread(self.client.files.delete, name=f.name)
                                except: pass
                                
                            return {
                                "ticker": item["ticker"],
                                "name": item["name"],
                                "strategy": strategy,
                                "analysis": result_text,
                                "data": item["row"].to_dict(),
                            }
                        except Exception as e:
                            print(f"       ❌ {ticker} 분석 실패: {e}")
                            # 파일 정리 (에러 시에도)
                            for f in uploaded_files:
                                try:
                                    await asyncio.to_thread(self.client.files.delete, name=f.name)
                                except: pass
                            return None

                    tasks = [_analyze_one(item) for item in batch_items]
                    return await asyncio.gather(*tasks)

                # asyncio.run으로 비동기 배치 실행
                batch_results = asyncio.run(_run_batch_async(batch))
                
                for res in batch_results:
                    if res:
                        print(f"       ✅ 완료: {res['ticker']}")
                        final_results.append(res)
            
            return final_results

        # 순차 실행 (Concurrency=1)
        for idx, (_, row) in enumerate(df.head(max_stocks).iterrows()):
            ticker = row.get("ticker", "Unknown")
            name = row.get("name", "Unknown")
            print(f"   [{idx+1}/{min(len(df), max_stocks)}] {ticker} ({name}) 분석 중...")

            try:
                # 1. 근거 데이터 생성 (yfinance)
                price_csv, fin_csv = self.prepare_evidence_for_row(row, market=market)

                # 2. 분석 수행 (ADK 호출 - 파일 첨부 방식)
                # 프롬프트 생성
                prompt = self._create_analysis_prompt(
                    row=row,
                    strategy=strategy,
                    market=market,
                    evidence_price_csv=price_csv,
                    evidence_fin_csv=fin_csv,
                )

                # 파일 업로드
                uploaded_files = []
                if price_csv:
                    f1 = self.client.files.upload(file=price_csv, config=types.UploadFileConfig(mime_type='text/csv'))
                    uploaded_files.append(f1)
                if fin_csv:
                    f2 = self.client.files.upload(file=fin_csv, config=types.UploadFileConfig(mime_type='text/csv'))
                    uploaded_files.append(f2)

                # Content 생성
                parts = [prompt]
                for f in uploaded_files:
                    parts.append(f)
                
                new_message = types.Content(role="user", parts=parts)
                
                # 실행 (동기 컨텍스트이므로 asyncio.run 사용)
                analysis_text = asyncio.run(self._runner.run_text_async(prompt="", new_message=new_message, session_id=str(uuid4())))
                
                # 파일 삭제
                for f in uploaded_files:
                    try:
                        self.client.files.delete(name=f.name)
                    except: pass

                if analysis_text:
                    results.append({
                        "ticker": ticker,
                        "name": name,
                        "strategy": strategy,
                        "analysis": analysis_text,
                        "data": row.to_dict(),
                    })
                
                time.sleep(API_DELAY)

            except Exception as e:
                print(f"       ❌ 분석 실패: {e}")
                # import traceback
                # traceback.print_exc()

        return results

    def _create_analysis_prompt(
        self,
        row: pd.Series,
        strategy: str,
        market: Optional[str],
        evidence_price_csv: str = "",
        evidence_fin_csv: str = "",
        evidence_text: str = "",
    ) -> str:
        ticker = row.get("ticker", "Unknown")
        name = row.get("name", "Unknown")
        market_name = MARKET_INFO.get(market, {}).get("name", market) if market else "Global"
        
        # 기본 정보 구성
        ticker_value = f"{ticker} ({name})" if name != "Unknown" else ticker
        
        # 전략 메타데이터
        strategy_info = STRATEGY_INFO.get(strategy, {})
        
        # 종목 기본 정보 블록
        stock_info_lines = [f"### 분석 대상: {ticker_value} ({market_name})"]
        for k, v in row.to_dict().items():
            if k not in ["ticker", "name"]:
                stock_info_lines.append(f"- {k}: {v}")
        stock_info = "\n".join(stock_info_lines)

        evidence_block = ""
        # 파일 경로가 있는 경우 (첨부 파일 안내)
        if evidence_price_csv or evidence_fin_csv:
             evidence_block = "\n".join(
                [
                    "## 근거 데이터 (첨부 파일)",
                    "분석 요청 메시지에 CSV 파일(가격/지표, 재무)이 첨부되어 있다.",
                    "이 파일들의 데이터를 **최우선 근거**로 사용하여 분석하라.",
                ]
            )

        return f"""
너는 월스트리트에서 일하고 있는 기업 분석 및 주식 시장 분석의 전문가야. 너의 이름은 'Gemini Stock Analyst'야. 너는 사용자가 입력한 주식 종목에({ticker_value}) 대해서 각 단계별로 분석하고 최종 투자 의사 결정에 도움을 주는 역할을 한다.

목표 및 역할:
* 사용자가 요청한 특정 주식 종목에 대해 심층적인 기업 및 시장 분석 보고서를 제공한다.
* 보고서는 투자 의사 결정에 실질적인 도움을 줄 수 있도록 최신 정보를 기반으로 상세하고 깊이 있게 작성한다.
* 모든 답변은 한국어로 제공하며, 전문적인 보고서 양식을 따른다.
* 마크다운 형식을 사용한다.

## 스크리닝/종목 메타 정보
전략: {strategy_info.get('name', strategy)}
전략 설명: {strategy_info.get('description', '')}
핵심 지표: {strategy_info.get('focus', '')}

{stock_info}

{evidence_block}

행동 및 규칙:
1) 분석 보고서 작성:
   a) 사용자가 입력한 종목({ticker_value})에 대해, **google_search** 도구를 활용하여 최신 뉴스와 이슈를 확인한다.
   b) 첨부된 CSV 파일들의 데이터를 기반으로 정량적 분석을 수행한다.
   c) 수집된 정보를 기반으로 아래 제시된 10단계 분석 과정을 철저히 따른다.
   d) 각 단계별 분석 내용은 가능한 한 상세하고 심층적이어야 하며, 데이터와 근거를 명확하게 제시해야 한다.
   e) 특히 '기술적 분석' 단계에서는 최근 1년간의 주가 트렌드와 차트 패턴 및 첨부된 CSV의 기술적 지표를 분석하고, '재무 상태 분석' 단계에서는 최근 3개년 및 최근 4개 분기 재무제표를 종합 분석한 내용을 필수로 포함한다.
   f) '가치 평가' 단계에서는 아래 절차에 명시된 가치평가기법을 필수로 활용하여 기업의 적정 가치와 현재 주가를 비교하여 투자 의견을 제시하도록 한다.
2) 10단계 분석 절차 (보고서 목차):
   1. 회사 개요: 기업의 핵심 사업, 역사, 현재 시장 위치.
   2. 기술적 분석: 최신 자료를 참고한 가격 움직임, 수급 상황, 추세, 모멘텀 등의 기술적 지표 및 차트 분석.
   3. 재무 상태 분석: 현재 시점으로부터 최근 3개년 회계 연도 및 최근 4개 분기 재무제표(매출, 영업이익, 순이익, 부채비율 등) 종합 분석.
   4. 정성적 리서치: 속한 산업 개요, 경쟁 구도, 기업의 경쟁 우위 및 지속 가능성, 거버넌스 등 정성적 요소 평가.
   5. 매크로적 고려사항: 거시 경제 환경(금리, 인플레이션, 환율 등)이 기업 사업 및 실적에 미치는 영향 분석.
   6. 가치 평가: 상대가치평가(Peer Group Analysis)와 내재가치평가방법(DCF, Reverse DCF, DDM, RIM)을 활용하여 적정 가치 도출 및 현재 주가 대비 투자 의견 제시 (예: '매수', '보유', '매도').
   7. 리스크 평가: 투자 시 고려해야 할 주요 리스크 요인(경영, 산업, 규제 등)과 리스크 완화 요소 제시.
    8. 외부 분석 평가: 외부 리서치 및 분석 보고서의 주요 가설 및 내용에 대한 비교 및 의견 제시.
   9. 현재 시점의 투자 매력도 평가: 매크로, 시장 상황, 산업 전망, 기업 비전 등을 종합한 최종 투자 매력도 평가.
   10. 최종 결론 및 투자 전략 제시: 분석 내용을 기반으로 사용자의 투자 의사 결정에 대한 최종 결론 및 구체적인 투자 포트폴리오 전략 제시.
3) 전문성 유지:
   a) 답변은 통계적 데이터와 금융 지표에 근거하여 작성한다.
   b) 주관적인 감정 표현이나 불필요한 사족은 피하고, 객관적이고 사실적인 정보를 제공하는 데 집중한다.
4) 근거 데이터 우선:
   a) 첨부된 CSV 파일 데이터와 웹 검색 결과가 충돌하면, 원칙적으로 첨부 데이터를 우선하되, 차이가 발생한 이유(시점/통화/단위 등)를 명시한다.
   b) **이전에 생성한 내용을 반복해서 출력하지 않는다.**

전반적인 어조:
* 전문적이고 신뢰감을 주는 어조를 사용한다.
* 보고서 형식에 맞춰 격식 있고 명확한 문체를 유지한다.
* 사용자의 투자 결정을 지원하는 조력자로서의 역할을 수행한다.
"""

    def merge_analysis_reports(self, analyzer_output_dir: str):
        """
        생성된 analysis_*.md 파일들을 읽어서 시장별 통합 보고서(investment_report.md)를 생성한다.
        PortfolioMakerADK가 올바른 입력 파일을 읽도록 보장하기 위함.
        """
        analysis_files = glob.glob(os.path.join(analyzer_output_dir, "analysis_*.md"))
        if not analysis_files:
            return

        print(f"\n📑 통합 보고서 생성 중... ({len(analysis_files)}개 파일 병합)")
        
        # 시장별로 분류
        market_files = {}
        for f in analysis_files:
            filename = os.path.basename(f)
            # analysis_us_growth.md -> market=us
            parts = filename.split("_")
            if len(parts) >= 2:
                market = parts[1]
                if market not in market_files:
                    market_files[market] = []
                market_files[market].append(f)

        for market, files in market_files.items():
            combined_report = []
            combined_report.append(f"# {market.upper()} Investment Report (Combined)")
            combined_report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
            
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8") as rf:
                        content = rf.read()
                        combined_report.append(content)
                        combined_report.append("\n\n---\n\n")
                except Exception as e:
                    print(f"  ⚠️ 읽기 실패: {f} / {e}")
            
            output_filename = f"{market}_investment_report.md"
            output_path = os.path.join(analyzer_output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as wf:
                wf.write("\n".join(combined_report))
            print(f"  ✅ 통합 보고서 저장: {output_path}")

    def run_analysis(
        self,
        screener_output_dir: str,
        max_stocks_per_strategy: int = 5,
        concurrency: int = 1,
    ) -> Tuple[List[Dict], str]:
        """
        스크리닝 결과 디렉토리를 로드하여 전체 분석 프로세스를 실행한다.
        (StockAnalyzer.run_analysis와 유사 인터페이스)
        """
        # 1. 스크리닝 결과 로드
        from stock_analyzer import StockAnalyzer
        # 임시 인스턴스로 로드 기능 사용
        sa_loader = StockAnalyzer(api_key="DUMMY") 
        screening_results = sa_loader.load_screening_results(screener_output_dir)
        
        # 2. 출력 디렉토리 생성
        analyzer_output_dir = create_analyzer_output_dir()
        print(f"📁 분석 결과 저장 경로: {analyzer_output_dir}")

        all_results = []

        # 3. 분석 수행
        for market, strategies in screening_results.items():
            for strategy, df in strategies.items():
                if df.empty:
                    continue
                
                strategy_results = self.analyze_strategy(
                    df, 
                    strategy, 
                    max_stocks=max_stocks_per_strategy,
                    market=market,
                    concurrency=concurrency
                )
                
                if not strategy_results:
                    continue

                # 리포트 저장
                filename = f"analysis_{market}_{strategy}.md"
                filepath = os.path.join(analyzer_output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {market.upper()} {strategy} Strategy Analysis Report\n\n")
                    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
                    f.write(f"Total Stocks: {len(strategy_results)}\n\n")
                    
                    for res in strategy_results:
                        f.write(f"## {res['ticker']} - {res['name']}\n\n")
                        f.write(res['analysis'])
                        f.write("\n\n---\n\n")
                
                print(f"   💾 리포트 저장 완료: {filepath}")
                all_results.extend(strategy_results)
        
        # 4. 통합 보고서 생성 (PortfolioMaker용)
        self.merge_analysis_reports(analyzer_output_dir)
        
        return all_results, analyzer_output_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Lens AI - Stock Analyzer (ADK Version)")
    parser.add_argument("--screener-output", type=str, help="Path to screener output directory")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent LLM calls (default: 1)")
    parser.add_argument("--debug-config", action="store_true", help="Print runtime LLM configuration")
    args = parser.parse_args()

    if args.debug_config:
        print_runtime_llm_config()
    
    analyzer = StockAnalyzerADK(use_tools=True) # 도구 사용 (google_search)

    # 1. 스크리너 결과 로드
    screener_dir = args.screener_output or get_latest_screener_dir()
    if not screener_dir:
        print("❌ 스크리닝 결과를 찾을 수 없습니다.")
        sys.exit(1)
    
    # run_analysis 메서드 사용
    all_results, analyzer_dir = analyzer.run_analysis(
        screener_dir,
        max_stocks_per_strategy=999, # CLI에서 호출 시 상위 제한은 analyze_strategy 내부에서 처리하거나 여기서 처리
        concurrency=args.concurrency
    )
    
    print("\n✨ 모든 분석이 완료되었습니다.")


if __name__ == "__main__":
    main()
