"""
Portfolio Maker (ADK 버전) - LLM 기반 최종 투자 추천 및 포트폴리오 전략 보고서 생성기

기존 portfolio_maker.py의 로직(입출력/파일명/보고서 헤더)은 유지하면서,
LLM 호출부만 Google ADK 기반으로 교체한 새 버전입니다.

- 입력: output/analyzer/{YYYYMMDD}/ (예: us_investment_report.md, kr_investment_report.md)
- 출력: output/portfolio/{YYYYMMDD}/ (예: us_final_recommendation.md, kr_final_recommendation.md)

Usage:
    python portfolio_maker_adk.py                                  # 가장 최근 analyzer 결과 처리
    python portfolio_maker_adk.py output/analyzer/20251222         # 특정 analyzer 폴더 처리
    python portfolio_maker_adk.py --text-mode                      # 보고서 내용을 프롬프트에 직접 삽입 (파일 첨부 대신 텍스트로)

Environment Variables:
    GOOGLE_API_KEY 또는 GEMINI_API_KEY: Gemini API 키
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime
from typing import Optional

# .env 파일 로드
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


try:
    from google.adk.agents import Agent  # type: ignore[import-not-found]
    from google.adk.tools import google_search  # type: ignore[import-not-found]
except ImportError:
    print("❌ google-adk 패키지가 설치되어 있지 않습니다.")
    print("   다음 명령어로 설치해주세요: pip install google-adk")
    sys.exit(1)


from adk_utils import AdkAgentRunner, print_runtime_llm_config, _ensure_google_api_key

# 파일 첨부용 GenAI
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass # 의존성 체크는 아래에서

try:
    from portfolio_maker import (
        DEFAULT_MODEL,
        FINAL_RECOMMENDATION_FILENAME,
        INPUT_REPORT_FILENAME,
        MARKET_INFO,
        create_portfolio_output_dir,
        get_latest_analyzer_dir,
    )
except Exception as e:
    print("❌ 기존 portfolio_maker 모듈 임포트 실패:", str(e))
    print("   requirements.txt의 의존성(google-genai 등)을 설치했는지 확인해주세요.")
    sys.exit(1)

ADK_DEFAULT_MODEL = os.environ.get("MARKET_LENS_ADK_MODEL", "gemini-2.5-flash")


# =============================================================================
# PortfolioMakerADK
# =============================================================================


class PortfolioMakerADK:
    """ADK 기반 포트폴리오 추천 생성기 (기존 포맷 유지)"""

    def __init__(self, model: str = ADK_DEFAULT_MODEL, use_tools: bool = True):
        # ADK는 tools를 function calling으로 실행하므로, tool 지원 모델 사용을 권장
        self.model = model or ADK_DEFAULT_MODEL
        self.use_tools = use_tools
        
        # 파일 업로드를 위한 GenAI 클라이언트 초기화
        _ensure_google_api_key()
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        if self.use_tools:
            # 파일 첨부 방식을 사용하므로 파일 읽기 도구는 불필요하고, 구글 서치만 사용
            tools = [google_search]
        else:
            tools = []

        self._agent = Agent(
            name="market_lens_portfolio_advisor",
            model=self.model,
            instruction=(
                "너는 월스트리트의 시니어 포트폴리오 매니저다. "
                "제공된 investment_report.md(시장별 종합 보고서)를 기반으로 "
                "최종 추천 종목/예산 분배/매수 전략/리스크 관리/최종 조언을 한국어로 작성한다. "
                "필요하면 google_search로 최신 시장 정보를 확인한다. "
                "분석 보고서는 파일로 첨부되어 있거나 프롬프트에 포함된다."
            ),
            description="Market Lens AI - Portfolio Advisor Agent (ADK)",
            # AFC를 위해 python callable을 그대로 전달
            tools=tools,
        )
        self._runner = AdkAgentRunner(agent=self._agent, app_name="market_lens_ai")

    def _get_recommendation_request_text(self) -> str:
        return """## 요청 사항:
위 분석 보고서를 바탕으로 다음 내용을 포함한 **최종 투자 추천 보고서**를 작성해줘:

### 1. 최종 추천 종목 요약
- 분석된 종목들 중 최종 추천 순위 결정
- 각 종목의 핵심 투자 포인트 3줄 요약
- 추천 등급 (5점 만점 척도, 예시: 4.0/5.0, 3.0/5.0, 2.0/5.0, 1.0/5.0, 0.0/5.0)

### 2. 포트폴리오 예산 분배 전략
최종 추천된 종목들을 바탕으로 포트폴리오 예산 분배 전략을 제시해줘:

### 3. 매수 전략
최종 추천된 종목들을 바탕으로 아래 매수 전략을 제시해줘:
- 분할 매수 vs 일괄 매수 권고
- 목표가 및 손절가 제시
- 최적 매수 타이밍

### 4. 리스크 관리 방안
- 포트폴리오 전체 리스크 평가
- 헤지 전략 제안
- 리밸런싱 주기 권고

### 5. 최종 투자 조언
- 현재 시장 상황을 고려한 종합적인 투자 의견
- 주의해야 할 거시경제 이벤트
- 모니터링해야 할 핵심 지표

---

보고서는 마크다운 형식으로 작성하고, 전문적이면서도 이해하기 쉽게 작성해줘.
모든 내용은 한국어로 작성해줘.
가장 최신 시장 정보를 반영하여 현실적이고 실행 가능한 조언을 제공해줘."""

    def _create_prompt_for_file_attachment(self) -> str:
        base_prompt = f"""너는 월스트리트의 시니어 포트폴리오 매니저 'Gemini Portfolio Advisor'야.
첨부된 파일(투자 분석 보고서)을 면밀히 검토하고, 이를 바탕으로 투자자에게 최종 추천 종목과 포트폴리오 예산 분배 전략을 제시해야 해.

---

"""
        return base_prompt + self._get_recommendation_request_text()

    def _create_prompt_with_content(self, report_content: str) -> str:
        base_prompt = f"""너는 월스트리트의 시니어 포트폴리오 매니저 'Gemini Portfolio Advisor'야.
아래에 제공된 투자 분석 보고서를 면밀히 검토하고, 투자자에게 최종 추천 종목과 포트폴리오 예산 분배 전략을 제시해야 해.

## 분석 보고서 내용:
{report_content}

---

"""
        return base_prompt + self._get_recommendation_request_text()

    def generate_recommendation(
        self,
        analyzer_dir: str,
        use_text_mode: bool = False,
        input_filename: str = INPUT_REPORT_FILENAME,
        output_filename: str = FINAL_RECOMMENDATION_FILENAME,
        portfolio_output_dir: str = None,
    ) -> tuple[Optional[str], str]:
        report_path = os.path.join(analyzer_dir, input_filename)
        if not os.path.exists(report_path):
            print(f"  ⚠️ {input_filename} 파일을 찾을 수 없습니다: {report_path}")
            return None, ""

        if portfolio_output_dir is None:
            portfolio_output_dir = create_portfolio_output_dir()
        else:
            os.makedirs(portfolio_output_dir, exist_ok=True)
            
        print("\n🎯 (ADK) 최종 투자 추천 보고서 생성 중...")
        print(f"  📂 입력 디렉토리: {analyzer_dir}")
        print(f"  📁 출력 디렉토리: {portfolio_output_dir}")
        print(f"  🤖 모델: {self.model}")
        print(f"  📎 분석 방식: {'텍스트 삽입(Context Injection)' if use_text_mode else '파일 첨부(File API)'}")

        uploaded_file = None
        try:
            # use_text_mode가 True면 텍스트 삽입, 아니면 파일 첨부
            if use_text_mode:
                with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                    report_content = f.read()
                prompt = self._create_prompt_with_content(report_content)
                new_message = None # 일반 텍스트 모드는 prompt만 넘김 (또는 new_message로 감싸도 됨)
            else:
                # 파일 첨부 모드
                print(f"  📤 파일 업로드 중: {input_filename} ...")
                uploaded_file = self.client.files.upload(
                    file=report_path,
                    config=types.UploadFileConfig(mime_type='text/markdown')
                )
                
                prompt = self._create_prompt_for_file_attachment()
                
                # Content 구성 (프롬프트 + 파일)
                parts = [
                    types.Part(text=prompt),
                    types.Part(
                        file_data=types.FileData(
                            mime_type=uploaded_file.mime_type,
                            file_uri=uploaded_file.uri
                        )
                    )
                ]
                new_message = types.Content(role="user", parts=parts)

            # 실행
            if new_message:
                recommendation_text = self._runner.run_text(prompt="", new_message=new_message)
            else:
                recommendation_text = self._runner.run_text(prompt)

            if not recommendation_text:
                raise ValueError("ADK 응답에서 텍스트를 추출하지 못했습니다.")

            final_report = f"""# 🎯 최종 투자 추천 및 포트폴리오 전략 보고서

> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 기반 보고서: {analyzer_dir}/{input_filename}
> 분석 모델: {self.model}
> 분석 방식: {'텍스트 삽입' if use_text_mode else '파일 첨부(File API)'}

---

{recommendation_text}

---

⚠️ **면책조항**: 본 보고서는 AI가 생성한 참고 자료이며, 투자 권유가 아닙니다.
실제 투자 결정은 추가적인 조사와 전문가 상담을 권장합니다.
투자의 책임은 전적으로 투자자 본인에게 있습니다.
"""

            out_path = os.path.join(portfolio_output_dir, output_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"  ✅ 저장: {out_path}")
            
            # 파일 삭제
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except: pass

            return final_report, portfolio_output_dir

        except Exception as e:
            print(f"  ⚠️ (ADK) 최종 추천 보고서 생성 실패: {str(e)}")
            # 에러 시에도 파일 삭제 시도
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except: pass
            return None, ""

    def generate_all_recommendations(
        self,
        analyzer_dir: str,
        use_text_mode: bool = False,
        portfolio_output_dir: str = None,
    ) -> tuple[dict, str]:
        if portfolio_output_dir is None:
            portfolio_output_dir = create_portfolio_output_dir()
        else:
            os.makedirs(portfolio_output_dir, exist_ok=True)

        report_files = glob.glob(os.path.join(analyzer_dir, "*investment_report.md"))
        if not report_files:
            print(f"❌ 분석 보고서를 찾을 수 없습니다: {analyzer_dir}")
            return {}, ""

        results: dict = {}

        for report_file in report_files:
            filename = os.path.basename(report_file)

            market = None
            for m in MARKET_INFO.keys():
                if filename.startswith(f"{m}_"):
                    market = m
                    break

            if market is None and filename == INPUT_REPORT_FILENAME:
                market = "default"
            elif market is None:
                continue

            market_info = MARKET_INFO.get(market, {})
            market_name = market_info.get("name", market)

            print(f"\n{'='*60}")
            print(f"🎯 [{market_name}] (ADK) 최종 추천 보고서 생성")
            print(f"{'='*60}")

            if market == "default":
                out_name = FINAL_RECOMMENDATION_FILENAME
            else:
                out_name = f"{market}_{FINAL_RECOMMENDATION_FILENAME}"

            result, _ = self.generate_recommendation(
                analyzer_dir=analyzer_dir,
                use_text_mode=use_text_mode,
                input_filename=filename,
                output_filename=out_name,
                portfolio_output_dir=portfolio_output_dir,
            )
            if result:
                results[market] = result

        return results, portfolio_output_dir


# =============================================================================
# 메인 실행 (기존 CLI 호환)
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="(ADK) LLM 기반 최종 투자 추천 및 포트폴리오 전략 보고서 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_maker_adk.py                                   # 가장 최근 analyzer 결과의 모든 시장 분석
  python portfolio_maker_adk.py output/analyzer/20251222          # 특정 analyzer 폴더 분석
  python portfolio_maker_adk.py --text-mode                       # 텍스트 삽입 방식 (파일 첨부 미사용)
        """,
    )
    parser.add_argument(
        "analyzer_dir",
        nargs="?",
        default=None,
        help="분석 보고서가 있는 디렉토리 (기본값: 가장 최근 output/analyzer 폴더)",
    )
    parser.add_argument(
        "--text-mode",
        "-t",
        action="store_true",
        help="텍스트 삽입 방식으로 분석 (기본값: 파일 첨부(File API) 방식)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ADK_DEFAULT_MODEL,
        help=f"사용할 Gemini 모델 (기본값: {ADK_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="현재 선택된 모델/endpoint(Dev API vs Vertex AI) 설정을 출력",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="(실험) ADK tool/function calling 사용(모델/엔드포인트가 tool 사용을 지원해야 함)",
    )

    args = parser.parse_args()

    analyzer_dir = args.analyzer_dir or get_latest_analyzer_dir()
    if not analyzer_dir or not os.path.exists(analyzer_dir):
        print("❌ 분석 보고서 디렉토리를 찾을 수 없습니다.")
        print("   사용법: python portfolio_maker_adk.py [analyzer_directory]")
        sys.exit(1)

    report_files = glob.glob(os.path.join(analyzer_dir, "*investment_report.md"))
    print("=" * 60)
    print("🎯 (ADK) 포트폴리오 추천 보고서 생성기")
    print("=" * 60)
    print(f"📂 분석 보고서 디렉토리: {analyzer_dir}")
    print(f"📄 발견된 보고서: {len(report_files)}개")
    for f in report_files:
        print(f"   - {os.path.basename(f)}")
    print(f"🤖 모델: {args.model}")
    print(f"📎 분석 방식: {'텍스트 삽입(Context Injection)' if args.text_mode else '파일 첨부(File API)'}")
    print("=" * 60)

    if args.debug_config:
        print_runtime_llm_config(model=args.model, tools=[google_search])

    maker = PortfolioMakerADK(model=args.model, use_tools=args.use_tools)
    results, portfolio_output_dir = maker.generate_all_recommendations(
        analyzer_dir,
        use_text_mode=args.text_mode,
    )

    if results and portfolio_output_dir:
        print("\n" + "=" * 60)
        print("✅ (ADK) 최종 추천 보고서 생성 완료!")
        print(f"📁 보고서 위치: {portfolio_output_dir}")
        for market in results.keys():
            market_info = MARKET_INFO.get(market, {})
            market_name = market_info.get("name", market)
            if market == "default":
                filename = FINAL_RECOMMENDATION_FILENAME
            else:
                filename = f"{market}_{FINAL_RECOMMENDATION_FILENAME}"
            print(f"   - [{market_name}] {filename}")
        print("=" * 60)
    else:
        print("❌ (ADK) 최종 추천 보고서 생성 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
