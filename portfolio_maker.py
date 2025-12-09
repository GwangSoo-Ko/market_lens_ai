"""
Portfolio Maker - LLM 기반 최종 투자 추천 및 포트폴리오 전략 보고서 생성기

Gemini API를 활용하여 investment_report.md를 분석하고 
최종 추천 종목 및 포트폴리오 예산 분배 전략을 제시합니다.

- 입력: output/analyzer/{timestamp}/ (분석 MD 보고서)
- 출력: output/portfolio/{timestamp}/ (포트폴리오 추천 보고서)

Usage:
    python portfolio_maker.py                                   # 가장 최근 analyzer 결과 분석
    python portfolio_maker.py output/analyzer/20251204_151114   # 특정 analyzer 폴더 분석
    python portfolio_maker.py --text-mode                       # 텍스트 삽입 방식으로 분석
    
    또는 모듈로 임포트:
    from portfolio_maker import PortfolioMaker
    maker = PortfolioMaker()
    maker.generate_recommendation('output/analyzer/20251204_151114')

Environment Variables:
    GOOGLE_API_KEY 또는 GEMINI_API_KEY: Gemini API 키
    (.env 파일에 설정하거나 환경변수로 설정 가능)
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv가 없어도 환경변수로 동작 가능
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ google-genai 패키지가 설치되어 있지 않습니다.")
    print("   다음 명령어로 설치해주세요: pip install google-genai")
    sys.exit(1)


# =============================================================================
# 상수 정의
# =============================================================================

OUTPUT_BASE_DIR = 'output'
ANALYZER_OUTPUT_DIR = 'output/analyzer'   # 분석 결과 읽기 경로
PORTFOLIO_OUTPUT_DIR = 'output/portfolio' # 포트폴리오 결과 저장 경로

# 기본 Gemini 모델
DEFAULT_MODEL = 'gemini-3-pro-preview'

# 입력 보고서 파일명
INPUT_REPORT_FILENAME = 'investment_report.md'

# 최종 추천 보고서 파일명
FINAL_RECOMMENDATION_FILENAME = 'final_recommendation.md'

# 시장 정보
MARKET_INFO = {
    'us': {'name': '미국', 'currency': 'USD'},
    'kr': {'name': '한국', 'currency': 'KRW'},
}


# =============================================================================
# PortfolioMaker 클래스
# =============================================================================

class PortfolioMaker:
    """LLM 기반 포트폴리오 추천 생성기"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        PortfolioMaker 초기화
        
        Parameters:
            api_key: Gemini API 키 (None이면 환경변수에서 읽음)
            model: 사용할 Gemini 모델명
        """
        self.model = model
        self.client = self._init_client(api_key)
        
    def _init_client(self, api_key: Optional[str] = None) -> genai.Client:
        """Gemini 클라이언트 초기화"""
        if api_key is None:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API 키가 설정되지 않았습니다.\n"
                "환경변수 GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정하거나,\n"
                "생성자에 api_key 파라미터를 전달해주세요."
            )
        
        return genai.Client(api_key=api_key)
    
    def _get_recommendation_request_text(self) -> str:
        """최종 추천 요청 텍스트 (공통)"""
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
    
    def _create_prompt_for_file(self) -> str:
        """파일 업로드 방식용 프롬프트 (파일이 별도로 첨부됨)"""
        base_prompt = """너는 월스트리트의 시니어 포트폴리오 매니저 'Gemini Portfolio Advisor'야. 
첨부된 투자 분석 보고서(investment_report.md)를 면밀히 검토하고, 투자자에게 최종 추천 종목과 포트폴리오 예산 분배 전략을 제시해야 해.

"""
        return base_prompt + self._get_recommendation_request_text()
    
    def _create_prompt_with_content(self, report_content: str) -> str:
        """텍스트 삽입 방식용 프롬프트 (보고서 내용이 프롬프트에 포함됨)"""
        base_prompt = f"""너는 월스트리트의 시니어 포트폴리오 매니저 'Gemini Portfolio Advisor'야. 
아래에 제공된 투자 분석 보고서를 면밀히 검토하고, 투자자에게 최종 추천 종목과 포트폴리오 예산 분배 전략을 제시해야 해.

## 분석 보고서 내용:
{report_content}

---

"""
        return base_prompt + self._get_recommendation_request_text()
    
    def _upload_file(self, file_path: str, display_name: str = None) -> Optional[object]:
        """
        파일을 Gemini API에 업로드
        
        Parameters:
            file_path: 업로드할 파일 경로
            display_name: 파일 표시 이름 (선택)
            
        Returns:
            업로드된 파일 객체 (실패 시 None)
        """
        try:
            upload_config = types.UploadFileConfig(
                display_name=display_name or os.path.basename(file_path),
                mime_type='text/markdown'
            )
            
            uploaded_file = self.client.files.upload(
                file=file_path,
                config=upload_config
            )
            
            print(f"  📤 파일 업로드 완료: {uploaded_file.name}")
            
            # 파일 처리 상태 확인 (PROCESSING 상태일 경우 대기)
            while hasattr(uploaded_file, 'state') and uploaded_file.state == 'PROCESSING':
                print(f"  ⏳ 파일 처리 중...")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            return uploaded_file
            
        except Exception as e:
            print(f"  ⚠️ 파일 업로드 실패: {str(e)}")
            return None
    
    def _delete_file(self, file_obj: object) -> None:
        """업로드된 파일 삭제 (정리용)"""
        try:
            if file_obj and hasattr(file_obj, 'name'):
                self.client.files.delete(name=file_obj.name)
                print(f"  🗑️ 업로드된 파일 정리 완료")
        except Exception:
            pass  # 삭제 실패해도 무시
    
    def generate_recommendation(
        self, 
        analyzer_dir: str, 
        use_file_upload: bool = True,
        input_filename: str = INPUT_REPORT_FILENAME,
        output_filename: str = FINAL_RECOMMENDATION_FILENAME,
        portfolio_output_dir: str = None
    ) -> tuple[Optional[str], str]:
        """
        투자 보고서를 기반으로 최종 추천 및 포트폴리오 분배 전략 생성
        
        Parameters:
            analyzer_dir: 분석 보고서가 저장된 디렉토리 경로 (output/analyzer/{timestamp})
            use_file_upload: True면 파일 업로드 방식, False면 텍스트 삽입 방식
            input_filename: 입력 보고서 파일명 (기본값: investment_report.md)
            output_filename: 출력 보고서 파일명 (기본값: final_recommendation.md)
            portfolio_output_dir: 포트폴리오 결과 저장 디렉토리 (None이면 자동 생성)
            
        Returns:
            (생성된 추천 보고서 내용, 저장 디렉토리) - 실패 시 (None, "")
        """
        report_path = os.path.join(analyzer_dir, input_filename)
        
        if not os.path.exists(report_path):
            print(f"  ⚠️ {input_filename} 파일을 찾을 수 없습니다: {report_path}")
            return None, ""
        
        # 포트폴리오 출력 디렉토리 생성
        if portfolio_output_dir is None:
            portfolio_output_dir = create_portfolio_output_dir()
        else:
            os.makedirs(portfolio_output_dir, exist_ok=True)
        
        print("\n🎯 최종 투자 추천 보고서 생성 중...")
        print(f"  📂 입력 디렉토리: {analyzer_dir}")
        print(f"  📁 출력 디렉토리: {portfolio_output_dir}")
        
        uploaded_file = None
        
        try:
            # Google Search 도구 설정
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            if use_file_upload:
                # 방법 1: 파일 업로드 방식 (토큰 제한 우회)
                print("  📎 파일 업로드 방식으로 분석 진행...")
                
                uploaded_file = self._upload_file(
                    report_path, 
                    display_name='investment_report'
                )
                
                if not uploaded_file:
                    print("  ⚠️ 파일 업로드 실패, 텍스트 삽입 방식으로 전환...")
                    use_file_upload = False
                else:
                    # 파일 첨부 방식 프롬프트
                    prompt_text = self._create_prompt_for_file()
                    
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[uploaded_file, prompt_text],
                        config=types.GenerateContentConfig(
                            tools=[google_search_tool],
                            temperature=0,
                            max_output_tokens=60000,
                        )
                    )
            
            if not use_file_upload:
                # 방법 2: 텍스트 삽입 방식 (기존 방식)
                print("  📝 텍스트 삽입 방식으로 분석 진행...")
                
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                prompt = self._create_prompt_with_content(report_content)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[google_search_tool],
                        temperature=0,
                        max_output_tokens=60000,
                    )
                )
            
            recommendation_text = response.text
            
            # 보고서에 헤더 추가
            final_report = f"""# 🎯 최종 투자 추천 및 포트폴리오 전략 보고서

> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 기반 보고서: {analyzer_dir}/{input_filename}
> 분석 모델: {self.model}
> 분석 방식: {'파일 업로드' if use_file_upload and uploaded_file else '텍스트 삽입'}

---

{recommendation_text}

---

⚠️ **면책조항**: 본 보고서는 AI가 생성한 참고 자료이며, 투자 권유가 아닙니다. 
실제 투자 결정은 추가적인 조사와 전문가 상담을 권장합니다.
투자의 책임은 전적으로 투자자 본인에게 있습니다.
"""
            
            # 파일 저장
            recommendation_path = os.path.join(portfolio_output_dir, output_filename)
            with open(recommendation_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            print(f"  ✅ 저장: {recommendation_path}")
            
            return final_report, portfolio_output_dir
            
        except Exception as e:
            print(f"  ⚠️ 최종 추천 보고서 생성 실패: {str(e)}")
            return None, ""
        
        finally:
            # 업로드된 파일 정리
            if uploaded_file:
                self._delete_file(uploaded_file)
    
    def generate_all_recommendations(
        self, 
        analyzer_dir: str, 
        use_file_upload: bool = True,
        portfolio_output_dir: str = None
    ) -> tuple[dict, str]:
        """
        모든 시장별 투자 보고서에 대한 최종 추천 생성
        
        Parameters:
            analyzer_dir: 분석 보고서가 저장된 디렉토리 경로 (output/analyzer/{timestamp})
            use_file_upload: True면 파일 업로드 방식, False면 텍스트 삽입 방식
            portfolio_output_dir: 포트폴리오 결과 저장 디렉토리 (None이면 자동 생성)
            
        Returns:
            (시장별 결과 딕셔너리, 저장 디렉토리)
        """
        import glob
        
        # 포트폴리오 출력 디렉토리 생성
        if portfolio_output_dir is None:
            portfolio_output_dir = create_portfolio_output_dir()
        else:
            os.makedirs(portfolio_output_dir, exist_ok=True)
        
        # investment_report.md 파일들 찾기
        report_files = glob.glob(os.path.join(analyzer_dir, '*investment_report.md'))
        
        if not report_files:
            print(f"❌ 분석 보고서를 찾을 수 없습니다: {analyzer_dir}")
            return {}, ""
        
        results = {}
        
        for report_file in report_files:
            filename = os.path.basename(report_file)
            
            # 시장 코드 추출 (us_investment_report.md -> us)
            market = None
            for m in MARKET_INFO.keys():
                if filename.startswith(f'{m}_'):
                    market = m
                    break
            
            # 시장 코드가 없으면 기본 파일 (단일 시장으로 간주)
            if market is None and filename == INPUT_REPORT_FILENAME:
                market = 'default'
            elif market is None:
                continue
            
            market_info = MARKET_INFO.get(market, {})
            market_name = market_info.get('name', market)
            
            print(f"\n{'='*60}")
            print(f"🎯 [{market_name}] 최종 추천 보고서 생성")
            print(f"{'='*60}")
            
            # 출력 파일명 설정
            if market == 'default':
                output_filename = FINAL_RECOMMENDATION_FILENAME
            else:
                output_filename = f'{market}_{FINAL_RECOMMENDATION_FILENAME}'
            
            # 추천 생성
            result, _ = self.generate_recommendation(
                analyzer_dir=analyzer_dir,
                use_file_upload=use_file_upload,
                input_filename=filename,
                output_filename=output_filename,
                portfolio_output_dir=portfolio_output_dir
            )
            
            if result:
                results[market] = result
        
        return results, portfolio_output_dir


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_latest_analyzer_dir(base_dir: str = ANALYZER_OUTPUT_DIR) -> Optional[str]:
    """가장 최근 analyzer 결과 디렉토리 반환"""
    if not os.path.exists(base_dir):
        return None
    
    subdirs = [
        os.path.join(base_dir, d) 
        for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    
    if not subdirs:
        return None
    
    # 수정 시간 기준 정렬
    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0]


def create_portfolio_output_dir(base_dir: str = PORTFOLIO_OUTPUT_DIR) -> str:
    """
    날짜 기반 포트폴리오 결과 출력 디렉토리 생성
    
    Parameters:
        base_dir: 기본 출력 디렉토리
        
    Returns:
        생성된 디렉토리 경로 (output/portfolio/{YYYYMMDD})
    """
    date_str = datetime.now().strftime('%Y%m%d')
    output_dir = os.path.join(base_dir, date_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """
    메인 실행 함수
    
    Usage:
        python portfolio_maker.py                                   # 가장 최근 analyzer 결과의 모든 시장 분석
        python portfolio_maker.py output/analyzer/20251204_151114   # 특정 analyzer 폴더 분석
        python portfolio_maker.py --text-mode                       # 텍스트 삽입 방식으로 분석
    """
    import argparse
    import glob as glob_module
    
    parser = argparse.ArgumentParser(
        description='LLM 기반 최종 투자 추천 및 포트폴리오 전략 보고서 생성기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_maker.py                                   # 가장 최근 analyzer 결과의 모든 시장 분석
  python portfolio_maker.py output/analyzer/20251204_151114   # 특정 analyzer 폴더 분석
  python portfolio_maker.py --text-mode                       # 텍스트 삽입 방식으로 분석

Directory Structure:
  입력: output/analyzer/{timestamp}/  (분석 MD 보고서)
         - us_investment_report.md (미국)
         - kr_investment_report.md (한국)
  출력: output/portfolio/{timestamp}/ (포트폴리오 추천 보고서)
         - us_final_recommendation.md (미국)
         - kr_final_recommendation.md (한국)
        """
    )
    parser.add_argument(
        'analyzer_dir', 
        nargs='?', 
        default=None,
        help='분석 보고서가 있는 디렉토리 (기본값: 가장 최근 output/analyzer 폴더)'
    )
    parser.add_argument(
        '--text-mode', '-t',
        action='store_true',
        help='텍스트 삽입 방식으로 분석 (기본값: 파일 업로드 방식)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'사용할 Gemini 모델 (기본값: {DEFAULT_MODEL})'
    )
    
    args = parser.parse_args()
    
    # analyzer 디렉토리 결정
    analyzer_dir = args.analyzer_dir or get_latest_analyzer_dir()
    
    if not analyzer_dir or not os.path.exists(analyzer_dir):
        print("❌ 분석 보고서 디렉토리를 찾을 수 없습니다.")
        print("   사용법: python portfolio_maker.py [analyzer_directory]")
        print("   예시: python portfolio_maker.py output/analyzer/20251204_151114")
        print(f"\n   힌트: 먼저 python stock_analyzer.py를 실행하여 분석 보고서를 생성하세요.")
        sys.exit(1)
    
    # 분석할 보고서 파일 확인
    report_files = glob_module.glob(os.path.join(analyzer_dir, '*investment_report.md'))
    
    print("=" * 60)
    print("🎯 포트폴리오 추천 보고서 생성기")
    print("=" * 60)
    print(f"📂 분석 보고서 디렉토리: {analyzer_dir}")
    print(f"📄 발견된 보고서: {len(report_files)}개")
    for f in report_files:
        print(f"   - {os.path.basename(f)}")
    print(f"🤖 모델: {args.model}")
    print(f"📎 분석 방식: {'텍스트 삽입' if args.text_mode else '파일 업로드'}")
    print("=" * 60)
    
    try:
        maker = PortfolioMaker(model=args.model)
        
        # 모든 시장의 보고서 처리
        results, portfolio_output_dir = maker.generate_all_recommendations(
            analyzer_dir,
            use_file_upload=not args.text_mode
        )
        
        if results and portfolio_output_dir:
            print("\n" + "=" * 60)
            print("✅ 최종 추천 보고서 생성 완료!")
            print(f"📁 보고서 위치: {portfolio_output_dir}")
            for market in results.keys():
                market_info = MARKET_INFO.get(market, {})
                market_name = market_info.get('name', market)
                if market == 'default':
                    filename = FINAL_RECOMMENDATION_FILENAME
                else:
                    filename = f'{market}_{FINAL_RECOMMENDATION_FILENAME}'
                print(f"   - [{market_name}] {filename}")
            print("=" * 60)
        else:
            print("❌ 최종 추천 보고서 생성 실패")
            sys.exit(1)
            
    except ValueError as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

