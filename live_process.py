"""
Live Process - 전체 투자 분석 파이프라인 실행기

stock_screener.py → stock_analyzer.py → portfolio_maker.py를 
순차적으로 실행하여 스크리닝부터 최종 포트폴리오 추천까지 자동화합니다.

Usage:
    python live_process.py                    # 기본 실행 (전략당 1개 종목)
    python live_process.py -m 3               # 전략당 3개 종목 분석
    python live_process.py --skip-screener    # 스크리닝 건너뛰기 (기존 결과 사용)
    python live_process.py --skip-portfolio   # 포트폴리오 추천 건너뛰기

Environment Variables:
    GOOGLE_API_KEY 또는 GEMINI_API_KEY: Gemini API 키
    (.env 파일에 설정하거나 환경변수로 설정 가능)
"""

import os
import sys
import argparse
from datetime import datetime

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_screener(market: str = None):
    """
    스크리닝 실행
    
    Parameters:
        market: 시장 코드 ('us', 'kr') 또는 None (모든 시장)
    """
    print("\n" + "=" * 70)
    print("📊 [1/3] 스크리닝 시작")
    print("=" * 70)
    
    from stock_screener import run_all_markets, run_all_screeners, save_results, print_summary
    
    if market is None:
        # 모든 시장 스크리닝
        all_results, saved_files = run_all_markets(
            markets=None,  # 모든 시장
            filter_sector=True
        )
    else:
        # 특정 시장만 스크리닝
        results = run_all_screeners(market=market, filter_sector=True)
        print_summary(results, market=market)
        
        print("\n📁 결과 저장 중...")
        saved_files = save_results(results, market=market)
    
    if not saved_files:
        print("❌ 스크리닝 결과가 없습니다.")
        return None
    
    # 저장된 디렉토리 반환
    screener_dir = os.path.dirname(saved_files[0])
    return screener_dir


def run_analyzer(screener_dir: str, max_stocks: int = 1):
    """분석 실행"""
    print("\n" + "=" * 70)
    print("🤖 [2/3] LLM 종목 분석 시작")
    print("=" * 70)
    
    from stock_analyzer import StockAnalyzer
    
    analyzer = StockAnalyzer()
    all_analyses, analyzer_dir = analyzer.run_analysis(
        screener_dir, 
        max_stocks_per_strategy=max_stocks
    )
    
    if not all_analyses or not analyzer_dir:
        print("❌ 분석 결과가 없습니다.")
        return None
    
    return analyzer_dir


def run_portfolio(analyzer_dir: str):
    """포트폴리오 추천 실행 (모든 시장)"""
    print("\n" + "=" * 70)
    print("🎯 [3/3] 포트폴리오 추천 생성 시작")
    print("=" * 70)
    
    from portfolio_maker import PortfolioMaker
    
    maker = PortfolioMaker()
    results, portfolio_dir = maker.generate_all_recommendations(analyzer_dir)
    
    if not results or not portfolio_dir:
        print("❌ 포트폴리오 추천 생성 실패.")
        return None
    
    return portfolio_dir


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='전체 투자 분석 파이프라인 실행기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_process.py                    # 기본 실행 (미국+한국 모두, 전략당 1개 종목)
  python live_process.py --market us        # 미국 주식만 스크리닝
  python live_process.py --market kr        # 한국 주식만 스크리닝
  python live_process.py -M kr -m 3         # 한국 주식만, 전략당 3개 종목 분석
  python live_process.py --skip-screener    # 스크리닝 건너뛰기 (기존 결과 사용)
  python live_process.py --skip-portfolio   # 포트폴리오 추천 건너뛰기

Pipeline:
  [1] stock_screener.py  → output/screener/{date}/
  [2] stock_analyzer.py  → output/analyzer/{date}/
  [3] portfolio_maker.py → output/portfolio/{date}/

Supported Markets:
  us  - 미국 (NASDAQ, NYSE, AMEX)
  kr  - 한국 (KOSPI, KOSDAQ)
        """
    )
    parser.add_argument(
        '--market', '-M',
        type=str,
        default=None,
        choices=['us', 'kr'],
        help='스크리닝할 시장 선택 (지정하지 않으면 미국+한국 모두 실행)'
    )
    parser.add_argument(
        '--max-stocks', '-m',
        type=int,
        default=1,
        help='전략당 최대 분석 종목 수 (기본값: 1)'
    )
    parser.add_argument(
        '--skip-screener',
        action='store_true',
        help='스크리닝 건너뛰기 (가장 최근 screener 결과 사용)'
    )
    parser.add_argument(
        '--skip-portfolio',
        action='store_true',
        help='포트폴리오 추천 건너뛰기'
    )
    parser.add_argument(
        '--screener-dir',
        type=str,
        default=None,
        help='사용할 screener 결과 디렉토리 (--skip-screener와 함께 사용)'
    )
    parser.add_argument(
        '--analyzer-dir',
        type=str,
        default=None,
        help='사용할 analyzer 결과 디렉토리 (analyzer만 건너뛸 때 사용)'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    if args.market is None:
        market_name = '미국 + 한국 (전체)'
    else:
        market_name = '미국' if args.market == 'us' else '한국'
    
    print("=" * 70)
    print("🚀 Market Lens AI - 전체 파이프라인 실행")
    print("=" * 70)
    print(f"⏰ 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 시장: {market_name}")
    print(f"📊 전략당 분석 종목 수: {args.max_stocks}")
    print("=" * 70)
    
    screener_dir = None
    analyzer_dir = None
    portfolio_dir = None
    
    try:
        # Step 1: 스크리닝
        if args.analyzer_dir:
            # analyzer 디렉토리가 지정된 경우 스크리닝과 분석 모두 건너뛰기
            print("\n⏭️ 스크리닝 및 분석 건너뛰기 (analyzer 디렉토리 사용)")
            analyzer_dir = args.analyzer_dir
        elif args.skip_screener:
            if args.screener_dir:
                screener_dir = args.screener_dir
            else:
                from stock_analyzer import get_latest_screener_dir
                screener_dir = get_latest_screener_dir()
            
            if not screener_dir or not os.path.exists(screener_dir):
                print("❌ 스크리닝 결과 디렉토리를 찾을 수 없습니다.")
                print("   --skip-screener 옵션을 제거하고 다시 실행하세요.")
                sys.exit(1)
            
            print(f"\n⏭️ 스크리닝 건너뛰기 (기존 결과 사용: {screener_dir})")
        else:
            screener_dir = run_screener(market=args.market)
            if not screener_dir:
                print("❌ 스크리닝 실패. 파이프라인을 종료합니다.")
                sys.exit(1)
        
        # Step 2: 분석
        if not analyzer_dir:
            analyzer_dir = run_analyzer(screener_dir, args.max_stocks)
            if not analyzer_dir:
                print("❌ 분석 실패. 파이프라인을 종료합니다.")
                sys.exit(1)
        
        # Step 3: 포트폴리오 추천
        if not args.skip_portfolio:
            portfolio_dir = run_portfolio(analyzer_dir)
            if not portfolio_dir:
                print("❌ 포트폴리오 추천 실패.")
        else:
            print("\n⏭️ 포트폴리오 추천 건너뛰기")
        
        # 완료 메시지
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("✅ 파이프라인 실행 완료!")
        print("=" * 70)
        print(f"⏱️ 총 소요 시간: {duration}")
        print()
        print("📁 결과 파일 위치:")
        if screener_dir:
            print(f"   • 스크리닝: {screener_dir}")
        if analyzer_dir:
            print(f"   • 분석:     {analyzer_dir}")
        if portfolio_dir:
            print(f"   • 포트폴리오: {portfolio_dir}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

