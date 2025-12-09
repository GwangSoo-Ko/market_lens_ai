"""
Stock Screener - 4가지 투자 전략 기반 주식 스크리너 (미국/한국)

TradingView Screener를 활용한 4가지 투자 전략:
1. Cyclical (경기민감형) - 저 PBR, 저 EV/EBITDA
2. Growth (고성장형) - 높은 매출 성장률, 저 PEG
3. Finance (금융/자산주) - 극저 PBR, 높은 ROE, 배당
4. Defensive (경기방어주) - 안정적 영업이익률, FCF, 배당

Usage:
    python stock_screener.py              # 기본: 미국 주식 스크리닝
    python stock_screener.py --market us  # 미국 주식 스크리닝
    python stock_screener.py --market kr  # 한국 주식 스크리닝
    python stock_screener.py -m kr        # 한국 주식 스크리닝 (단축)
    
    또는 모듈로 임포트:
    from stock_screener import run_all_screeners
    results = run_all_screeners(market='korea')
"""

import os
import argparse
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import pandas as pd
from tradingview_screener import Query, col


# =============================================================================
# 출력 디렉토리 설정
# =============================================================================

OUTPUT_BASE_DIR = 'output'
SCREENER_OUTPUT_DIR = 'output/screener'


# =============================================================================
# 시장 설정
# =============================================================================

# 지원하는 시장 목록
SUPPORTED_MARKETS = {
    'us': {
        'code': 'america',
        'name': '미국',
        'prefix': 'us',
        'min_analyst_count': 3,  # 미국은 애널리스트 커버리지가 넓음
        'currency': 'USD',
    },
    'kr': {
        'code': 'korea',
        'name': '한국',
        'prefix': 'kr',
        'min_analyst_count': 1,  # 한국은 애널리스트 커버리지가 상대적으로 적음
        'currency': 'KRW',
    },
}

# 기본 시장 (None이면 모든 시장 실행)
DEFAULT_MARKET = None


# =============================================================================
# 상수 정의
# =============================================================================

# 기술 등급 기준 (Technical Rating)
# Recommend.All: -1(Strong Sell) ~ 1(Strong Buy)
TECH_RATING_BUY = 0.1
TECH_RATING_STRONG_BUY = 0.5

# 애널리스트 평점 기준
# 가중 평균 점수 (-2 ~ 2 스케일)
ANALYST_SCORE_BUY = 0.5
ANALYST_SCORE_STRONG_BUY = 1.0

# 섹터 매핑 (TradingView 영문 섹터명)
SECTORS: Dict[str, List[str]] = {
    'cyclical': [
        'Process Industries', 'Non-Energy Minerals', 'Producer Manufacturing',
        'Consumer Durables', 'Energy Minerals', 'Electronic Technology',
    ],
    'growth': [
        'Technology Services', 'Health Services', 'Commercial Services', 'Health Technology',
    ],
    'finance': ['Finance'],
    'defensive': ['Consumer Non-Durables', 'Utilities', 'Communications'],
}

# 공통 선택 필드 (애널리스트 평점 관련)
ANALYST_FIELDS = [
    'recommendation_buy',
    'recommendation_over',
    'recommendation_hold',
    'recommendation_under',
    'recommendation_sell',
    'recommendation_total',
    'Recommend.All',
]


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_market_config(market: str) -> dict:
    """
    시장 설정 가져오기
    
    Parameters:
        market: 시장 코드 ('us', 'kr') 또는 전체 코드 ('america', 'korea')
        
    Returns:
        시장 설정 딕셔너리
    """
    # 단축 코드 또는 전체 코드 모두 지원
    if market in SUPPORTED_MARKETS:
        return SUPPORTED_MARKETS[market]
    
    # 전체 코드로 검색
    for key, config in SUPPORTED_MARKETS.items():
        if config['code'] == market:
            return config
    
    # 기본값 반환
    print(f"⚠️ 지원하지 않는 시장: {market}, 기본값(미국) 사용")
    return SUPPORTED_MARKETS[DEFAULT_MARKET]


def calculate_analyst_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    애널리스트 평점 계산 함수
    
    점수 = (2×strong_buy + 1×buy + 0×hold - 1×sell - 2×strong_sell) / total
    
    Parameters:
        df: 스크리닝 결과 DataFrame
        
    Returns:
        analyst_score, analyst_rating 컬럼이 추가된 DataFrame
    """
    if df.empty:
        return df
    
    required_cols = [
        'recommendation_buy', 'recommendation_over', 'recommendation_hold',
        'recommendation_under', 'recommendation_sell', 'recommendation_total'
    ]
    
    if not all(c in df.columns for c in required_cols):
        return df
    
    df = df.copy()
    
    for c in required_cols:
        df[c] = df[c].fillna(0)
    
    df['analyst_score'] = (
        2 * df['recommendation_buy'] +
        1 * df['recommendation_over'] +
        0 * df['recommendation_hold'] +
        -1 * df['recommendation_under'] +
        -2 * df['recommendation_sell']
    ) / df['recommendation_total'].replace(0, 1)
    
    def get_rating(score: float, total: float) -> str:
        if pd.isna(score) or total == 0:
            return 'N/A'
        elif score >= 1.0:
            return 'Strong Buy'
        elif score >= 0.5:
            return 'Buy'
        elif score >= -0.5:
            return 'Hold'
        elif score >= -1.0:
            return 'Sell'
        else:
            return 'Strong Sell'
    
    df['analyst_rating'] = df.apply(
        lambda r: get_rating(r['analyst_score'], r['recommendation_total']), 
        axis=1
    )
    
    return df


def filter_by_analyst(
    df: pd.DataFrame, 
    min_score: float = ANALYST_SCORE_BUY
) -> pd.DataFrame:
    """
    애널리스트 평점으로 필터링
    
    Parameters:
        df: 스크리닝 결과 DataFrame
        min_score: 최소 애널리스트 점수 (기본: 0.5 = Buy 이상)
        
    Returns:
        필터링된 DataFrame
    """
    if df.empty or 'analyst_score' not in df.columns:
        return df
    return df[df['analyst_score'] >= min_score]


def filter_by_sector(
    df: pd.DataFrame, 
    strategy: str
) -> pd.DataFrame:
    """
    전략에 맞는 섹터로 필터링
    
    Parameters:
        df: 스크리닝 결과 DataFrame
        strategy: 전략명 ('cyclical', 'growth', 'finance', 'defensive')
        
    Returns:
        섹터 필터링된 DataFrame
    """
    if df.empty or 'sector' not in df.columns:
        return df
    
    if strategy not in SECTORS:
        return df
    
    return df[df['sector'].isin(SECTORS[strategy])]


# =============================================================================
# 스크리너 함수
# =============================================================================

def screen_cyclical(
    market: str = DEFAULT_MARKET,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY,
    limit: int = 1000
) -> Tuple[int, pd.DataFrame]:
    """
    Cyclical (경기민감형) 스크리너
    
    목표: 자산 가치 대비 저평가되고, 현금 창출력이 좋은 기업
    
    조건:
        - PBR < 1 (자산가치 대비 저평가)
        - EV/EBITDA < 6 (현금 창출력 대비 저평가)
        - 유동비율 >= 1.5 (경기 침체 시 버틸 현금 체력)
        - 애널리스트/기술 등급 Buy 이상
    
    Parameters:
        market: 시장 코드 ('us', 'kr', 'america', 'korea')
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    market_config = get_market_config(market)
    min_analyst_count = market_config['min_analyst_count']
    
    count, df = (
        Query()
        .set_markets(market_config['code'])
        .select(
            'name', 'close', 'change', 'volume', 'market_cap_basic',
            'sector', 'industry',
            'price_book_fq',               # PBR
            'enterprise_value_ebitda_ttm', # EV/EBITDA
            'current_ratio_fq',            # 유동비율
            *ANALYST_FIELDS,
        )
        .where(
            col('is_primary') == True,
            col('price_book_fq') < 1,
            col('price_book_fq') > 0,
            col('enterprise_value_ebitda_ttm') < 6,
            col('enterprise_value_ebitda_ttm') > 0,
            col('current_ratio_fq') >= 1.5,
            col('recommendation_total') >= min_analyst_count,
            col('Recommend.All') >= TECH_RATING_BUY,
        )
        .order_by('enterprise_value_ebitda_ttm', ascending=True)
        .limit(limit)
        .get_scanner_data()
    )
    
    # 애널리스트 점수 계산 및 필터링
    df = calculate_analyst_score(df)
    df = filter_by_analyst(df, min_analyst_score)
    
    # 섹터 필터링
    if filter_sector:
        df = filter_by_sector(df, 'cyclical')
    
    return count, df


def screen_growth(
    market: str = DEFAULT_MARKET,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY,
    limit: int = 1000
) -> Tuple[int, pd.DataFrame]:
    """
    Growth (고성장형) 스크리너
    
    목표: 매출이 빠르게 늘면서, 성장성 대비 주가가 싼 기업
    
    조건:
        - 매출 성장률 YoY >= 20% (전년 대비 고속 성장)
        - PEG 비율 < 1 (성장률 감안 시 저평가)
        - 부채비율 < 150% (금리 리스크 관리)
        - 애널리스트/기술 등급 Buy 이상
    
    Parameters:
        market: 시장 코드 ('us', 'kr', 'america', 'korea')
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    market_config = get_market_config(market)
    min_analyst_count = market_config['min_analyst_count']
    
    count, df = (
        Query()
        .set_markets(market_config['code'])
        .select(
            'name', 'close', 'change', 'volume', 'market_cap_basic',
            'sector', 'industry',
            'total_revenue_yoy_growth_ttm',  # 매출 성장률 YoY
            'price_earnings_growth_ttm',     # PEG 비율
            'debt_to_equity_fq',             # 부채비율
            'earnings_per_share_diluted_yoy_growth_ttm',  # EPS 성장률
            *ANALYST_FIELDS,
        )
        .where(
            col('is_primary') == True,
            col('total_revenue_yoy_growth_ttm') >= 20,
            col('price_earnings_growth_ttm') < 1,
            col('price_earnings_growth_ttm') >= 0.1,
            col('debt_to_equity_fq') < 1.5,
            col('recommendation_total') >= min_analyst_count,
            col('Recommend.All') >= TECH_RATING_BUY,
        )
        .order_by('price_earnings_growth_ttm', ascending=True)
        .limit(limit)
        .get_scanner_data()
    )
    
    # 애널리스트 점수 계산 및 필터링
    df = calculate_analyst_score(df)
    df = filter_by_analyst(df, min_analyst_score)
    
    # 섹터 필터링
    if filter_sector:
        df = filter_by_sector(df, 'growth')
    
    return count, df


def screen_finance(
    market: str = DEFAULT_MARKET,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY,
    limit: int = 1000
) -> Tuple[int, pd.DataFrame]:
    """
    Finance (금융/자산주) 스크리너
    
    목표: 극도로 저평가된 자산과 높은 자본효율, 배당 매력
    
    조건:
        - PBR < 0.6 (절대적 저평가 영역)
        - ROE >= 10% (저평가지만 돈은 잘 버는 곳)
        - 배당수익률 >= 4% (확실한 현금 보상)
        - 애널리스트/기술 등급 Buy 이상
    
    Parameters:
        market: 시장 코드 ('us', 'kr', 'america', 'korea')
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    market_config = get_market_config(market)
    min_analyst_count = market_config['min_analyst_count']
    
    count, df = (
        Query()
        .set_markets(market_config['code'])
        .select(
            'name', 'close', 'change', 'volume', 'market_cap_basic',
            'sector', 'industry',
            'price_book_fq',              # PBR
            'return_on_equity_fq',        # ROE
            'dividend_yield_recent',      # 배당수익률
            *ANALYST_FIELDS,
        )
        .where(
            col('is_primary') == True,
            col('price_book_fq') < 0.6,
            col('price_book_fq') > 0,
            col('return_on_equity_fq') >= 10,
            col('dividend_yield_recent') >= 4,
            col('recommendation_total') >= min_analyst_count,
            col('Recommend.All') >= TECH_RATING_BUY,
        )
        .order_by('dividend_yield_recent', ascending=False)
        .limit(limit)
        .get_scanner_data()
    )
    
    # 애널리스트 점수 계산 및 필터링
    df = calculate_analyst_score(df)
    df = filter_by_analyst(df, min_analyst_score)
    
    # 섹터 필터링
    if filter_sector:
        df = filter_by_sector(df, 'finance')
    
    return count, df


def screen_defensive(
    market: str = DEFAULT_MARKET,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY,
    limit: int = 1000
) -> Tuple[int, pd.DataFrame]:
    """
    Defensive (경기방어주) 스크리너
    
    목표: 마진이 안정적이고, 현금이 잘 돌며 배당을 주는 기업
    
    조건:
        - 영업이익률 >= 5% (안정적인 마진 확보)
        - FCF > 0 (현금이 플러스인지 확인)
        - 배당수익률 >= 3% (은행 이자 이상의 수익)
        - 애널리스트/기술 등급 Buy 이상
    
    Parameters:
        market: 시장 코드 ('us', 'kr', 'america', 'korea')
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    market_config = get_market_config(market)
    min_analyst_count = market_config['min_analyst_count']
    
    count, df = (
        Query()
        .set_markets(market_config['code'])
        .select(
            'name', 'close', 'change', 'volume', 'market_cap_basic',
            'sector', 'industry',
            'operating_margin_ttm',       # 영업이익률
            'free_cash_flow_ttm',         # 잉여현금흐름
            'dividend_yield_recent',      # 배당수익률
            *ANALYST_FIELDS,
        )
        .where(
            col('is_primary') == True,
            col('operating_margin_ttm') >= 5,
            col('free_cash_flow_ttm') > 0,
            col('dividend_yield_recent') >= 3,
            col('recommendation_total') >= min_analyst_count,
            col('Recommend.All') >= TECH_RATING_BUY,
        )
        .order_by('dividend_yield_recent', ascending=False)
        .limit(limit)
        .get_scanner_data()
    )
    
    # 애널리스트 점수 계산 및 필터링
    df = calculate_analyst_score(df)
    df = filter_by_analyst(df, min_analyst_score)
    
    # 섹터 필터링
    if filter_sector:
        df = filter_by_sector(df, 'defensive')
    
    return count, df


# =============================================================================
# 전체 실행 함수
# =============================================================================

def run_all_screeners(
    market: str = DEFAULT_MARKET,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY
) -> Dict[str, pd.DataFrame]:
    """
    모든 전략의 스크리너 실행
    
    Parameters:
        market: 시장 코드 ('us', 'kr', 'america', 'korea')
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        
    Returns:
        전략별 DataFrame 딕셔너리
    """
    market_config = get_market_config(market)
    results = {}
    
    print(f"📊 스크리닝 시작... (시장: {market_config['name']})")
    print("-" * 60)
    
    # Cyclical
    count, df = screen_cyclical(market, filter_sector, min_analyst_score)
    results['cyclical'] = df
    print(f"  • Cyclical (경기민감형): {count}개 중 {len(df)}개 필터링됨")
    
    # Growth
    count, df = screen_growth(market, filter_sector, min_analyst_score)
    results['growth'] = df
    print(f"  • Growth (고성장형): {count}개 중 {len(df)}개 필터링됨")
    
    # Finance
    count, df = screen_finance(market, filter_sector, min_analyst_score)
    results['finance'] = df
    print(f"  • Finance (금융/자산주): {count}개 중 {len(df)}개 필터링됨")
    
    # Defensive
    count, df = screen_defensive(market, filter_sector, min_analyst_score)
    results['defensive'] = df
    print(f"  • Defensive (경기방어주): {count}개 중 {len(df)}개 필터링됨")
    
    print("-" * 60)
    
    return results


def create_output_dir(base_dir: str = SCREENER_OUTPUT_DIR) -> str:
    """
    날짜 기반 출력 디렉토리 생성
    
    Parameters:
        base_dir: 기본 출력 디렉토리
        
    Returns:
        생성된 디렉토리 경로 (output/screener/{YYYYMMDD})
    """
    date_str = datetime.now().strftime('%Y%m%d')
    output_dir = os.path.join(base_dir, date_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(
    results: Dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
    market: str = DEFAULT_MARKET
) -> List[str]:
    """
    스크리닝 결과를 CSV 파일로 저장
    
    Parameters:
        results: 전략별 DataFrame 딕셔너리
        output_dir: 저장 디렉토리 (None이면 output/{timestamp} 자동 생성)
        market: 시장 코드 (파일명 접두사로 사용)
        
    Returns:
        저장된 파일명 리스트
    """
    # 출력 디렉토리 설정 (없으면 자동 생성)
    if output_dir is None:
        output_dir = create_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    market_config = get_market_config(market)
    prefix = market_config['prefix']
    
    saved_files = []
    
    print(f"📂 출력 디렉토리: {output_dir}")
    
    for strategy, df in results.items():
        if not df.empty:
            filename = os.path.join(output_dir, f'{prefix}_{strategy}.csv')
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            saved_files.append(filename)
            print(f"  ✅ 저장: {filename}")
    
    return saved_files


def print_summary(results: Dict[str, pd.DataFrame], market: str) -> None:
    """
    스크리닝 결과 요약 출력
    
    Parameters:
        results: 전략별 DataFrame 딕셔너리
        market: 시장 코드
    """
    market_config = get_market_config(market)
    
    print("=" * 60)
    print(f"📊 4가지 투자 전략 스크리닝 결과 요약 ({market_config['name']} 시장)")
    print("=" * 60)
    print(f"  • Cyclical (경기민감형): {len(results.get('cyclical', []))}개 종목")
    print(f"  • Growth (고성장형): {len(results.get('growth', []))}개 종목")
    print(f"  • Finance (금융/자산주): {len(results.get('finance', []))}개 종목")
    print(f"  • Defensive (경기방어주): {len(results.get('defensive', []))}개 종목")
    print("=" * 60)


def run_all_markets(
    markets: Optional[List[str]] = None,
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], List[str]]:
    """
    여러 시장의 스크리너를 실행하고 결과 저장
    
    Parameters:
        markets: 스크리닝할 시장 리스트 (None이면 모든 시장)
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        
    Returns:
        (시장별 전략별 DataFrame 딕셔너리, 저장된 파일 리스트)
    """
    if markets is None:
        markets = list(SUPPORTED_MARKETS.keys())
    
    all_results = {}
    all_saved_files = []
    output_dir = create_output_dir()
    
    for market in markets:
        market_config = get_market_config(market)
        print(f"\n🌍 [{market_config['name']}] 시장 스크리닝")
        
        # 스크리닝 실행
        results = run_all_screeners(market, filter_sector, min_analyst_score)
        all_results[market] = results
        
        # 결과 요약
        print_summary(results, market)
        
        # 결과 저장
        saved_files = save_results(results, output_dir=output_dir, market=market)
        all_saved_files.extend(saved_files)
    
    return all_results, all_saved_files


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='4가지 투자 전략 기반 주식 스크리너 (미국/한국)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_screener.py              # 기본: 미국 + 한국 모두 스크리닝
  python stock_screener.py --market us  # 미국 주식만 스크리닝
  python stock_screener.py --market kr  # 한국 주식만 스크리닝
  python stock_screener.py -m kr        # 한국 주식만 스크리닝 (단축)

Supported Markets:
  us  - 미국 (NASDAQ, NYSE, AMEX)
  kr  - 한국 (KOSPI, KOSDAQ)
        """
    )
    parser.add_argument(
        '--market', '-m',
        type=str,
        default=None,
        choices=['us', 'kr'],
        help='스크리닝할 시장 선택 (지정하지 않으면 미국+한국 모두 실행)'
    )
    parser.add_argument(
        '--no-sector-filter',
        action='store_true',
        help='섹터 필터링 비활성화'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Market Lens AI - 주식 스크리너")
    print("=" * 60)
    
    if args.market is None:
        # 기본: 모든 시장 스크리닝
        market_names = ', '.join([cfg['name'] for cfg in SUPPORTED_MARKETS.values()])
        print(f"🌍 시장: {market_names} (전체)")
        print("=" * 60)
        
        all_results, saved_files = run_all_markets(
            markets=None,  # 모든 시장
            filter_sector=not args.no_sector_filter
        )
        
        # 전체 결과 요약
        print("\n" + "=" * 60)
        print("📊 전체 스크리닝 완료!")
        print("=" * 60)
        total_files = len(saved_files)
        print(f"📁 저장된 파일 수: {total_files}개")
        if saved_files:
            print(f"📂 출력 디렉토리: {os.path.dirname(saved_files[0])}")
        print("=" * 60)
        
        return all_results
    else:
        # 특정 시장만 스크리닝
        market_config = get_market_config(args.market)
        print(f"🌍 시장: {market_config['name']}")
        print(f"💰 통화: {market_config['currency']}")
        print(f"👥 최소 애널리스트 수: {market_config['min_analyst_count']}명")
        print("=" * 60)
        
        # 스크리닝 실행
        results = run_all_screeners(
            market=args.market,
            filter_sector=not args.no_sector_filter
        )
        
        # 결과 요약
        print_summary(results, args.market)
        
        # 결과 저장
        print("\n📁 결과 저장 중...")
        save_results(results, market=args.market)
        
        return results


if __name__ == "__main__":
    main()
