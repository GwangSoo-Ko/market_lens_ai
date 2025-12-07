"""
Stock Screener - 4가지 투자 전략 기반 글로벌 스크리너

TradingView Screener를 활용한 4가지 투자 전략:
1. Cyclical (경기민감형) - 저 PBR, 저 EV/EBITDA
2. Growth (고성장형) - 높은 매출 성장률, 저 PEG
3. Finance (금융/자산주) - 극저 PBR, 높은 ROE, 배당
4. Defensive (경기방어주) - 안정적 영업이익률, FCF, 배당

Usage:
    python stock_screener.py
    
    또는 모듈로 임포트:
    from stock_screener import screen_growth, screen_defensive
"""

import os
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

# 최소 애널리스트 수 (신뢰도 기준)
MIN_ANALYST_COUNT = 3

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
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    count, df = (
        Query()
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
            col('recommendation_total') >= MIN_ANALYST_COUNT,
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
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    count, df = (
        Query()
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
            col('recommendation_total') >= MIN_ANALYST_COUNT,
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
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    count, df = (
        Query()
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
            col('recommendation_total') >= MIN_ANALYST_COUNT,
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
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        limit: 최대 조회 수
        
    Returns:
        (전체 조건 만족 종목 수, 필터링된 DataFrame)
    """
    count, df = (
        Query()
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
            col('recommendation_total') >= MIN_ANALYST_COUNT,
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
    filter_sector: bool = True,
    min_analyst_score: float = ANALYST_SCORE_BUY
) -> Dict[str, pd.DataFrame]:
    """
    모든 전략의 스크리너 실행
    
    Parameters:
        filter_sector: 섹터 필터링 적용 여부
        min_analyst_score: 최소 애널리스트 점수
        
    Returns:
        전략별 DataFrame 딕셔너리
    """
    results = {}
    
    print("📊 스크리닝 시작...")
    print("-" * 60)
    
    # Cyclical
    count, df = screen_cyclical(filter_sector, min_analyst_score)
    results['cyclical'] = df
    print(f"  • Cyclical (경기민감형): {count}개 중 {len(df)}개 필터링됨")
    
    # Growth
    count, df = screen_growth(filter_sector, min_analyst_score)
    results['growth'] = df
    print(f"  • Growth (고성장형): {count}개 중 {len(df)}개 필터링됨")
    
    # Finance
    count, df = screen_finance(filter_sector, min_analyst_score)
    results['finance'] = df
    print(f"  • Finance (금융/자산주): {count}개 중 {len(df)}개 필터링됨")
    
    # Defensive
    count, df = screen_defensive(filter_sector, min_analyst_score)
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
    prefix: str = 'global'
) -> List[str]:
    """
    스크리닝 결과를 CSV 파일로 저장
    
    Parameters:
        results: 전략별 DataFrame 딕셔너리
        output_dir: 저장 디렉토리 (None이면 output/{timestamp} 자동 생성)
        prefix: 파일명 접두사
        
    Returns:
        저장된 파일명 리스트
    """
    # 출력 디렉토리 설정 (없으면 자동 생성)
    if output_dir is None:
        output_dir = create_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    print(f"📂 출력 디렉토리: {output_dir}")
    
    for strategy, df in results.items():
        if not df.empty:
            filename = os.path.join(output_dir, f'{prefix}_{strategy}.csv')
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            saved_files.append(filename)
            print(f"  ✅ 저장: {filename}")
    
    return saved_files


def print_summary(results: Dict[str, pd.DataFrame]) -> None:
    """
    스크리닝 결과 요약 출력
    
    Parameters:
        results: 전략별 DataFrame 딕셔너리
    """
    print("=" * 60)
    print("📊 4가지 투자 전략 스크리닝 결과 요약")
    print("=" * 60)
    print(f"  • Cyclical (경기민감형): {len(results.get('cyclical', []))}개 종목")
    print(f"  • Growth (고성장형): {len(results.get('growth', []))}개 종목")
    print(f"  • Finance (금융/자산주): {len(results.get('finance', []))}개 종목")
    print(f"  • Defensive (경기방어주): {len(results.get('defensive', []))}개 종목")
    print("=" * 60)


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    # 모든 스크리너 실행
    results = run_all_screeners(filter_sector=True)
    
    # 결과 요약
    print_summary(results)
    
    # 결과 저장
    print("\n📁 결과 저장 중...")
    save_results(results)
    
    return results


if __name__ == "__main__":
    main()

