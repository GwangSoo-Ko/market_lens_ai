"""
Stock Analyzer - LLM 기반 주식 종합 분석 및 투자 조언 보고서 생성기

Gemini API를 활용하여 스크리닝된 종목들에 대한 심층 분석 보고서를 생성합니다.
- 입력: output/screener/{timestamp}/ (스크리닝 CSV 결과)
- 출력: output/analyzer/{timestamp}/ (분석 MD 보고서)

Usage:
    python stock_analyzer.py                                      # 가장 최근 screener 결과 분석
    python stock_analyzer.py output/screener/20251204_151114      # 특정 screener 폴더 분석
    python stock_analyzer.py -m 3                                 # 전략당 3개 종목 분석
    
    또는 모듈로 임포트:
    from stock_analyzer import StockAnalyzer
    analyzer = StockAnalyzer()
    analyzer.run_analysis('output/screener/20251204_151114')

Note:
    최종 추천 보고서는 portfolio_maker.py를 사용하세요:
    python portfolio_maker.py output/analyzer/20251204_151114

Environment Variables:
    GOOGLE_API_KEY 또는 GEMINI_API_KEY: Gemini API 키
    (.env 파일에 설정하거나 환경변수로 설정 가능)
"""

import os
import sys
import time
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

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

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance 패키지가 설치되어 있지 않습니다.")
    print("   다음 명령어로 설치해주세요: pip install yfinance")
    sys.exit(1)

try:
    import talib
except ImportError:
    print("❌ ta-lib(talib) 패키지가 설치되어 있지 않습니다.")
    print("   다음 명령어로 설치해주세요: pip install ta-lib")
    sys.exit(1)


# =============================================================================
# 상수 정의
# =============================================================================

OUTPUT_BASE_DIR = 'output'
SCREENER_OUTPUT_DIR = 'output/screener'  # 스크리닝 결과 읽기 경로
ANALYZER_OUTPUT_DIR = 'output/analyzer'  # 분석 결과 저장 경로
MARKET_DATA_OUTPUT_DIR = 'output/market_data'  # yfinance 기반 근거 데이터 저장 경로

# 시장 정보
MARKET_INFO = {
    'us': {'name': '미국', 'currency': 'USD'},
    'kr': {'name': '한국', 'currency': 'KRW'},
}

# 전략별 한글명 및 설명
STRATEGY_INFO = {
    'cyclical': {
        'name': '경기민감형 (Cyclical)',
        'description': '자산 가치 대비 저평가되고, 현금 창출력이 좋은 기업',
        'focus': 'PBR, EV/EBITDA, 유동비율',
    },
    'growth': {
        'name': '고성장형 (Growth)',
        'description': '매출이 빠르게 늘면서, 성장성 대비 주가가 싼 기업',
        'focus': '매출 성장률, PEG 비율, 부채비율, EPS 성장률',
    },
    'finance': {
        'name': '금융/자산주 (Finance)',
        'description': '극도로 저평가된 자산과 높은 자본효율, 배당 매력',
        'focus': 'PBR, ROE, 배당수익률',
    },
    'defensive': {
        'name': '경기방어주 (Defensive)',
        'description': '마진이 안정적이고, 현금이 잘 돌며 배당을 주는 기업',
        'focus': '영업이익률, FCF, 배당수익률',
    },
}

# 기본 Gemini 모델
DEFAULT_MODEL = 'gemini-3-flash-preview'

# API 호출 간 대기 시간 (초) - Rate Limiting 대응
API_DELAY = 1.0


# =============================================================================
# yfinance 기반 근거 데이터 생성 (CSV)
# =============================================================================

def _safe_str(value) -> str:
    return '' if value is None else str(value)


def _slugify_filename(value: str) -> str:
    value = _safe_str(value).strip()
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        value = value.replace(ch, '_')
    return value


def normalize_yfinance_ticker(raw_ticker: str, market: Optional[str]) -> List[str]:
    """
    스크리너 티커를 yfinance(Yahoo) 심볼 후보 리스트로 변환.
    - 미국: class 주식 등 'BRK.B' -> 'BRK-B' 보정 시도
    - 한국: 6자리 숫자 -> .KS/.KQ 순으로 시도
    """
    t = _safe_str(raw_ticker).strip()
    if not t:
        return []

    # TradingView 형식(예: KRX:005930) 대응
    if ':' in t:
        t = t.split(':', 1)[1]

    candidates: List[str] = []

    if market == 'us':
        candidates.append(t)
        if '.' in t:
            candidates.append(t.replace('.', '-'))
    elif market == 'kr':
        # 이미 suffix가 있으면 그대로
        if '.' in t:
            candidates.append(t)
        else:
            if t.isdigit() and len(t) == 6:
                candidates.extend([f'{t}.KS', f'{t}.KQ'])
            else:
                candidates.append(t)
    else:
        candidates.append(t)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _fetch_ohlcv_1y(ticker: yf.Ticker) -> pd.DataFrame:
    df = ticker.history(period='1y', interval='1d', auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # Date/Datetime 컬럼명을 통일
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    elif 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'date'})
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})

    # 지표 계산을 위해 날짜 오름차순 정렬
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

    # =============================================================================
    # TA-Lib 보조지표 추가
    # - SMA: 5, 20, 60, 120
    # - RSI: 14
    # - STOCH: (14, 3, 3) → slowk/slowd
    # - MFI: 14
    # - ATR: 14
    # =============================================================================
    required_cols = {'High', 'Low', 'Close', 'Volume'}
    if required_cols.issubset(df.columns):
        high = df['High'].astype(float).to_numpy()
        low = df['Low'].astype(float).to_numpy()
        close = df['Close'].astype(float).to_numpy()
        volume = df['Volume'].astype(float).to_numpy()

        for p in (5, 20, 60, 120):
            df[f'sma_{p}'] = talib.SMA(close, timeperiod=p)

        df['rsi_14'] = talib.RSI(close, timeperiod=14)

        stoch_k, stoch_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )
        df['stoch_k_14_3_3'] = stoch_k
        df['stoch_d_14_3_3'] = stoch_d

        df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
        df['macd_12_26_9'] = macd
        df['macd_signal_12_26_9'] = macd_signal
        df['macd_hist_12_26_9'] = macd_hist

        df['plus_di_14'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['minus_di_14'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)

    return df


def _as_long_statement(
    df: pd.DataFrame,
    statement: str,
    frequency: str,
    ticker_input: str,
    ticker_yfinance: str,
    currency: Optional[str],
) -> pd.DataFrame:
    """
    yfinance 재무 DataFrame(행=계정, 열=기간)을 LLM 친화적인 long 포맷으로 변환.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'ticker_input', 'ticker_yfinance', 'statement', 'frequency',
            'period_end', 'item', 'value', 'currency',
        ])

    wide = df.copy()
    wide.index = wide.index.astype(str)
    wide.columns = [pd.to_datetime(c).date().isoformat() if hasattr(c, 'date') else str(c) for c in wide.columns]

    # pandas 2.1+에서 stack 구현 변경으로 FutureWarning 발생 → future_stack=True 사용.
    # 단, future_stack=True에서는 dropna 인자를 함께 지정할 수 없음(예외 발생).
    # 구버전(pandas<2.1) 호환을 위해 실패 시 dropna=False로 fallback.
    try:
        stacked = wide.stack(future_stack=True)
    except TypeError:
        stacked = wide.stack(dropna=False)
    except Exception:
        stacked = wide.stack(dropna=False)

    long_df = (
        stacked
        .reset_index()
        .rename(columns={'level_0': 'item', 'level_1': 'period_end', 0: 'value'})
    )
    long_df.insert(0, 'ticker_input', ticker_input)
    long_df.insert(1, 'ticker_yfinance', ticker_yfinance)
    long_df.insert(2, 'statement', statement)
    long_df.insert(3, 'frequency', frequency)
    long_df['currency'] = currency
    return long_df


def _trim_last_n_periods(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = list(df.columns)
    # 가능한 경우 날짜로 정렬(최신 우선) 후 상위 n개 사용
    try:
        cols_dt = pd.to_datetime(cols, errors='coerce')
        order = (
            pd.Series(range(len(cols)), index=cols_dt)
            .sort_index(ascending=False)
            .tolist()
        )
        cols_sorted = [cols[i] for i in order if pd.notna(cols_dt[i])]
        # 변환 불가 컬럼이 섞여 있으면 원래 순서 유지로 fallback
        if len(cols_sorted) >= 1:
            cols = cols_sorted + [c for c in cols if c not in cols_sorted]
    except Exception:
        pass
    return df.loc[:, cols[:n]]


def _ttm_from_quarterly(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    분기 재무(열=기간)에서 최근 4개 분기 합으로 TTM 1개 열 생성.
    (yfinance trailing 데이터가 없을 때 fallback)
    """
    if df_quarterly is None or df_quarterly.empty:
        return pd.DataFrame()
    q = df_quarterly.copy()
    # yfinance는 최신 기간이 먼저 오는 경우가 많아 그대로 사용
    cols = list(q.columns)[:4]
    if len(cols) < 4:
        return pd.DataFrame()
    ttm_col_name = cols[0]  # 가장 최신 분기 end-date를 대표값으로 사용
    ttm = q.loc[:, cols].sum(axis=1, min_count=1).to_frame(name=ttm_col_name)
    return ttm


def fetch_and_save_market_data_for_stock(
    raw_ticker: str,
    market: Optional[str],
    output_dir: str,
    refresh: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    종목 1개에 대해:
    - 최근 1년 OHLCV(+ TA-Lib 보조지표) CSV
    - 최근 5개년(가능한 범위) + 분기 + TTM(가능하면 trailing, 아니면 4Q 합) 재무 CSV
    를 생성하고 경로를 반환.

    Returns:
        (resolved_yfinance_ticker, price_csv_path, financials_csv_path)
    """
    ticker_input = _safe_str(raw_ticker).strip()
    if not ticker_input:
        return None, None, None

    output_root = Path(output_dir)
    price_dir = output_root / 'prices'
    fin_dir = output_root / 'financials'
    price_dir.mkdir(parents=True, exist_ok=True)
    fin_dir.mkdir(parents=True, exist_ok=True)

    # 저장 파일명은 input ticker 기반으로 고정 (시장별 중복 대비)
    slug = _slugify_filename(f'{market}_{ticker_input}' if market else ticker_input)
    price_path = str(price_dir / f'{slug}_ohlcv_1y_ta.csv')
    fin_path = str(fin_dir / f'{slug}_financials_5y_ttm.csv')

    if not refresh and os.path.exists(price_path) and os.path.exists(fin_path):
        return ticker_input, price_path, fin_path

    resolved = None
    last_err = None
    for candidate in normalize_yfinance_ticker(ticker_input, market):
        try:
            t = yf.Ticker(candidate)

            # 가격 1년(+ 지표)
            ohlcv = _fetch_ohlcv_1y(t)
            if ohlcv is None or ohlcv.empty:
                raise ValueError(f'No OHLCV data for {candidate}')

            # 통화
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}
            currency = info.get('financialCurrency') or info.get('currency')

            # 재무(연간/분기/TTM)
            # income statement
            income_y = getattr(t, 'income_stmt', pd.DataFrame())
            income_q = getattr(t, 'quarterly_income_stmt', pd.DataFrame())
            if income_y is None or income_y.empty:
                income_y = getattr(t, 'financials', pd.DataFrame())
            if income_q is None or income_q.empty:
                income_q = getattr(t, 'quarterly_financials', pd.DataFrame())

            income_ttm = pd.DataFrame()
            try:
                if hasattr(t, 'get_income_stmt'):
                    income_ttm = t.get_income_stmt(freq='trailing')  # 최신 yfinance
            except Exception:
                income_ttm = pd.DataFrame()
            if income_ttm is None or income_ttm.empty:
                income_ttm = _ttm_from_quarterly(income_q)

            # cash flow
            cash_y = getattr(t, 'cash_flow', pd.DataFrame())
            cash_q = getattr(t, 'quarterly_cash_flow', pd.DataFrame())
            cash_ttm = pd.DataFrame()
            try:
                if hasattr(t, 'get_cash_flow'):
                    cash_ttm = t.get_cash_flow(freq='trailing')
            except Exception:
                cash_ttm = pd.DataFrame()
            if cash_ttm is None or cash_ttm.empty:
                cash_ttm = _ttm_from_quarterly(cash_q)

            # balance sheet (TTM 없음: 최신 분기/연간 제공)
            bal_y = getattr(t, 'balance_sheet', pd.DataFrame())
            bal_q = getattr(t, 'quarterly_balance_sheet', pd.DataFrame())

            # 최근 5개년(가능 범위)만 남김: yfinance는 종종 최신이 먼저 정렬되어 있음
            income_y = _trim_last_n_periods(income_y, 5)
            cash_y = _trim_last_n_periods(cash_y, 5)
            bal_y = _trim_last_n_periods(bal_y, 5)

            # 금융 long 포맷 통합
            fin_long_parts = [
                _as_long_statement(income_y, 'income_statement', 'yearly', ticker_input, candidate, currency),
                _as_long_statement(income_q, 'income_statement', 'quarterly', ticker_input, candidate, currency),
                _as_long_statement(income_ttm, 'income_statement', 'ttm', ticker_input, candidate, currency),
                _as_long_statement(cash_y, 'cash_flow', 'yearly', ticker_input, candidate, currency),
                _as_long_statement(cash_q, 'cash_flow', 'quarterly', ticker_input, candidate, currency),
                _as_long_statement(cash_ttm, 'cash_flow', 'ttm', ticker_input, candidate, currency),
                _as_long_statement(bal_y, 'balance_sheet', 'yearly', ticker_input, candidate, currency),
                _as_long_statement(bal_q, 'balance_sheet', 'quarterly', ticker_input, candidate, currency),
            ]
            fin_long = pd.concat([p for p in fin_long_parts if p is not None and not p.empty], ignore_index=True)

            # 저장
            ohlcv.to_csv(price_path, index=False, encoding='utf-8-sig')
            fin_long.to_csv(fin_path, index=False, encoding='utf-8-sig')

            resolved = candidate
            break
        except Exception as e:
            last_err = e
            continue

    if resolved is None:
        raise RuntimeError(f"yfinance 데이터 수집 실패: {ticker_input} (market={market}) / last_error={last_err}")

    return resolved, price_path, fin_path


# =============================================================================
# StockAnalyzer 클래스
# =============================================================================

class StockAnalyzer:
    """LLM 기반 주식 분석기"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        StockAnalyzer 초기화
        
        Parameters:
            api_key: Gemini API 키 (None이면 환경변수에서 읽음)
            model: 사용할 Gemini 모델명
        """
        self.model = model
        self.client = self._init_client(api_key)
        self.market_data_dir: Optional[str] = None

    def _extract_text_from_response(self, response: object) -> str:
        """
        google-genai 응답에서 텍스트만 안전하게 추출.
        response.text는 비텍스트 파트(thought_signature 등)가 섞이면 경고를 출력할 수 있어,
        candidates.content.parts[*].text를 직접 join하여 경고를 방지한다.
        """
        try:
            candidates = getattr(response, 'candidates', None) or []
            for cand in candidates:
                content = getattr(cand, 'content', None)
                parts = getattr(content, 'parts', None) or []
                texts = []
                for part in parts:
                    text = getattr(part, 'text', None)
                    if text:
                        texts.append(text)
                if texts:
                    return '\n'.join(texts).strip()
        except Exception:
            pass

        return ""
        
    def _init_client(self, api_key: Optional[str] = None) -> genai.Client:
        """Gemini 클라이언트 초기화"""
        # API 키 설정
        if api_key is None:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API 키가 설정되지 않았습니다.\n"
                "환경변수 GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정하거나,\n"
                "생성자에 api_key 파라미터를 전달해주세요."
            )
        
        return genai.Client(api_key=api_key)
    
    def load_screening_results(self, output_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        스크리닝 결과 CSV 파일들 로드 (시장별로 분리)
        
        Parameters:
            output_dir: 스크리닝 결과가 저장된 디렉토리 경로
            
        Returns:
            시장별 > 전략별 DataFrame 딕셔너리
            예: {'us': {'growth': df, ...}, 'kr': {'growth': df, ...}}
        """
        results = {}  # {market: {strategy: df}}
        
        # CSV 파일 패턴 매칭
        csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            filename_lower = filename.lower()
            
            # 파일명에서 시장 코드와 전략명 추출 (예: us_growth.csv -> us, growth)
            market = None
            for m in MARKET_INFO.keys():
                if filename_lower.startswith(f'{m}_'):
                    market = m
                    break
            
            # 시장 코드가 없으면 기존 global_ 형식으로 간주 (us로 처리)
            if market is None:
                if filename_lower.startswith('global_'):
                    market = 'us'
                else:
                    continue
            
            # 전략명 추출
            for strategy in STRATEGY_INFO.keys():
                if strategy in filename_lower:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        if market not in results:
                            results[market] = {}
                        results[market][strategy] = df
                        market_name = MARKET_INFO.get(market, {}).get('name', market)
                        print(f"  ✅ [{market_name}] {strategy}: {len(df)}개 종목 로드됨")
                    break
        
        return results
    
    def _format_number(self, value, format_type: str = 'default') -> str:
        """숫자 포맷팅"""
        if pd.isna(value):
            return 'N/A'
        
        if format_type == 'currency':
            if abs(value) >= 1e12:
                return f"${value/1e12:.2f}T"
            elif abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            else:
                return f"${value:,.2f}"
        elif format_type == 'percent':
            return f"{value:.2f}%"
        elif format_type == 'ratio':
            return f"{value:.2f}"
        else:
            return f"{value:,.2f}"
    
    def _create_stock_info_text(self, row: pd.Series, strategy: str) -> str:
        """종목 정보를 텍스트로 변환"""
        ticker_value = row.get('ticker')
        if pd.isna(ticker_value) or ticker_value is None:
            ticker_value = row.get('name')

        info_parts = [
            f"- 티커: {ticker_value if ticker_value is not None else 'N/A'}",
            f"- 회사명: {row.get('name', 'N/A')}",
            f"- 현재가: ${row.get('close', 0):.2f}",
            f"- 일간 변동률: {row.get('change', 0):.2f}%",
            f"- 거래량: {self._format_number(row.get('volume', 0))}",
            f"- 시가총액: {self._format_number(row.get('market_cap_basic', 0), 'currency')}",
            f"- 섹터: {row.get('sector', 'N/A')}",
            f"- 산업: {row.get('industry', 'N/A')}",
        ]
        
        # 전략별 추가 지표
        if strategy == 'cyclical':
            info_parts.extend([
                f"- PBR: {self._format_number(row.get('price_book_fq'), 'ratio')}",
                f"- EV/EBITDA: {self._format_number(row.get('enterprise_value_ebitda_ttm'), 'ratio')}",
                f"- 유동비율: {self._format_number(row.get('current_ratio_fq'), 'ratio')}",
            ])
        elif strategy == 'growth':
            info_parts.extend([
                f"- 매출 성장률 (YoY): {self._format_number(row.get('total_revenue_yoy_growth_ttm'), 'percent')}",
                f"- PEG 비율: {self._format_number(row.get('price_earnings_growth_ttm'), 'ratio')}",
                f"- 부채비율: {self._format_number(row.get('debt_to_equity_fq'), 'ratio')}",
                f"- EPS 성장률 (YoY): {self._format_number(row.get('earnings_per_share_diluted_yoy_growth_ttm'), 'percent')}",
            ])
        elif strategy == 'finance':
            info_parts.extend([
                f"- PBR: {self._format_number(row.get('price_book_fq'), 'ratio')}",
                f"- ROE: {self._format_number(row.get('return_on_equity_fq'), 'percent')}",
                f"- 배당수익률: {self._format_number(row.get('dividend_yield_recent'), 'percent')}",
            ])
        elif strategy == 'defensive':
            info_parts.extend([
                f"- 영업이익률: {self._format_number(row.get('operating_margin_ttm'), 'percent')}",
                f"- 잉여현금흐름 (FCF): {self._format_number(row.get('free_cash_flow_ttm'), 'currency')}",
                f"- 배당수익률: {self._format_number(row.get('dividend_yield_recent'), 'percent')}",
            ])
        
        # 애널리스트 정보
        info_parts.extend([
            f"- 애널리스트 점수: {self._format_number(row.get('analyst_score'), 'ratio')}",
            f"- 애널리스트 등급: {row.get('analyst_rating', 'N/A')}",
            f"- 애널리스트 수: {int(row.get('recommendation_total', 0))}명",
        ])
        
        return '\n'.join(info_parts)
    
    def _create_analysis_prompt(self, row: pd.Series, strategy: str) -> str:
        """분석 프롬프트 생성"""
        strategy_info = STRATEGY_INFO.get(strategy, {})
        stock_info = self._create_stock_info_text(row, strategy)

        ticker_value = row.get('ticker')
        if pd.isna(ticker_value) or ticker_value is None:
            ticker_value = row.get('name')
        
        prompt = f"""너는 월스트리트에서 일하고 있는 기업 분석 및 주식 시장 분석의 전문가야. 너의 이름은 'Gemini Stock Analyst'야. 너는 사용자가 입력한 주식 종목에({ticker_value if ticker_value is not None else 'Unknown'}) 대해서 각 단계별로 분석하고 최종 투자 의사 결정에 도움을 주는 역할을 한다.
목표 및 역할:
* 사용자가 요청한 특정 주식 종목에 대해 심층적인 기업 및 시장 분석 보고서를 제공한다.
* 보고서는 투자 의사 결정에 실질적인 도움을 줄 수 있도록 최신 정보를 기반으로 상세하고 깊이 있게 작성한다.
* 모든 답변은 한국어로 제공하며, 전문적인 보고서 양식을 따른다.
* 마크다운 형식을 사용한다.
행동 및 규칙:
1) 분석 보고서 작성:
   a) 사용자가 입력한 종목({ticker_value if ticker_value is not None else 'Unknown'})에 대해, 즉시 웹 검색 및 가능한 모든 도구를 활용하여 가장 최신 정보를 수집한다.
   b) 수집된 정보를 기반으로 아래 제시된 10단계 분석 과정을 철저히 따른다.
   c) 각 단계별 분석 내용은 가능한 한 상세하고 심층적이어야 하며, 데이터와 근거를 명확하게 제시해야 한다.
   d) 특히 '기술적 분석' 단계에서는 최근 1년간의 주가 트렌드와 차트 패턴 및 첨부된 모든 기술적 지표를 분석하고, '재무 상태 분석' 단계에서는 최근 3개년 및 최근 4개 분기 재무제표를 종합 분석한 내용을 필수로 포함한다.
   e) '가치 평가' 단계에서는 아래 절차에 명시된 가치평가기법을 필수로 활용하여 기업의 적정 가치와 현재 주가를 비교하여 투자 의견을 제시하도록 한다.
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
4) 첨부 데이터 우선:
   a) 요청에 CSV 파일(최근 1년 OHLCV+보조지표, 재무제표)이 첨부되었으면, 해당 첨부 데이터를 **가장 우선적인 근거 데이터**로 사용한다.
   b) 첨부 데이터와 웹 검색 결과가 충돌하면, 원칙적으로 첨부 데이터를 우선하되, 차이가 발생한 이유/가능한 원인(시점/통화/단위/정정 공시 등)을 명시한다.
전반적인 어조:
* 전문적이고 신뢰감을 주는 어조를 사용한다.
* 보고서 형식에 맞춰 격식 있고 명확한 문체를 유지한다.
* 사용자의 투자 결정을 지원하는 조력자로서의 역할을 수행한다."""
        
        return prompt
    
    def analyze_stock(self, row: pd.Series, strategy: str, market: Optional[str] = None) -> Optional[str]:
        """
        단일 종목 분석
        
        Parameters:
            row: 종목 데이터 (pandas Series)
            strategy: 투자 전략명
            
        Returns:
            분석 결과 텍스트 (실패 시 None)
        """
        ticker = row.get('ticker')
        if pd.isna(ticker) or ticker is None:
            ticker = row.get('name')
        if pd.isna(ticker) or ticker is None:
            ticker = 'Unknown'
        
        try:
            prompt = self._create_analysis_prompt(row, strategy)

            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            # yfinance 기반 근거 CSV 생성 + Gemini에 파일 첨부
            contents = [prompt]
            uploaded_files = []
            if self.market_data_dir:
                try:
                    _, price_csv, fin_csv = fetch_and_save_market_data_for_stock(
                        raw_ticker=_safe_str(ticker),
                        market=market,
                        output_dir=self.market_data_dir,
                        refresh=True,
                    )

                    # CSV 업로드 후 첨부
                    price_file = self.client.files.upload(
                        file=price_csv,
                        config=types.UploadFileConfig(
                            display_name=f'{_safe_str(ticker)}_ohlcv_1y_ta',
                            mime_type='text/csv',
                        ),
                    )
                    fin_file = self.client.files.upload(
                        file=fin_csv,
                        config=types.UploadFileConfig(
                            display_name=f'{_safe_str(ticker)}_financials_5y_ttm',
                            mime_type='text/csv',
                        ),
                    )
                    uploaded_files.extend([price_file, fin_file])
                    contents.extend([
                        "다음 첨부된 CSV 파일(최근 1년 OHLCV+보조지표, 재무제표)을 최우선 근거로 사용해 분석해줘.",
                        price_file,
                        fin_file,
                    ])
                except Exception as e:
                    # 첨부 실패 시에도 분석은 계속 진행(텍스트 프롬프트 + 웹검색)
                    print(f"    ⚠️ {ticker} 근거 데이터(CSV) 첨부 실패(분석은 계속): {str(e)}")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0,
                    max_output_tokens=60000,
                )
            )

            # 업로드 파일 정리(베스트 프랙티스)
            for f in uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                except Exception:
                    pass

            analysis_text = self._extract_text_from_response(response)
            if not analysis_text:
                raise ValueError("Gemini 응답에서 텍스트를 추출하지 못했습니다. (content.parts에 text 없음)")

            return analysis_text
            
        except Exception as e:
            print(f"    ⚠️ {ticker} 분석 실패: {str(e)}")
            return None
    
    def analyze_strategy(
        self, 
        df: pd.DataFrame, 
        strategy: str,
        max_stocks: int = 10,
        market: Optional[str] = None,
    ) -> List[Dict]:
        """
        전략별 종목 분석
        
        Parameters:
            df: 종목 DataFrame
            strategy: 투자 전략명
            max_stocks: 최대 분석 종목 수
            
        Returns:
            분석 결과 리스트
        """
        results = []
        strategy_info = STRATEGY_INFO.get(strategy, {})
        
        print(f"\n📊 {strategy_info.get('name', strategy)} 전략 분석 시작...")
        print(f"   총 {len(df)}개 종목 중 상위 {min(len(df), max_stocks)}개 분석")
        
        for idx, (_, row) in enumerate(df.head(max_stocks).iterrows()):
            ticker = row.get('ticker', 'Unknown')
            name = row.get('name', 'Unknown')
            
            print(f"   [{idx+1}/{min(len(df), max_stocks)}] {ticker} ({name}) 분석 중...")
            
            analysis = self.analyze_stock(row, strategy, market=market)
            
            if analysis:
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'strategy': strategy,
                    'analysis': analysis,
                    'data': row.to_dict(),
                })
                print(f"       ✅ 완료")
            else:
                print(f"       ❌ 실패")
            
            # Rate Limiting 대응
            if idx < min(len(df), max_stocks) - 1:
                time.sleep(API_DELAY)
        
        return results
    
    def generate_strategy_report(
        self, 
        analyses: List[Dict], 
        strategy: str,
        market: str = None
    ) -> str:
        """전략별 보고서 생성"""
        strategy_info = STRATEGY_INFO.get(strategy, {})
        market_info = MARKET_INFO.get(market, {})
        market_name = market_info.get('name', '')
        market_suffix = f" ({market_name})" if market_name else ""
        
        report_parts = [
            f"# {strategy_info.get('name', strategy)} 투자 분석 보고서{market_suffix}",
            f"\n> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if market_name:
            report_parts.append(f"> 시장: {market_name}")
        
        report_parts.extend([
            f"\n## 전략 개요",
            f"- **목표**: {strategy_info.get('description', '')}",
            f"- **핵심 지표**: {strategy_info.get('focus', '')}",
            f"- **분석 종목 수**: {len(analyses)}개",
            "\n---\n",
        ])
        
        for idx, item in enumerate(analyses, 1):
            report_parts.extend([
                f"## {idx}. {item['ticker']} - {item['name']}",
                "",
                item['analysis'],
                "\n---\n",
            ])
        
        report_parts.append("\n⚠️ **면책조항**: 본 보고서는 AI가 생성한 참고 자료이며, 투자 권유가 아닙니다. 실제 투자 결정은 추가적인 조사와 전문가 상담을 권장합니다.")
        
        return '\n'.join(report_parts)
    
    def generate_summary_report(
        self, 
        all_analyses: Dict[str, List[Dict]],
        market: str = None
    ) -> str:
        """시장별 종합 보고서 생성"""
        market_info = MARKET_INFO.get(market, {})
        market_name = market_info.get('name', '')
        market_suffix = f" ({market_name})" if market_name else ""
        
        report_parts = [
            f"# 📈 투자 종합 분석 보고서{market_suffix}",
            f"\n> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if market_name:
            report_parts.append(f"> 시장: {market_name}")
        
        report_parts.extend([
            "",
            "## 목차",
        ])
        
        # 목차 생성
        for strategy, analyses in all_analyses.items():
            if analyses:
                strategy_info = STRATEGY_INFO.get(strategy, {})
                report_parts.append(f"- [{strategy_info.get('name', strategy)}](#{strategy}) ({len(analyses)}개 종목)")
        
        report_parts.append("\n---\n")
        
        # 각 전략별 요약
        for strategy, analyses in all_analyses.items():
            if analyses:
                strategy_info = STRATEGY_INFO.get(strategy, {})
                report_parts.extend([
                    f"<a name=\"{strategy}\"></a>",
                    f"## {strategy_info.get('name', strategy)}",
                    f"**전략 설명**: {strategy_info.get('description', '')}",
                    "",
                    "| 순위 | 티커 | 회사명 | 애널리스트 등급 |",
                    "|------|------|--------|-----------------|",
                ])
                
                for idx, item in enumerate(analyses, 1):
                    data = item['data']
                    report_parts.append(
                        f"| {idx} | {item['ticker']} | {item['name']} | {data.get('analyst_rating', 'N/A')} |"
                    )
                
                report_parts.append("\n")
                
                # 각 종목 상세 분석 링크
                report_parts.append("### 상세 분석")
                for item in analyses:
                    report_parts.extend([
                        f"#### {item['ticker']} - {item['name']}",
                        "",
                        item['analysis'],
                        "\n---\n",
                    ])
        
        report_parts.append("\n⚠️ **면책조항**: 본 보고서는 AI가 생성한 참고 자료이며, 투자 권유가 아닙니다. 실제 투자 결정은 추가적인 조사와 전문가 상담을 권장합니다.")
        
        return '\n'.join(report_parts)
    
    def save_reports(
        self, 
        all_analyses: Dict[str, List[Dict]],
        output_dir: str,
        market: str = None
    ) -> List[str]:
        """보고서 저장 (시장별)"""
        saved_files = []
        market_prefix = f'{market}_' if market else ''
        market_info = MARKET_INFO.get(market, {})
        market_name = market_info.get('name', '')
        
        if market_name:
            print(f"\n📝 [{market_name}] 보고서 저장 중...")
        
        # 전략별 보고서 저장
        for strategy, analyses in all_analyses.items():
            if analyses:
                report = self.generate_strategy_report(analyses, strategy, market)
                filename = os.path.join(output_dir, f'analysis_{market_prefix}{strategy}.md')
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                saved_files.append(filename)
                print(f"  ✅ 저장: {filename}")
        
        # 시장별 종합 보고서 저장
        if any(all_analyses.values()):
            summary_report = self.generate_summary_report(all_analyses, market)
            summary_filename = os.path.join(output_dir, f'{market_prefix}investment_report.md')
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            saved_files.append(summary_filename)
            print(f"  ✅ 저장: {summary_filename}")
        
        return saved_files
    
    def run_analysis(
        self, 
        screener_dir: str,
        max_stocks_per_strategy: int = 5,
        analyzer_output_dir: str = None
    ) -> Tuple[Dict[str, Dict[str, List[Dict]]], str]:
        """
        전체 분석 실행 (시장별로 분리)
        
        Parameters:
            screener_dir: 스크리닝 결과 디렉토리 (output/screener/{timestamp})
            max_stocks_per_strategy: 전략당 최대 분석 종목 수
            analyzer_output_dir: 분석 결과 저장 디렉토리 (None이면 자동 생성)
            
        Returns:
            (시장별 > 전략별 분석 결과 딕셔너리, 분석 결과 저장 디렉토리)
            예: {'us': {'growth': [...]}, 'kr': {'growth': [...]}}
        """
        print("=" * 60)
        print("🤖 LLM 기반 주식 종합 분석 시작")
        print(f"   모델: {self.model}")
        print("=" * 60)
        
        # 1. 스크리닝 결과 로드 (시장별로 분리)
        print(f"\n📂 스크리닝 결과 로드 중... ({screener_dir})")
        screening_results = self.load_screening_results(screener_dir)
        
        if not screening_results:
            print("❌ 분석할 종목이 없습니다.")
            return {}, ""
        
        # 2. 분석 결과 저장 디렉토리 생성
        if analyzer_output_dir is None:
            analyzer_output_dir = create_analyzer_output_dir()
        else:
            os.makedirs(analyzer_output_dir, exist_ok=True)
        
        print(f"📁 분석 결과 저장 경로: {analyzer_output_dir}")

        # 2-1. yfinance 근거 데이터 저장 디렉토리 설정 (screener_dir 날짜 폴더와 동일하게)
        screener_folder_name = os.path.basename(os.path.normpath(screener_dir))
        self.market_data_dir = os.path.join(MARKET_DATA_OUTPUT_DIR, screener_folder_name)
        os.makedirs(self.market_data_dir, exist_ok=True)
        print(f"📎 근거 데이터(CSV) 저장 경로: {self.market_data_dir}")
        
        # 3. 시장별 > 전략별 분석
        all_market_analyses = {}  # {market: {strategy: [analyses]}}
        total_analyzed = 0
        
        for market, strategies in screening_results.items():
            market_info = MARKET_INFO.get(market, {})
            market_name = market_info.get('name', market)
            
            print(f"\n{'='*60}")
            print(f"🌍 [{market_name}] 시장 분석 시작")
            print(f"{'='*60}")
            
            market_analyses = {}
            
            for strategy, df in strategies.items():
                strategy_info = STRATEGY_INFO.get(strategy, {})
                print(f"\n📊 [{market_name}] {strategy_info.get('name', strategy)} 전략 분석...")
                
                analyses = self.analyze_strategy(df, strategy, max_stocks_per_strategy, market=market)
                market_analyses[strategy] = analyses
                total_analyzed += len(analyses)
            
            all_market_analyses[market] = market_analyses
            
            # 시장별 보고서 저장
            self.save_reports(market_analyses, analyzer_output_dir, market)
        
        # 4. 완료 메시지
        print("\n" + "=" * 60)
        print(f"✅ 분석 완료! 총 {total_analyzed}개 종목 분석됨")
        for market, analyses in all_market_analyses.items():
            market_info = MARKET_INFO.get(market, {})
            market_name = market_info.get('name', market)
            market_count = sum(len(a) for a in analyses.values())
            print(f"   • {market_name}: {market_count}개 종목")
        print(f"📁 보고서 위치: {analyzer_output_dir}")
        print("=" * 60)
        
        return all_market_analyses, analyzer_output_dir


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_latest_screener_dir(base_dir: str = SCREENER_OUTPUT_DIR) -> Optional[str]:
    """가장 최근 screener 결과 디렉토리 반환"""
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


def create_analyzer_output_dir(base_dir: str = ANALYZER_OUTPUT_DIR) -> str:
    """
    날짜 기반 분석 결과 출력 디렉토리 생성
    
    Parameters:
        base_dir: 기본 출력 디렉토리
        
    Returns:
        생성된 디렉토리 경로 (output/analyzer/{YYYYMMDD})
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
        python stock_analyzer.py                                    # 가장 최근 screener 결과 분석
        python stock_analyzer.py output/screener/20251204_151114    # 특정 screener 폴더 분석
    
    Note:
        최종 추천 보고서는 portfolio_maker.py를 사용하세요.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM 기반 주식 종합 분석 및 투자 조언 보고서 생성기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_analyzer.py                                    # 가장 최근 screener 결과 분석
  python stock_analyzer.py output/screener/20251204_151114    # 특정 screener 폴더 분석
  python stock_analyzer.py -m 3                               # 전략당 3개 종목 분석

Directory Structure:
  입력: output/screener/{timestamp}/  (스크리닝 CSV 결과)
  출력: output/analyzer/{timestamp}/  (분석 MD 보고서)

Note:
  최종 추천 보고서는 portfolio_maker.py를 사용하세요:
  python portfolio_maker.py output/analyzer/20251204_151114
        """
    )
    parser.add_argument(
        'screener_dir', 
        nargs='?', 
        default=None,
        help='분석할 스크리닝 결과 디렉토리 (기본값: 가장 최근 output/screener 폴더)'
    )
    parser.add_argument(
        '--max-stocks', '-m',
        type=int,
        default=1,
        help='전략당 최대 분석 종목 수 (기본값: 1)'
    )
    
    args = parser.parse_args()
    
    # screener 디렉토리 결정
    screener_dir = args.screener_dir or get_latest_screener_dir()
    
    if not screener_dir or not os.path.exists(screener_dir):
        print("❌ 분석할 스크리닝 결과 디렉토리를 찾을 수 없습니다.")
        print("   사용법: python stock_analyzer.py [screener_directory]")
        print("   예시: python stock_analyzer.py output/screener/20251204_151114")
        print(f"\n   힌트: 먼저 python stock_screener.py를 실행하여 스크리닝 결과를 생성하세요.")
        sys.exit(1)
    
    print(f"📂 스크리닝 결과 디렉토리: {screener_dir}")
    
    try:
        analyzer = StockAnalyzer()
        _, analyzer_output_dir = analyzer.run_analysis(
            screener_dir, 
            max_stocks_per_strategy=args.max_stocks
        )
        
        if analyzer_output_dir:
            print(f"\n💡 포트폴리오 추천을 생성하려면:")
            print(f"   python portfolio_maker.py {analyzer_output_dir}")
            
    except ValueError as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

