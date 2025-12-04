# TradingView Screener - 글로벌 주식 종목 스크리닝

`tradingview-screener` 라이브러리를 사용하여 TradingView 데이터 기반으로 **4가지 투자 전략**에 따라 글로벌 주식 종목을 스크리닝하는 Python 프로젝트입니다.

## 🎯 4가지 투자 전략

### 1. Cyclical (경기민감형)
> **목표:** 자산 가치 대비 저평가되고, 현금 창출력이 좋은 기업

| 필터 | 조건 | 설명 |
|------|------|------|
| 섹터 | 공정 산업, 비에너지 광물, 생산자 제조, 소비자 내구재, 에너지 광물, 전자 기술 | 경기민감 업종 |
| PBR | < 1 | 자산가치 대비 저평가 |
| EV/EBITDA | < 6 | 현금 창출력 대비 저평가 |
| 유동비율 | >= 1.5 | 경기 침체 시 버틸 현금 체력 |
| 애널리스트/기술 등급 | Buy 이상 | 전문가 평가 매수 이상 |

### 2. Growth (고성장형)
> **목표:** 매출이 빠르게 늘면서, 성장성 대비 주가가 싼 기업

| 필터 | 조건 | 설명 |
|------|------|------|
| 섹터 | 기술 서비스, 보건 서비스, 상업 서비스, 의료 기술 | 성장 섹터 |
| 매출 성장률 YoY | >= 20% | 전년 대비 고속 성장 |
| PEG 비율 | < 1 | 성장률 감안 시 저평가 |
| 부채비율 | < 150% | 금리 리스크 관리 |
| 애널리스트/기술 등급 | Buy 이상 | 전문가 평가 매수 이상 |

### 3. Finance (금융/자산주)
> **목표:** 극도로 저평가된 자산과 높은 자본효율, 배당 매력

| 필터 | 조건 | 설명 |
|------|------|------|
| 섹터 | 금융 (Finance) | 은행, 보험, 투자은행 등 |
| PBR | < 0.6 | 절대적 저평가 영역 |
| ROE | >= 10% | 저평가지만 돈은 잘 버는 곳 |
| 배당수익률 | >= 4% | 확실한 현금 보상 |
| 애널리스트/기술 등급 | Buy 이상 | 전문가 평가 매수 이상 |

### 4. Defensive (경기방어주)
> **목표:** 마진이 안정적이고, 현금이 잘 돌며 배당을 주는 기업

| 필터 | 조건 | 설명 |
|------|------|------|
| 섹터 | 소비재 비내구재, 유틸리티, 커뮤니케이션 | 필수 소비재 |
| 영업이익률 | >= 5% | 안정적인 마진 확보 |
| FCF (잉여현금흐름) | > 0 | 현금이 플러스인지 확인 |
| 배당수익률 | >= 3% | 은행 이자 이상의 수익 |
| 애널리스트/기술 등급 | Buy 이상 | 전문가 평가 매수 이상 |

---

## 📦 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install tradingview-screener pandas matplotlib seaborn jupyter
```

### 의존성

| 패키지 | 버전 | 설명 |
|--------|------|------|
| tradingview-screener | >= 3.0.0 | TradingView 스크리너 API |
| pandas | >= 2.0.0 | 데이터 분석 |
| matplotlib | >= 3.7.0 | 시각화 |
| seaborn | >= 0.12.0 | 통계 시각화 |
| jupyter | >= 1.0.0 | 노트북 환경 |

---

## 📁 파일 구조

```
tradingview_screener/
├── README.md                      # 프로젝트 설명 (현재 파일)
├── requirements.txt               # 의존성 목록
├── .gitignore                     # Git 제외 파일 목록
├── global_screener_spec.md        # 4가지 투자 전략 스펙 문서
├── stock_screener.py              # 모듈화된 스크리너 (import용)
├── global_screener.py             # CLI 스크리너 (직접 실행용)
├── stock_screener_notebook.ipynb  # Jupyter 노트북 (인터랙티브)
└── output/                        # 스크리닝 결과 저장 디렉토리
    └── {timestamp}/               # 실행 시점별 결과 폴더
        ├── global_cyclical.csv
        ├── global_growth.csv
        ├── global_finance.csv
        └── global_defensive.csv
```

| 파일 | 용도 |
|------|------|
| `stock_screener.py` | 모듈로 import하여 사용, 함수 단위 호출 가능 |
| `global_screener.py` | CLI로 직접 실행, 결과 터미널 출력 |
| `stock_screener_notebook.ipynb` | Jupyter 환경에서 인터랙티브 분석 |
| `output/` | CSV 결과 파일 저장 (타임스탬프별 폴더) |

---

## 🚀 빠른 시작

### 1. Python 스크립트 실행 (CLI)

```bash
# 4가지 전략 모두 실행
python global_screener.py

# 또는
python stock_screener.py
```

### 2. 모듈로 import하여 사용

```python
from stock_screener import (
    screen_cyclical,
    screen_growth,
    screen_finance,
    screen_defensive,
    run_all_screeners,
    save_results,
    create_output_dir
)

# 개별 전략 실행
count, df = screen_growth(filter_sector=True)
print(f"Growth 종목: {len(df)}개")

# 전체 전략 실행
results = run_all_screeners()

# 결과 저장 (CSV) - output/{timestamp}/ 디렉토리에 자동 저장
save_results(results)

# 커스텀 디렉토리에 저장
save_results(results, output_dir='my_results')
```

### 3. Jupyter Notebook 실행

```bash
jupyter notebook stock_screener_notebook.ipynb
```

---

## 💡 주요 기능

### 기본 스크리닝

```python
from tradingview_screener import Query, col

# 상위 50개 종목 조회
count, df = (
    Query()
    .select('name', 'close', 'volume', 'market_cap_basic')
    .get_scanner_data()
)
```

### 조건부 필터링

```python
count, df = (
    Query()
    .select('name', 'close', 'volume', 'change')
    .where(
        col('market_cap_basic') > 1_000_000_000,  # 시총 10억 달러 이상
        col('change') > 5,                        # 5% 이상 상승
        col('volume') > 1_000_000                 # 거래량 100만주 이상
    )
    .order_by('change', ascending=False)
    .limit(20)
    .get_scanner_data()
)
```

### 기술적 지표 활용 (MACD, RSI)

```python
# MACD 골든 크로스 + RSI 조건
count, df = (
    Query()
    .select('name', 'close', 'MACD.macd', 'MACD.signal', 'RSI')
    .where(
        col('MACD.macd') >= col('MACD.signal'),  # MACD 골든크로스
        col('RSI').between(30, 70)               # RSI 30~70 구간
    )
    .get_scanner_data()
)
```

### 애널리스트 평점 계산

```python
from stock_screener import calculate_analyst_score, filter_by_analyst

# 애널리스트 점수 계산 (-2 ~ 2 스케일)
df = calculate_analyst_score(df)

# Buy 이상 필터링 (score >= 0.5)
df = filter_by_analyst(df, min_score=0.5)
```

---

## 📊 사용 가능한 필드

### 가격 관련
| 필드 | 설명 |
|------|------|
| `close` | 종가 |
| `open` | 시가 |
| `high` | 고가 |
| `low` | 저가 |
| `change` | 변동률 (%) |
| `price_52_week_high` | 52주 최고가 |
| `price_52_week_low` | 52주 최저가 |

### 거래량 관련
| 필드 | 설명 |
|------|------|
| `volume` | 거래량 |
| `relative_volume_10d_calc` | 10일 평균 대비 상대 거래량 |

### 기술적 지표
| 필드 | 설명 |
|------|------|
| `RSI` | RSI (14일) |
| `MACD.macd` | MACD 라인 |
| `MACD.signal` | MACD 시그널 라인 |
| `Recommend.All` | 기술 등급 (-1 ~ 1) |

### 펀더멘탈
| 필드 | 설명 |
|------|------|
| `market_cap_basic` | 시가총액 |
| `price_earnings_ttm` | P/E 비율 (TTM) |
| `price_book_fq` | P/B 비율 (PBR) |
| `price_earnings_growth_ttm` | PEG 비율 |
| `dividend_yield_recent` | 배당 수익률 |
| `return_on_equity_fq` | ROE |
| `enterprise_value_ebitda_ttm` | EV/EBITDA |
| `current_ratio_fq` | 유동비율 |
| `debt_to_equity_fq` | 부채비율 |
| `operating_margin_ttm` | 영업이익률 |
| `free_cash_flow_ttm` | 잉여현금흐름 (FCF) |
| `total_revenue_yoy_growth_ttm` | 매출 성장률 YoY |

### 애널리스트 평점
| 필드 | 설명 |
|------|------|
| `recommendation_buy` | Strong Buy 의견 수 |
| `recommendation_over` | Buy 의견 수 |
| `recommendation_hold` | Hold 의견 수 |
| `recommendation_under` | Sell 의견 수 |
| `recommendation_sell` | Strong Sell 의견 수 |
| `recommendation_total` | 총 애널리스트 수 |

### 분류
| 필드 | 설명 |
|------|------|
| `sector` | 섹터 |
| `industry` | 산업 |
| `exchange` | 거래소 |

### 성과
| 필드 | 설명 |
|------|------|
| `Perf.W` | 주간 수익률 |
| `Perf.1M` | 월간 수익률 |
| `Perf.3M` | 3개월 수익률 |
| `Perf.6M` | 6개월 수익률 |
| `Perf.Y` | 연간 수익률 |

---

## 🔧 col() 함수 사용법

```python
from tradingview_screener import col

# 비교 연산
col('volume') > 1_000_000              # 거래량 100만 초과
col('change') >= 5                     # 변동률 5% 이상
col('RSI') < 30                        # RSI 30 미만

# 범위 조건
col('market_cap_basic').between(1_000_000_000, 10_000_000_000)  # 시총 10~100억 달러

# 컬럼 간 비교
col('MACD.macd') >= col('MACD.signal')  # MACD가 시그널보다 높음

# 동일 조건
col('is_primary') == True               # Primary 종목만
```

---

## 📈 평점 기준

### 기술 등급 (Recommend.All)
| 점수 범위 | 등급 |
|-----------|------|
| 0.5 ~ 1.0 | Strong Buy |
| 0.1 ~ 0.5 | Buy |
| -0.1 ~ 0.1 | Neutral |
| -0.5 ~ -0.1 | Sell |
| -1.0 ~ -0.5 | Strong Sell |

### 애널리스트 평점 (계산 방식)
```
점수 = (2×Strong Buy + 1×Buy + 0×Hold - 1×Sell - 2×Strong Sell) / 총 애널리스트 수
```

| 점수 범위 | 등급 |
|-----------|------|
| >= 1.0 | Strong Buy |
| 0.5 ~ 1.0 | Buy |
| -0.5 ~ 0.5 | Hold |
| -1.0 ~ -0.5 | Sell |
| < -1.0 | Strong Sell |

---

## 📋 출력 예시

```
📊 스크리닝 시작...
------------------------------------------------------------
  • Cyclical (경기민감형): 10개 중 5개 필터링됨
  • Growth (고성장형): 40개 중 12개 필터링됨
  • Finance (금융/자산주): 5개 중 3개 필터링됨
  • Defensive (경기방어주): 50개 중 8개 필터링됨
------------------------------------------------------------
============================================================
📊 4가지 투자 전략 스크리닝 결과 요약
============================================================
  • Cyclical (경기민감형): 5개 종목
  • Growth (고성장형): 12개 종목
  • Finance (금융/자산주): 3개 종목
  • Defensive (경기방어주): 8개 종목
============================================================

📁 결과 저장 중...
📂 출력 디렉토리: output/20251204_151234
  ✅ 저장: output/20251204_151234/global_cyclical.csv
  ✅ 저장: output/20251204_151234/global_growth.csv
  ✅ 저장: output/20251204_151234/global_finance.csv
  ✅ 저장: output/20251204_151234/global_defensive.csv
```

### 출력 디렉토리 구조

```
output/
├── 20251204_151234/          # 첫 번째 실행
│   ├── global_cyclical.csv
│   ├── global_growth.csv
│   ├── global_finance.csv
│   └── global_defensive.csv
├── 20251204_160000/          # 두 번째 실행
│   ├── global_cyclical.csv
│   └── ...
└── ...
```

---

## ⚠️ 주의사항

1. **API 제한**: TradingView API에는 요청 제한이 있을 수 있습니다.
2. **데이터 지연**: 실시간 데이터가 아닌 지연된 데이터일 수 있습니다.
3. **필드 가용성**: 모든 필드가 모든 종목에서 사용 가능한 것은 아닙니다.
4. **투자 조언 아님**: 본 스크리너는 참고용이며, 투자 결정에 대한 책임은 투자자 본인에게 있습니다.

---

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.
