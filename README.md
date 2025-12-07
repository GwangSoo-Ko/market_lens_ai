# 🔍 Market Lens AI - AI 기반 글로벌 주식 분석 플랫폼

TradingView Screener와 Gemini AI를 활용하여 **스크리닝 → 심층 분석 → 포트폴리오 추천**까지 자동화된 투자 분석 파이프라인을 제공합니다.

## ✨ 주요 기능

| 단계 | 스크립트 | 설명 |
|------|----------|------|
| 1️⃣ | `stock_screener.py` | 4가지 투자 전략 기반 글로벌 종목 스크리닝 |
| 2️⃣ | `stock_analyzer.py` | Gemini AI를 활용한 종목별 심층 분석 보고서 생성 |
| 3️⃣ | `portfolio_maker.py` | 최종 추천 종목 및 포트폴리오 예산 분배 전략 제시 |
| 🚀 | `live_process.py` | 1→2→3 전체 파이프라인 자동 실행 |

---

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

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

Gemini API 키를 `.env` 파일에 설정하세요:

```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

또는 환경 변수로 직접 설정:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 의존성 목록

| 패키지 | 버전 | 설명 |
|--------|------|------|
| tradingview-screener | >= 3.0.0 | TradingView 스크리너 API |
| google-genai | >= 1.0.0 | Gemini AI API |
| pandas | >= 2.0.0 | 데이터 분석 |
| python-dotenv | >= 1.0.0 | 환경 변수 관리 |

---

## 📁 프로젝트 구조

```
market_lens_ai/
├── README.md                 # 프로젝트 설명 (현재 파일)
├── requirements.txt          # 의존성 목록
├── .env                      # API 키 설정 (gitignore)
│
├── live_process.py           # 🚀 전체 파이프라인 실행기
├── stock_screener.py         # 📊 종목 스크리닝
├── stock_analyzer.py         # 🤖 AI 종목 분석
├── portfolio_maker.py        # 🎯 포트폴리오 추천
│
└── output/                   # 결과 저장 디렉토리
    ├── screener/             # 스크리닝 결과 (CSV)
    │   └── {YYYYMMDD}/
    │       ├── global_cyclical.csv
    │       ├── global_growth.csv
    │       ├── global_finance.csv
    │       └── global_defensive.csv
    ├── analyzer/             # 분석 보고서 (MD)
    │   └── {YYYYMMDD}/
    │       ├── analysis_cyclical.md
    │       ├── analysis_growth.md
    │       ├── analysis_finance.md
    │       ├── analysis_defensive.md
    │       └── investment_report.md
    └── portfolio/            # 포트폴리오 추천 (MD)
        └── {YYYYMMDD}/
            └── final_recommendation.md
```

---

## 🚀 빠른 시작

### 전체 파이프라인 실행 (권장)

```bash
# 기본 실행 (전략당 1개 종목 분석)
python live_process.py

# 전략당 3개 종목 분석
python live_process.py -m 3

# 스크리닝 건너뛰기 (기존 결과 사용)
python live_process.py --skip-screener

# 포트폴리오 추천 건너뛰기
python live_process.py --skip-portfolio
```

### 개별 단계 실행

```bash
# Step 1: 스크리닝만 실행
python stock_screener.py

# Step 2: 분석만 실행 (가장 최근 screener 결과 사용)
python stock_analyzer.py
# 또는 특정 screener 폴더 지정
python stock_analyzer.py output/screener/20251207

# Step 3: 포트폴리오 추천만 실행 (가장 최근 analyzer 결과 사용)
python portfolio_maker.py
# 또는 특정 analyzer 폴더 지정
python portfolio_maker.py output/analyzer/20251207
```

---

## 💡 모듈 사용법

### 스크리너 모듈

```python
from stock_screener import (
    screen_cyclical,
    screen_growth,
    screen_finance,
    screen_defensive,
    run_all_screeners,
    save_results,
)

# 개별 전략 실행
count, df = screen_growth(filter_sector=True)
print(f"Growth 종목: {len(df)}개")

# 전체 전략 실행 및 저장
results = run_all_screeners()
save_results(results)
```

### 분석 모듈

```python
from stock_analyzer import StockAnalyzer

analyzer = StockAnalyzer()
analyses, output_dir = analyzer.run_analysis(
    screener_dir='output/screener/20251207',
    max_stocks_per_strategy=3
)
```

### 포트폴리오 모듈

```python
from portfolio_maker import PortfolioMaker

maker = PortfolioMaker()
result, output_dir = maker.generate_recommendation(
    analyzer_dir='output/analyzer/20251207'
)
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
======================================================================
🚀 Market Lens AI - 전체 파이프라인 실행
======================================================================
⏰ 시작 시간: 2025-12-07 15:30:00
📊 전략당 분석 종목 수: 1
======================================================================

======================================================================
📊 [1/3] 스크리닝 시작
======================================================================
📊 스크리닝 시작...
------------------------------------------------------------
  • Cyclical (경기민감형): 10개 중 5개 필터링됨
  • Growth (고성장형): 40개 중 12개 필터링됨
  • Finance (금융/자산주): 5개 중 3개 필터링됨
  • Defensive (경기방어주): 50개 중 8개 필터링됨
------------------------------------------------------------

======================================================================
🤖 [2/3] LLM 종목 분석 시작
======================================================================
...

======================================================================
🎯 [3/3] 포트폴리오 추천 생성 시작
======================================================================
...

======================================================================
✅ 파이프라인 실행 완료!
======================================================================
⏱️ 총 소요 시간: 0:05:32.123456

📁 결과 파일 위치:
   • 스크리닝: output/screener/20251207
   • 분석:     output/analyzer/20251207
   • 포트폴리오: output/portfolio/20251207
======================================================================
```

---

## ⚠️ 주의사항

1. **API 키 필수**: Gemini API 키가 필요합니다 (`.env` 파일 또는 환경 변수 설정)
2. **API 제한**: TradingView 및 Gemini API에는 요청 제한이 있을 수 있습니다
3. **데이터 지연**: 실시간 데이터가 아닌 지연된 데이터일 수 있습니다
4. **투자 조언 아님**: 본 도구는 참고용이며, 투자 결정에 대한 책임은 투자자 본인에게 있습니다

---

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.
