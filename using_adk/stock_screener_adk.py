"""
Stock Screener (ADK 버전) - 4가지 투자 전략 기반 주식 스크리너 (미국/한국)

기존 stock_screener.py의 스크리닝/저장 로직은 그대로 재사용합니다.
추가로, 원하면(--adk-summary) 스크리닝 결과를 ADK Agent로 간단 요약할 수 있습니다.

기본 동작(조건/CSV 출력/폴더 구조)은 기존과 동일합니다.

Usage:
    python stock_screener_adk.py              # 기본: 미국+한국 모두 스크리닝
    python stock_screener_adk.py --market us  # 미국만
    python stock_screener_adk.py --market kr  # 한국만
    python stock_screener_adk.py --adk-summary

Environment Variables:
    GOOGLE_API_KEY 또는 GEMINI_API_KEY: (요약 기능 사용 시 필요)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


from stock_screener import (
    ANALYST_SCORE_BUY,
    SUPPORTED_MARKETS,
    print_summary,
    run_all_markets,
    run_all_screeners,
    save_results,
)


def _try_build_summary_agent(model: str):
    try:
        from google.adk.agents import Agent  # type: ignore[import-not-found]
        from google.adk.tools import google_search  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError("google-adk 패키지가 설치되어 있지 않습니다. (pip install google-adk)")

    from adk_utils import AdkAgentRunner

    agent = Agent(
        name="market_lens_screener_summarizer",
        model=model,
        instruction=(
            "너는 퀀트 리서처다. 제공된 스크리닝 결과(전략별 상위 종목 리스트)를 바탕으로 "
            "시장별로 핵심 관찰사항을 5~10줄로 요약해라. 한국어로 작성하고, 근거를 간단히 언급한다. "
            "필요 시 google_search로 산업/뉴스를 확인할 수 있으나, 데이터(티커/지표)가 우선이다."
        ),
        description="Market Lens AI - Screener Summary Agent (ADK)",
        tools=[google_search],
    )
    return AdkAgentRunner(agent=agent, app_name="market_lens_ai")


def _summarize_results_with_adk(
    results_by_market: Dict[str, Dict[str, pd.DataFrame]],
    model: str,
) -> str:
    runner = _try_build_summary_agent(model)

    lines = ["다음은 주식 스크리닝 결과 요약 요청이다.", ""]
    for market, strategies in results_by_market.items():
        market_name = SUPPORTED_MARKETS.get(market, {}).get("name", market)
        lines.append(f"## 시장: {market_name} ({market})")
        for strat, df in strategies.items():
            if df is None or df.empty:
                continue
            cols = [c for c in ["ticker", "name", "sector", "industry", "close", "analyst_rating", "analyst_score"] if c in df.columns]
            sample = df.loc[:, cols].head(10).to_csv(index=False) if cols else df.head(10).to_csv(index=False)
            lines.append(f"\n### 전략: {strat} (상위 10개 샘플)\n{sample}")
        lines.append("")

    prompt = "\n".join(lines)
    return runner.run_text(prompt)


def main():
    parser = argparse.ArgumentParser(
        description="(ADK) 4가지 투자 전략 기반 주식 스크리너 (미국/한국)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_screener_adk.py
  python stock_screener_adk.py --market us
  python stock_screener_adk.py --market kr
  python stock_screener_adk.py --adk-summary
        """,
    )
    parser.add_argument(
        "--market",
        "-m",
        type=str,
        default=None,
        choices=["us", "kr"],
        help="스크리닝할 시장 선택 (지정하지 않으면 미국+한국 모두 실행)",
    )
    parser.add_argument(
        "--no-sector-filter",
        action="store_true",
        help="섹터 필터링 비활성화",
    )
    parser.add_argument(
        "--min-analyst-score",
        type=float,
        default=ANALYST_SCORE_BUY,
        help=f"최소 애널리스트 점수 (기본값: {ANALYST_SCORE_BUY})",
    )
    parser.add_argument(
        "--adk-summary",
        action="store_true",
        help="(선택) ADK로 스크리닝 결과를 간단히 요약 출력",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="gemini-2.0-flash",
        help="(선택) 요약용 Gemini 모델 (기본값: gemini-2.0-flash)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 Market Lens AI - 주식 스크리너 (ADK 버전 래퍼)")
    print("=" * 60)

    if args.market is None:
        market_names = ", ".join([cfg["name"] for cfg in SUPPORTED_MARKETS.values()])
        print(f"🌍 시장: {market_names} (전체)")
        print("=" * 60)

        all_results, saved_files = run_all_markets(
            markets=None,
            filter_sector=not args.no_sector_filter,
            min_analyst_score=args.min_analyst_score,
        )
        # 전체 결과 요약
        print("\n" + "=" * 60)
        print("📊 전체 스크리닝 완료!")
        print("=" * 60)
        print(f"📁 저장된 파일 수: {len(saved_files)}개")
        if saved_files:
            print(f"📂 출력 디렉토리: {os.path.dirname(saved_files[0])}")
        print("=" * 60)

        if args.adk_summary:
            try:
                print("\n🧠 (ADK) 스크리닝 결과 요약 생성 중...")
                summary = _summarize_results_with_adk(all_results, model=args.summary_model)
                if summary:
                    print("\n" + "=" * 60)
                    print("📝 (ADK) 스크리닝 요약")
                    print("=" * 60)
                    print(summary)
            except Exception as e:
                print(f"⚠️ (ADK) 요약 생성 실패: {e}")

        return all_results

    # 특정 시장만 스크리닝
    market = args.market
    results = run_all_screeners(
        market=market,
        filter_sector=not args.no_sector_filter,
        min_analyst_score=args.min_analyst_score,
    )
    print_summary(results, market=market)
    print("\n📁 결과 저장 중...")
    saved_files = save_results(results, market=market)

    if args.adk_summary:
        try:
            print("\n🧠 (ADK) 스크리닝 결과 요약 생성 중...")
            summary = _summarize_results_with_adk({market: results}, model=args.summary_model)
            if summary:
                print("\n" + "=" * 60)
                print("📝 (ADK) 스크리닝 요약")
                print("=" * 60)
                print(summary)
        except Exception as e:
            print(f"⚠️ (ADK) 요약 생성 실패: {e}")

    return results


if __name__ == "__main__":
    main()


