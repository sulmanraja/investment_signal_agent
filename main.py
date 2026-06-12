"""Investment Signal Agent — main entry point.

Full pipeline for a single ticker:
  1. Retrieve SEC context (edgar_retriever)
  2. Fetch macro snapshot (fred_macro_tool)
  3. Fetch news (newsdata_tool)
  4. Fetch GitHub trends (github_tool)
  5. Compute signal alignment score (signals)
  6. Run OODA loop chain
  7. Run Tree-of-Thoughts beam search → stance
  8. Run standard scoring chain
  9. Synthesise into investment brief
  10. Write report to output/reports/
"""

import argparse

from tools.edgar_retriever import retrieve_context
from tools.fred_macro_tool import get_latest_snapshot
from tools.newsdata_tool import fetch_news
from tools.github_tool import fetch_github_trends
from signals.alignment_scorer import score_alignment
from chains.ooda_loop import build_ooda_chain
from chains.standard_scoring import build_standard_scoring_chain
from chains.synthesis_chain import build_synthesis_chain
from chains.tot_stance_chain import build_tot_stance_chain
from output.investment_brief import generate_brief
from memory.long_term import LongTermMemory


def build_signal_context(
    ticker: str,
    sec_query: str = "revenue growth margins guidance AI",
) -> tuple[str, dict]:
    """Collect all raw signals and return (context_str, raw_signals_dict)."""

    # 1. SEC filings
    sec_result = retrieve_context(sec_query, ticker=ticker, k=5)
    sec_snippets = "\n".join(
        f"[SEC {doc.metadata.get('date')}] {doc.page_content[:200]}"
        for doc in sec_result["source_documents"]
    )

    # 2. Macro
    try:
        macro = get_latest_snapshot()
        macro_text = "\n".join(f"  {k}: {v}" for k, v in macro.items())
        macro_rate = float(macro.get("fed_funds_rate", 5.0))
        macro_yc = float(macro.get("yield_curve", -0.3))
    except (EnvironmentError, ValueError):
        macro_text = "  [FRED key not set — using defaults]"
        macro_rate, macro_yc = 5.25, -0.30

    # 3. News
    try:
        articles = fetch_news(f"{ticker} earnings revenue guidance", limit=5)
        news_text = "\n".join(f"  [{a['published_at']}] {a['title']}" for a in articles)
        news_sentiment = 0.3
    except EnvironmentError:
        news_text = "  [NEWSDATA_API_KEY not set]"
        news_sentiment = 0.0

    # 4. GitHub
    try:
        repos = fetch_github_trends(f"{ticker} technology", limit=5)
        github_text = "\n".join(f"  {r['name']} ({r['stars']:,} ⭐)" for r in repos)
        github_growth = sum(r["stars"] for r in repos) // max(len(repos), 1)
    except Exception:
        github_text = "  [GitHub fetch failed]"
        github_growth = 0

    context = (
        f"=== SEC Filings ({ticker}) ===\n{sec_snippets}\n\n"
        f"=== Macro Environment ===\n{macro_text}\n\n"
        f"=== News ===\n{news_text}\n\n"
        f"=== GitHub Trends ===\n{github_text}"
    )

    raw_signals = {
        "macro_rate": macro_rate,
        "macro_yield_curve": macro_yc,
        "news_sentiment": news_sentiment,
        "github_growth": github_growth,
    }

    return context, raw_signals


def run_pipeline(ticker: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Investment Signal Agent — {ticker}")
    print(f"{'='*60}\n")

    context, raw_signals = build_signal_context(ticker)

    # Signal alignment
    alignment = score_alignment(raw_signals)
    print(f"[signals] Composite: {alignment['composite']:+.2f} → {alignment['label']}")

    # OODA loop
    ooda_chain = build_ooda_chain()
    ooda_result = ooda_chain.invoke({"ticker": ticker, "signals": context})

    # Standard scoring
    scoring_chain = build_standard_scoring_chain()
    score_result = scoring_chain.invoke({"ticker": ticker, "signals": context})

    # ToT stance
    tot_chain = build_tot_stance_chain()
    stance_result = tot_chain.invoke({"ticker": ticker, "context": context})

    # Synthesis
    synth_chain = build_synthesis_chain()
    brief = synth_chain.invoke({
        "ticker": ticker,
        "signal_score": score_result,
        "signal_alignment": f"Composite: {alignment['composite']:+.2f}  {alignment['label']}",
        "tot_stance": stance_result,
        "ooda_decision": ooda_result,
    })

    # Write report
    report_path = generate_brief(
        ticker=ticker,
        synthesis=brief,
        signal_score=alignment["composite"] * 50 + 50,
        alignment_label=alignment["label"],
    )

    # Persist run
    mem = LongTermMemory()
    mem.store_run(
        ticker=ticker,
        signal_score=alignment["composite"] * 50 + 50,
        stance=alignment["label"],
        recommendation="BUY" if alignment["composite"] > 0.2 else "HOLD",
        summary=brief[:200],
    )

    print(f"\n✅ Pipeline complete. Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Investment Signal Agent")
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Stock ticker (default: NVDA)")
    args = parser.parse_args()
    run_pipeline(args.ticker.upper())


if __name__ == "__main__":
    main()
