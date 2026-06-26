"""Data Collector Agent.

Collects signals from all five sources in parallel. Each source is scored
and narrated by an inline CrewAI agent (Agent + Task + Crew). The CrewAI
agent produces a narrative analysis that ends with "Score: X/100"; that
integer is extracted as the sub-score. The full narrative becomes the
per-source rationale concatenated into SubScoreReport.signal_context.

Math-formula fallbacks are used when the CrewAI call fails or the score
line cannot be parsed, ensuring the pipeline never stalls.

Sources:
  - GitHub trending repos      → CrewAI GitHub Trend Analyst
  - NewsData.io articles       → CrewAI News Sentiment Analyst
  - Google Trends              → CrewAI Search Trend Analyst
  - FRED macro indicators      → CrewAI Macro Economist
  - SEC EDGAR                  → FAISS retrieval (edgar_retriever) + CrewAI SEC Filing Analyst
"""

import asyncio
import os
import re
from datetime import datetime, timezone

from crewai import Agent, Task, Crew

from schemas.messages import TaskAssignment, SubScoreReport
from tools.edgar_retriever import retrieve_context as _edgar_retrieve
from utils.llm_factory import make_llm


# ── Score parser ───────────────────────────────────────────────────────────────

def _crew_score(narrative: str, fallback: float) -> tuple[float, str]:
    """Extract 'Score: X/100' from CrewAI narrative. Returns (score, narrative).

    If the line is not found the math fallback score is used but the full
    narrative is still returned for signal_context.
    """
    match = re.search(r"Score:\s*(\d+)\s*/\s*100", narrative, re.IGNORECASE)
    if match:
        score = max(0.0, min(100.0, float(match.group(1))))
        return score, narrative.strip()
    return fallback, narrative.strip()


# ── Math fallback normalizers (used when CrewAI call fails) ───────────────────

def _math_github_score(repos: list[dict]) -> float:
    if not repos:
        return 0.0
    return min(100.0, sum(r.get("stars", 0) for r in repos) / len(repos) / 50.0)


def _math_news_score(articles: list[dict]) -> float:
    sentiments = []
    for a in articles:
        try:
            sentiments.append(float(a["sentiment"]))
        except (TypeError, ValueError, KeyError):
            pass
    if not sentiments:
        return 50.0
    return round(50.0 + sum(sentiments) / len(sentiments) * 50.0, 1)


def _math_trends_score(data: dict) -> float:
    iot = data.get("interest_over_time", {})
    values = []
    for kw_data in iot.values():
        if isinstance(kw_data, dict):
            values.extend(v for v in kw_data.values() if isinstance(v, (int, float)))
    return round(sum(values) / len(values), 1) if values else 50.0


def _math_macro_score(snapshot: dict) -> float:
    try:
        rate = float(snapshot.get("fed_funds_rate", 5.0))
        yc = float(snapshot.get("yield_curve", 0.0))
        unemp = float(snapshot.get("unemployment", 4.0))
        score = 50.0 - (rate - 3.0) * 3.0 + yc * 10.0 - (unemp - 4.0) * 5.0
        return max(0.0, min(100.0, round(score, 1)))
    except (TypeError, ValueError):
        return 50.0


def _math_edgar_score(results_by_ticker: dict) -> float:
    if not results_by_ticker:
        return 0.0
    scores = []
    for result in results_by_ticker.values():
        docs = result.get("source_documents", [])
        if not docs:
            scores.append(0.0)
            continue
        coverage = min(1.0, len(docs) / 5.0)
        avg_len = sum(len(d.page_content) for d in docs) / len(docs)
        quality = min(1.0, avg_len / 500.0)
        scores.append((coverage * 0.7 + quality * 0.3) * 100.0)
    return round(sum(scores) / len(scores), 1) if scores else 0.0


# ── Async collection functions ─────────────────────────────────────────────────

async def _collect_github(
    category: str, query_terms: list[str]
) -> tuple[float, list[dict], str, bool]:
    from tools.github_tool import fetch_github_trends
    topic = " ".join(query_terms[:2])
    try:
        repos = await asyncio.to_thread(fetch_github_trends, topic, limit=10)
    except Exception as e:
        print(f"  [DataCollector] GitHub API failed: {e}")
        return 50.0, [], "GitHub data unavailable.", False

    repo_text = "\n".join(
        f"  {r.get('name', '?')} ({r.get('stars', 0):,} stars): {str(r.get('description', ''))[:80]}"
        for r in repos
    ) or "  No trending repositories found."

    math_fallback = _math_github_score(repos)

    def _run() -> str:
        llm = make_llm(0.3)
        agent = Agent(
            role="GitHub Trend Analyst",
            goal="Interpret GitHub repository trends to identify technology adoption momentum as an investment signal.",
            backstory=(
                "You are a tech-savvy analyst who monitors open-source activity as a "
                "leading indicator of enterprise technology adoption. You translate "
                "star counts, fork rates, and repo growth into investment insights."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=(
                f"Analyse the following GitHub trending repositories for the technology "
                f"category '{category}':\n\n{repo_text}\n\n"
                "Provide:\n"
                "1. What technology themes are gaining momentum?\n"
                "2. How does this activity signal developer/enterprise adoption?\n"
                "3. Investment implication (1-2 sentences).\n\n"
                "End your response with exactly this line: Score: X/100 "
                "(where X is 0-100 reflecting adoption momentum; high stars and growth = high score)."
            ),
            agent=agent,
            expected_output="Narrative analysis ending with 'Score: X/100'.",
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

    try:
        narrative = await asyncio.to_thread(_run)
        score, rationale = _crew_score(narrative, math_fallback)
    except Exception as e:
        print(f"  [DataCollector] GitHub CrewAI failed: {e}")
        score, rationale = math_fallback, f"GitHub momentum (math fallback): {math_fallback:.0f}/100."

    return score, repos, rationale, True


async def _collect_news(
    category: str, query_terms: list[str]
) -> tuple[float, list[dict], str, bool]:
    from tools.newsdata_tool import fetch_news
    query = " ".join(query_terms[:3])
    try:
        articles = await asyncio.to_thread(fetch_news, query, limit=10)
    except Exception as e:
        print(f"  [DataCollector] News API failed: {e}")
        return 50.0, [], "News data unavailable.", False

    headlines = "\n".join(
        f"  [{a.get('published_at', '?')}] {a.get('source', '?')}: {a.get('title', '')}"
        for a in articles
    ) or "  No recent articles found."

    math_fallback = _math_news_score(articles)

    def _run() -> str:
        llm = make_llm(0.3)
        agent = Agent(
            role="News Sentiment Analyst",
            goal="Analyse recent news to extract sentiment signals and assess their investment implications.",
            backstory=(
                "You are a news-driven analyst who monitors business and technology "
                "news to identify events that materially impact technology investment decisions. "
                "You rate sentiment and flag key catalysts."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )
        task = Task(
            description=(
                f"Analyse the following recent news headlines for the technology "
                f"category '{category}':\n\n{headlines}\n\n"
                "Provide:\n"
                "1. Overall sentiment: Positive / Neutral / Negative.\n"
                "2. Top 2-3 material events or themes.\n"
                "3. Investment implication (1-2 sentences).\n\n"
                "End your response with exactly this line: Score: X/100 "
                "(where X is 0=very negative, 50=neutral, 100=very positive)."
            ),
            agent=agent,
            expected_output="Sentiment analysis ending with 'Score: X/100'.",
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

    try:
        narrative = await asyncio.to_thread(_run)
        score, rationale = _crew_score(narrative, math_fallback)
    except Exception as e:
        print(f"  [DataCollector] News CrewAI failed: {e}")
        score, rationale = math_fallback, f"News sentiment (math fallback): {math_fallback:.0f}/100."

    return score, articles, rationale, True


async def _collect_trends(
    category: str, query_terms: list[str]
) -> tuple[float, dict, str, bool]:
    from tools.google_trends_tool import fetch_google_trends
    keywords = query_terms[:5]
    try:
        data = await asyncio.to_thread(fetch_google_trends, keywords)
    except Exception as e:
        print(f"  [DataCollector] Google Trends failed: {e}")
        return 50.0, {}, "Google Trends data unavailable.", False

    iot = data.get("interest_over_time", {})
    if iot:
        peak_kw = max(iot.keys(), key=lambda k: sum(iot[k].values()) if isinstance(iot[k], dict) else 0)
        interest_summary = f"Keywords: {', '.join(keywords)}. Peak interest keyword: '{peak_kw}'."
    else:
        interest_summary = f"Keywords: {', '.join(keywords)}. No trend data returned."

    math_fallback = _math_trends_score(data)

    def _run() -> str:
        llm = make_llm(0.3)
        agent = Agent(
            role="Search Trend Analyst",
            goal="Interpret Google Trends search interest data as a signal of enterprise and developer adoption.",
            backstory=(
                "You are a market analyst who tracks search interest trends as a proxy "
                "for technology awareness, adoption momentum, and investment opportunity."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=(
                f"Interpret the following Google Trends data for the technology "
                f"category '{category}':\n\n{interest_summary}\n\n"
                "Provide:\n"
                "1. What does the search interest pattern signal about adoption momentum?\n"
                "2. Is this trend accelerating, stable, or declining?\n"
                "3. Investment implication (1-2 sentences).\n\n"
                "End your response with exactly this line: Score: X/100 "
                "(where X is 0-100 reflecting search momentum strength)."
            ),
            agent=agent,
            expected_output="Trend analysis ending with 'Score: X/100'.",
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

    try:
        narrative = await asyncio.to_thread(_run)
        score, rationale = _crew_score(narrative, math_fallback)
    except Exception as e:
        print(f"  [DataCollector] Trends CrewAI failed: {e}")
        score, rationale = math_fallback, f"Google Trends (math fallback): {math_fallback:.0f}/100."

    return score, data, rationale, True


async def _collect_macro(category: str) -> tuple[float, dict, str, bool]:
    from tools.fred_macro_tool import get_latest_snapshot
    try:
        snapshot = await asyncio.to_thread(get_latest_snapshot)
    except Exception as e:
        print(f"  [DataCollector] FRED API failed: {e}")
        return 50.0, {}, "Macro data unavailable.", False

    indicators = "\n".join(f"  {k:<28s} {v}" for k, v in snapshot.items())
    math_fallback = _math_macro_score(snapshot)

    def _run() -> str:
        llm = make_llm(0.2)
        agent = Agent(
            role="Macro Economist",
            goal=(
                "Interpret macroeconomic indicators to assess the top-down investment "
                "environment for technology growth assets."
            ),
            backstory=(
                "You are a top-down macro strategist who monitors the Federal Reserve, "
                "yield curve dynamics, inflation, and credit markets to contextualise "
                "technology equity valuations and sector rotation signals."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=(
                f"Assess the macro environment for the technology category '{category}' "
                f"given these current indicators:\n\n{indicators}\n\n"
                "Provide:\n"
                "1. Is the rate/inflation environment a tailwind or headwind for growth tech?\n"
                "2. What does the yield curve signal about near-term risk?\n"
                "3. Overall macro stance: Supportive / Neutral / Headwind (1-2 sentences).\n\n"
                "End your response with exactly this line: Score: X/100 "
                "(where X is 0=severe headwind, 50=neutral, 100=strong tailwind)."
            ),
            agent=agent,
            expected_output="Macro assessment ending with 'Score: X/100'.",
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

    try:
        narrative = await asyncio.to_thread(_run)
        score, rationale = _crew_score(narrative, math_fallback)
    except Exception as e:
        print(f"  [DataCollector] Macro CrewAI failed: {e}")
        score, rationale = math_fallback, f"Macro environment (math fallback): {math_fallback:.0f}/100."

    return score, snapshot, rationale, True


async def _collect_edgar(
    category: str, tickers: list[str], query_terms: list[str]
) -> tuple[float, dict, str, bool]:
    """Retrieve SEC EDGAR excerpts via edgar_retriever, then score via CrewAI."""
    query = " ".join(query_terms[:4]) + " investment guidance capital expenditure"

    async def _query_ticker(ticker: str) -> tuple[str, dict]:
        try:
            result = await asyncio.to_thread(_edgar_retrieve, query, ticker, 5, False)
            print(f"  [DataCollector] EDGAR {ticker}: {len(result.get('source_documents', []))} doc(s)")
            return ticker, result
        except Exception as e:
            print(f"  [DataCollector] EDGAR {ticker} failed: {e}")
            return ticker, {"source_documents": []}

    pairs = await asyncio.gather(*[_query_ticker(t) for t in tickers])
    results_by_ticker = dict(pairs)

    all_docs = [
        (ticker, doc)
        for ticker, res in results_by_ticker.items()
        for doc in res.get("source_documents", [])[:2]
    ]
    excerpts_text = "\n".join(
        f"  [{ticker}] {doc.page_content[:250]}"
        for ticker, doc in all_docs[:6]
    ) or "  No SEC filing excerpts retrieved."

    math_fallback = _math_edgar_score(results_by_ticker)

    def _run() -> str:
        llm = make_llm(0.2)
        agent = Agent(
            role="SEC Filing Analyst",
            goal=(
                "Analyse SEC 10-K and 10-Q filing excerpts to assess capital commitment "
                "and strategic conviction for investment analysis."
            ),
            backstory=(
                "You are a meticulous financial analyst specialising in SEC filings. "
                "You surface key risk factors, management guidance, and capital expenditure "
                "signals from 10-K and 10-Q reports to inform investment decisions."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=(
                f"Analyse the following SEC filing excerpts for the technology "
                f"category '{category}':\n\n{excerpts_text}\n\n"
                "Provide:\n"
                "1. What do these filings signal about management's capital commitment to this technology?\n"
                "2. Are there material risk factors or forward guidance statements?\n"
                "3. Investment implication (1-2 sentences).\n\n"
                "End your response with exactly this line: Score: X/100 "
                "(where X is 0-100; high capex commitment and positive guidance = high score)."
            ),
            agent=agent,
            expected_output="Filing analysis ending with 'Score: X/100'.",
        )
        return str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

    try:
        narrative = await asyncio.to_thread(_run)
        score, rationale = _crew_score(narrative, math_fallback)
    except Exception as e:
        print(f"  [DataCollector] EDGAR CrewAI failed: {e}")
        score, rationale = math_fallback, f"SEC EDGAR (math fallback): {math_fallback:.0f}/100."

    raw = {
        ticker: {
            "sec_edgar_excerpts": [
                {"date": getattr(d, "metadata", {}).get("date", ""),
                 "content": d.page_content[:300]}
                for d in res.get("source_documents", [])[:2]
            ]
        }
        for ticker, res in results_by_ticker.items()
    }
    any_docs = any(v.get("sec_edgar_excerpts") for v in raw.values())
    return score, raw, rationale, any_docs


# ── Main agent ─────────────────────────────────────────────────────────────────

class DataCollectorAgent:
    """Collects and scores signals from all five sources in parallel.

    Each source is fetched via its API tool, then analysed by an inline
    CrewAI agent that produces a narrative ending with 'Score: X/100'.
    The score is parsed from that line; the full narrative becomes the
    per-source rationale concatenated into SubScoreReport.signal_context,
    which feeds the Thought Generator and Critic agents.
    """

    name = "DataCollectorAgent"

    async def run(self, task: TaskAssignment) -> SubScoreReport:
        from guardrails.data_integrity import (
            build_attribution, check_news_staleness, check_edgar_staleness,
        )

        print(f"[{self.name}] Collecting data for '{task.category_label}' …")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        label = task.category_label

        (
            (github_score,  github_raw,  github_rationale,  github_ok),
            (news_score,    news_raw,    news_rationale,    news_ok),
            (trends_score,  trends_raw,  trends_rationale,  trends_ok),
            (macro_score,   macro_raw,   macro_rationale,   macro_ok),
            (edgar_score,   edgar_raw,   edgar_rationale,   edgar_ok),
        ) = await asyncio.gather(
            _collect_github(label, task.query_terms),
            _collect_news(label, task.query_terms),
            _collect_trends(label, task.query_terms),
            _collect_macro(label),
            _collect_edgar(label, task.tickers, task.query_terms),
        )

        print(
            f"  GitHub={github_score:.0f}  News={news_score:.0f}  "
            f"Trends={trends_score:.0f}  Macro={macro_score:.0f}  "
            f"SEC_EDGAR={edgar_score:.0f}"
        )

        # ── Data integrity: source attribution ────────────────────────────────
        query = " ".join(task.query_terms[:3])
        attributions = [
            build_attribution("github",       "github_tool.fetch_github_trends",        query, len(github_raw),  github_ok,  retrieved_at),
            build_attribution("news",          "newsdata_tool.fetch_news",               query, len(news_raw),    news_ok,    retrieved_at),
            build_attribution("google_trends", "google_trends_tool.fetch_google_trends", query, int(bool(trends_raw)), trends_ok, retrieved_at),
            build_attribution("fred_macro",    "fred_macro_tool.get_latest_snapshot",    "",    int(bool(macro_raw)),  macro_ok,  retrieved_at),
            build_attribution("sec_edgar",     "tools.edgar_retriever.retrieve_context", query,
                              sum(len(v.get("sec_edgar_excerpts", [])) for v in edgar_raw.values()),
                              edgar_ok, retrieved_at),
        ]
        failed_sources = [a.source_key for a in attributions if a.status == "failed"]

        # ── Data integrity: staleness flags ───────────────────────────────────
        staleness_flags = [
            check_news_staleness(news_raw, retrieved_at),
            check_edgar_staleness(edgar_raw, retrieved_at),
        ]

        signal_context = (
            f"=== {label} — Signal Summary ({task.cycle_date}) ===\n\n"
            f"GitHub (score={github_score:.0f}/100):\n{github_rationale}\n\n"
            f"News (score={news_score:.0f}/100):\n{news_rationale}\n\n"
            f"Google Trends (score={trends_score:.0f}/100):\n{trends_rationale}\n\n"
            f"FRED Macro (score={macro_score:.0f}/100):\n{macro_rationale}\n\n"
            f"SEC EDGAR (score={edgar_score:.0f}/100):\n{edgar_rationale}"
        )

        return SubScoreReport(
            run_id=task.run_id,
            category_id=task.category_id,
            github_score=github_score,
            news_score=news_score,
            google_trends_score=trends_score,
            fred_macro_score=macro_score,
            sec_edgar_score=edgar_score,
            signal_context=signal_context,
            source_attributions=attributions,
            staleness_flags=staleness_flags,
            failed_sources=failed_sources,
            raw_evidence={
                "github_repos":       github_raw[:3],
                "news_articles":      [a.get("title") for a in news_raw[:3]],
                "google_trends_data": trends_raw,
                "macro_snapshot":     macro_raw,
                "sec_edgar":          edgar_raw,
            },
        )


def collect_data(task: TaskAssignment) -> SubScoreReport:
    """Synchronous wrapper for non-async callers."""
    return asyncio.run(DataCollectorAgent().run(task))


if __name__ == "__main__":
    task = TaskAssignment(
        run_id="test-001",
        category_id="ai_ml_infrastructure",
        category_label="AI / ML Infrastructure",
        tickers=["NVDA", "AMD"],
        query_terms=["AI infrastructure", "GPU compute", "LLM training"],
        cycle_date=datetime.now(timezone.utc).date().isoformat(),
    )
    report = collect_data(task)
    print(f"\nSubScoreReport:")
    print(f"  GitHub:        {report.github_score}")
    print(f"  News:          {report.news_score}")
    print(f"  Google Trends: {report.google_trends_score}")
    print(f"  FRED Macro:    {report.fred_macro_score}")
    print(f"  SEC EDGAR:     {report.sec_edgar_score}")
    print(f"\nSignal Context:\n{report.signal_context}")
