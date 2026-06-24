"""Retrieval Agent.

Performs semantic retrieval over SEC EDGAR filings using the FAISS vector
store. Runs three query types per company, applies a similarity threshold
of ≥0.72 plus keyword confirmation, and returns the top-5 passages with
metadata. Designed to run in parallel with the Data Collector Agent.

Query types (per claud.md spec):
  1. capital_commitment     — capital expenditure, infrastructure spend
  2. platform_prioritization — strategic product bets, focus areas
  3. forward_guidance        — management outlook, next quarter/year
"""

import asyncio
from datetime import datetime, timezone

from schemas.messages import TaskAssignment, RetrievalReport, RetrievalPassage

SIMILARITY_THRESHOLD = 0.72
TOP_K = 5

QUERY_TEMPLATES = {
    "capital_commitment": [
        "capital expenditure infrastructure investment spending plans",
        "capex data center AI chip procurement budget",
    ],
    "platform_prioritization": [
        "strategic priority platform focus technology investment thesis",
        "product bet core platform competitive advantage roadmap",
    ],
    "forward_guidance": [
        "management guidance outlook next quarter revenue growth forecast",
        "forward looking statements revenue target projection",
    ],
}

# Keywords required for confirmation (at least one must appear in passage)
CONFIRMATION_KEYWORDS = {
    "capital_commitment": ["capex", "investment", "billion", "spend", "infrastructure", "build"],
    "platform_prioritization": ["platform", "priority", "strategy", "focus", "bet", "invest"],
    "forward_guidance": ["guidance", "expect", "outlook", "forecast", "target", "quarter"],
}


def _keyword_confirmed(text: str, query_type: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CONFIRMATION_KEYWORDS[query_type])


def _sec_score_from_passages(passages: list[RetrievalPassage]) -> float:
    """Convert passage similarity scores to a 0-100 SEC sub-score.

    Weighted by: average similarity of confirmed passages × 100,
    with a coverage bonus for having all 3 query types represented.
    """
    if not passages:
        return 0.0
    confirmed = [p for p in passages if p.keyword_confirmed]
    if not confirmed:
        return 20.0  # passages found but none keyword-confirmed
    avg_sim = sum(p.similarity_score for p in confirmed) / len(confirmed)
    query_types_covered = len({p.query_type for p in confirmed})
    coverage_bonus = (query_types_covered - 1) * 5.0  # +5 per additional type
    return min(100.0, round(avg_sim * 100.0 + coverage_bonus, 1))


class RetrievalAgent:
    """SEC EDGAR semantic retrieval agent with threshold and keyword filtering."""

    name = "RetrievalAgent"

    def run(self, task: TaskAssignment) -> RetrievalReport:
        from sec_rag.retriever import load_vectorstore

        print(f"[{self.name}] Retrieving SEC passages for '{task.category_label}' …")

        try:
            vectorstore = load_vectorstore()
        except Exception as e:
            print(f"  [RetrievalAgent] FAISS index unavailable: {e}")
            return RetrievalReport(
                run_id=task.run_id,
                category_id=task.category_id,
                sec_score=0.0,
                passages=[],
                passages_below_threshold=0,
            )

        all_passages: list[RetrievalPassage] = []
        below_threshold = 0

        for ticker in task.tickers:
            for query_type, queries in QUERY_TEMPLATES.items():
                for query in queries:
                    results = vectorstore.similarity_search_with_score(query, k=TOP_K * 3)
                    # Filter to this ticker
                    results = [
                        (doc, score) for doc, score in results
                        if doc.metadata.get("ticker") == ticker
                    ]
                    for doc, raw_score in results:
                        # FAISS returns L2 distance; convert to cosine-like similarity [0,1]
                        # Lower L2 distance = higher similarity; we normalize naively here.
                        # For inner-product indexes the score is already a similarity.
                        similarity = max(0.0, min(1.0, 1.0 - raw_score / 2.0))
                        if similarity < SIMILARITY_THRESHOLD:
                            below_threshold += 1
                            continue
                        confirmed = _keyword_confirmed(doc.page_content, query_type)
                        all_passages.append(RetrievalPassage(
                            ticker=ticker,
                            date=doc.metadata.get("date", ""),
                            accession=doc.metadata.get("accession", ""),
                            query_type=query_type,
                            content=doc.page_content[:500],
                            similarity_score=round(similarity, 4),
                            keyword_confirmed=confirmed,
                        ))

        # Deduplicate and keep top-K by similarity
        seen = set()
        unique_passages = []
        for p in sorted(all_passages, key=lambda x: x.similarity_score, reverse=True):
            key = (p.ticker, p.content[:100])
            if key not in seen:
                seen.add(key)
                unique_passages.append(p)
            if len(unique_passages) >= TOP_K:
                break

        sec_score = _sec_score_from_passages(unique_passages)
        print(f"  SEC score={sec_score:.0f}  passages={len(unique_passages)}  "
              f"below_threshold={below_threshold}")

        return RetrievalReport(
            run_id=task.run_id,
            category_id=task.category_id,
            sec_score=sec_score,
            passages=unique_passages,
            passages_below_threshold=below_threshold,
        )

    async def run_async(self, task: TaskAssignment) -> RetrievalReport:
        return await asyncio.to_thread(self.run, task)


if __name__ == "__main__":
    task = TaskAssignment(
        run_id="test-001",
        category_id="ai_ml_infrastructure",
        category_label="AI / ML Infrastructure",
        tickers=["NVDA", "AMD"],
        query_terms=["AI infrastructure", "GPU compute"],
        cycle_date=datetime.now(timezone.utc).date().isoformat(),
    )
    agent = RetrievalAgent()
    report = agent.run(task)
    print(f"\nRetrievalReport:")
    print(f"  SEC score:  {report.sec_score}")
    print(f"  Passages:   {len(report.passages)}")
    print(f"  Filtered:   {report.passages_below_threshold}")
    for p in report.passages[:2]:
        print(f"  [{p.ticker}|{p.query_type}|sim={p.similarity_score:.3f}|confirmed={p.keyword_confirmed}]")
        print(f"    {p.content[:120]}…")
