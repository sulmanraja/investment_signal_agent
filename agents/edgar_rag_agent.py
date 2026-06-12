"""EDGAR RAG Agent — retrieves SEC filing context for a given ticker/query.

This agent wraps the edgar_retriever tool and exposes a simple .run()
interface compatible with the CrewAI agent pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.edgar_retriever import retrieve_context


class EdgarRagAgent:
    """Standalone agent that retrieves SEC MD&A context via the FAISS index."""

    name = "EdgarRagAgent"
    role = "SEC Filing Analyst"
    goal = "Retrieve and summarise relevant SEC filing excerpts for investment analysis."
    backstory = (
        "You are a meticulous financial analyst specialising in SEC filings. "
        "You surface key risk factors, management guidance, and financial metrics "
        "from 10-K and 10-Q reports to inform investment decisions."
    )

    def run(
        self,
        query: str,
        ticker: str | None = None,
        k: int = 5,
        use_llm: bool = False,
    ) -> dict:
        """Run the RAG retrieval.

        Args:
            query: The research question or search string.
            ticker: Optional ticker to filter results.
            k: Number of source documents to return.
            use_llm: If True, generate an LLM answer over retrieved context.

        Returns:
            Dict with "answer" (str | None) and "source_documents" (list).
        """
        print(f"[{self.name}] Retrieving context for: '{query}' (ticker={ticker})")
        result = retrieve_context(query, ticker=ticker, k=k, use_llm=use_llm)
        print(f"[{self.name}] Found {len(result['source_documents'])} documents.")
        return result


if __name__ == "__main__":
    agent = EdgarRagAgent()
    out = agent.run(
        "What did management say about AI infrastructure spending?",
        ticker="NVDA",
        k=3,
    )
    print(f"\nDocuments returned: {len(out['source_documents'])}")
    for doc in out["source_documents"]:
        meta = doc.metadata
        print(f"  [{meta.get('ticker')} | {meta.get('date')}] {doc.page_content[:120]}…")
