"""Thin wrapper around sec_rag.retriever for use within the agent pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_rag.retriever import semantic_search, rag_query, build_rag_chain


def retrieve_context(
    query: str,
    ticker: str = '',
    k: int = 5,
    use_llm: bool = False,
) -> dict:
    """Retrieve relevant SEC filing context for a query.

    Args:
        query: Natural-language question or search string.
        ticker: Optional ticker filter (e.g. "AAPL").
        k: Number of results to return.
        use_llm: If True, run a full RAG query through the LLM.

    Returns:
        Dict with "answer" (str) and "source_documents" (list).
    """
    if use_llm:
        return rag_query(query, ticker=ticker)

    results = semantic_search(query, ticker=ticker, k=k)
    return {
        "answer": None,
        "source_documents": [doc for doc, _ in results],
    }


if __name__ == "__main__":
    print("=== edgar_retriever local test ===")
    result = retrieve_context(
        "capital expenditure guidance AI infrastructure",
        ticker="NVDA",
        k=3,
    )
    print(f"\nReturned {len(result['source_documents'])} documents.")
