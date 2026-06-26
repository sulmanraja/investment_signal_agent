"""MCP server — exposes the Investment Signal Agent as a Claude plugin.

Four tools are registered:
  list_investment_categories   — instant; returns available sectors
  search_sec_filings           — fast FAISS semantic search (~1 s)
  get_run_history              — instant; returns recent analysis results
  analyze_investment_categories — long-running (5-15 min); full 7-agent pipeline

Provider is controlled by env vars (set automatically via .mcp.json):
  AGENT_LLM_PROVIDER=anthropic   uses the Claude API (no Ollama needed)
  EMBEDDING_PROVIDER=huggingface  uses local sentence-transformers (no Ollama needed)
"""

from dotenv import load_dotenv
load_dotenv()  # pick up .env before any agent modules are imported

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Investment Signal Agent",
    instructions=(
        "Provides investment signal analysis for five technology sectors using "
        "SEC EDGAR filings, GitHub trends, news sentiment, Google Trends, and "
        "FRED macroeconomic data. "
        "Use list_investment_categories first to see available sectors. "
        "Use search_sec_filings for targeted SEC filing questions (fast). "
        "Use get_run_history to recall prior analysis results. "
        "Use analyze_investment_categories for a full pipeline run — "
        "this produces a complete Investment Horizon Report but takes 5-15 minutes."
    ),
)


@mcp.tool()
def list_investment_categories() -> list[dict]:
    """Return the five technology investment categories with IDs, labels, and tickers.

    Call this first to discover valid category IDs for analyze_investment_categories.
    """
    from signals.category_registry import load_registry
    cats = load_registry().get("technology_categories", {})
    return [
        {
            "id": cat_id,
            "label": cat.get("label", cat_id),
            "tickers": cat.get("representative_tickers", []),
            "query_terms": cat.get("query_terms", [])[:4],
        }
        for cat_id, cat in cats.items()
    ]


@mcp.tool()
def search_sec_filings(query: str, ticker: str = "") -> list[dict]:
    """Search SEC 10-K / 10-Q filings semantically using the FAISS vector index.

    Args:
        query:  Natural language question, e.g. "What did management say about AI capex?"
        ticker: Optional ticker symbol to narrow results, e.g. "NVDA" or "MSFT".
                Leave blank to search across all indexed companies.

    Returns up to 5 relevant passages with source metadata and similarity scores.
    Requires the FAISS index to exist (run `make run-ingest` to build it).
    """
    from sec_rag.retriever import load_vectorstore
    try:
        vs = load_vectorstore()
        results = vs.similarity_search_with_score(query, k=20)
        if ticker:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("ticker") == ticker.upper()
            ]
        return [
            {
                "ticker":     doc.metadata.get("ticker", ""),
                "date":       doc.metadata.get("date", ""),
                "form":       doc.metadata.get("form", ""),
                "similarity": round(max(0.0, 1.0 - score / 2.0), 3),
                "content":    doc.page_content[:500],
            }
            for doc, score in results[:5]
        ]
    except FileNotFoundError:
        return [{"error": "FAISS index not found. Run `make run-ingest` to build it."}]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def get_run_history(limit: int = 5) -> list[dict]:
    """Return recent investment analysis results stored in long-term memory.

    Args:
        limit: Number of most-recent runs to return (default 5, max 20).

    Each record includes category, stance (BUY/HOLD/REDUCE), score, and a summary.
    """
    from memory.long_term import LongTermMemory
    try:
        runs = LongTermMemory().get_latest(n=min(limit, 20))
        return runs if runs else [{"message": "No prior runs found."}]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
async def analyze_investment_categories(
    categories: list[str] | None = None,
) -> str:
    """Run the full 7-agent investment analysis pipeline and return a Markdown report.

    Agents: Orchestrator → DataCollector (GitHub / News / Trends / FRED / SEC EDGAR)
            → RetrievalAgent → SignalAnalyst → ThoughtGenerator → Critic → Synthesis.

    Args:
        categories: Category IDs to analyse. Omit to run all five.
                    Valid IDs: ai_ml_infrastructure, cloud_edge, semiconductors,
                               developer_tooling, cybersecurity

    Returns:
        A complete Technology Investment Horizon Report in Markdown format.

    Warning: This tool takes 5-15 minutes depending on the number of categories.
             Claude will wait for the result before responding.
    """
    from agents.orchestrator_agent import OrchestratorAgent
    # Call the async method directly to avoid nesting asyncio.run() inside FastMCP's loop.
    return await OrchestratorAgent().run(category_ids=categories or None)


if __name__ == "__main__":
    mcp.run(transport="stdio")
