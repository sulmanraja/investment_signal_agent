"""Thin wrapper around sec_rag.ingest for use within the agent pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sec_rag.ingest import ingest, ingest_incremental


def ingest_tickers(
    tickers: list[str],
    form_types: list[str] = ["10-K", "10-Q"],
    max_filings: int = 3,
    incremental: bool = False,
) -> None:
    """Ingest SEC filings for the given tickers into the FAISS index.

    Args:
        tickers: List of stock ticker symbols, e.g. ["AAPL", "MSFT"].
        form_types: SEC form types to fetch.
        max_filings: Maximum filings per ticker.
        incremental: If True, merge into existing index rather than rebuild.
    """
    if incremental:
        ingest_incremental(tickers, form_types=form_types, max_filings=max_filings)
    else:
        ingest(tickers, form_types=form_types, max_filings_per_ticker=max_filings)


if __name__ == "__main__":
    print("=== edgar_ingest local test ===")
    print("Ingesting 1 filing for NVDA (10-K only) …")
    ingest_tickers(["NVDA"], form_types=["10-K"], max_filings=1)
