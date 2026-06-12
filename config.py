"""Root configuration for the Investment Signal Agent."""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
SEC_RAG_DIR = ROOT_DIR / "sec_rag"
REPORTS_DIR = ROOT_DIR / "output" / "reports"
RUN_STORE_PATH = ROOT_DIR / "memory" / "run_store.json"

# ── LLM ───────────────────────────────────────────────────────────────────────
AGENT_LLM = os.getenv("AGENT_LLM", "granite3.3:8b")

# ── SEC RAG ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
FAISS_PATH = str(SEC_RAG_DIR / "faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Tree of Thoughts ──────────────────────────────────────────────────────────
TOT_MAX_DEPTH = int(os.getenv("TOT_MAX_DEPTH", "2"))
TOT_BEAM_WIDTH = int(os.getenv("TOT_BEAM_WIDTH", "2"))

# ── Default tickers ───────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META"]
