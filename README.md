# Investment Signal Agent

A multi-agent investment analysis system that aggregates signals from SEC filings, macroeconomic data, GitHub trends, and news to generate structured investment briefs using a Tree-of-Thoughts reasoning pipeline.

---

## Architecture

```
Raw signals (5 sources)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  tools/                                                          │
│  edgar_ingest ──► edgar_retriever   (SEC 10-K / 10-Q via FAISS) │
│  fred_macro_tool                    (FRED economic indicators)   │
│  newsdata_tool                      (NewsData.io articles)       │
│  github_tool                        (GitHub trending repos)      │
│  google_trends_tool                 (Google Trends interest)     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  signals/                                                        │
│  normalizer ──► alignment_scorer    ([-1,1] weighted composite)  │
└──────────────────────────────────────────────────────────────────┘
       │
       ├──► chains/ooda_loop          (Observe → Orient → Decide → Act)
       │
       ├──► chains/standard_scoring   (0-100 score across 5 dimensions)
       │
       └──► chains/tot_stance_chain
                  │
                  ▼
            tot/ (Tree of Thoughts beam search)
            branches ──► evaluator ──► pruner (beam width=2)
                  │
                  ▼
            chains/synthesis_chain    (final investment brief)
                  │
                  ▼
            output/investment_brief   (Markdown report → output/reports/)
                  │
                  ▼
            memory/long_term          (run_store.json persistence)
```

**Agents** (CrewAI) wrap each tool and provide role-based reasoning:
`edgar_rag_agent` · `thought_generator_agent` · `critic_agent` · `github_trends_agent` · `news_agent` · `macro_agent`

**LLM** — all chains and agents use a local Ollama model (default: `granite3.3:8b`). No cloud LLM required.

---

## Project Structure

```
investment_signal_agent/
├── agents/                     # CrewAI agent roles
│   ├── edgar_rag_agent.py      # SEC filing retrieval agent
│   ├── thought_generator_agent.py  # A/B/C/D thesis branch generation
│   ├── critic_agent.py         # Thesis scoring and ranking
│   ├── github_trends_agent.py  # Developer adoption signal
│   ├── news_agent.py           # News sentiment
│   └── macro_agent.py          # Macro environment assessment
├── tot/                        # Tree of Thoughts beam search
│   ├── thought_node.py         # Dataclass for a thesis node
│   ├── branches.py             # Generates A/B/C/D stances (LLM)
│   ├── evaluator.py            # Scores nodes with rubric (LLM)
│   ├── pruner.py               # Beam pruning (width=2)
│   └── beam_search.py          # LCEL beam search orchestrator
├── chains/                     # LangChain LCEL chains
│   ├── ooda_loop.py            # OODA loop chain
│   ├── signal_alignment.py     # Cross-source alignment scoring
│   ├── standard_scoring.py     # 0-100 signal score
│   ├── tot_stance_chain.py     # ToT → formatted stance
│   └── synthesis_chain.py      # Final brief synthesizer
├── tools/                      # Data source integrations
│   ├── edgar_ingest.py         # Wraps sec_rag/ingest.py
│   ├── edgar_retriever.py      # Wraps sec_rag/retriever.py
│   ├── github_tool.py          # GitHub API trending repos
│   ├── google_trends_tool.py   # Google Trends via pytrends
│   ├── newsdata_tool.py        # NewsData.io API
│   └── fred_macro_tool.py      # FRED economic indicators
├── signals/                    # Signal normalisation and scoring
│   ├── normalizer.py           # Maps raw values to [-1, 1]
│   ├── alignment_scorer.py     # Weighted composite alignment
│   ├── category_registry.py    # Registry loader
│   └── category_registry.json  # Signal metadata and weights
├── memory/
│   ├── short_term.py           # In-process message buffer
│   ├── long_term.py            # JSON-backed run persistence
│   └── run_store.json          # Auto-created on first run
├── output/
│   ├── investment_brief.py     # Markdown brief writer
│   └── reports/                # Generated briefs (git-ignored)
├── sec_rag/                    # SEC EDGAR RAG subsystem
│   ├── ingest.py               # Fetch 10-K/10-Q → FAISS index
│   ├── retriever.py            # Semantic search + RAG queries
│   ├── config.py               # Embedding model, paths, chunk size
│   └── faiss_index/            # Auto-created after ingestion
├── tests/                      # Local test scripts (no LLM needed)
│   ├── test_signals.py
│   ├── test_memory.py
│   ├── test_tot.py
│   ├── test_output.py
│   └── test_tools_noapi.py
├── main.py                     # Full pipeline entry point
├── config.py                   # Root configuration
├── requirements.txt            # All dependencies
├── Makefile                    # Dev shortcuts
├── .env.example                # API key template
└── .vscode/
    ├── settings.json           # Python interpreter, formatting
    ├── launch.json             # 30 run/debug configurations
    └── extensions.json         # Recommended extensions
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.13 recommended. [python.org](https://www.python.org/downloads/) |
| **Ollama** | Local LLM runtime. [ollama.com](https://ollama.com/download) |
| **Ollama models** | `granite3.3:8b` (LLM) + `nomic-embed-text` (embeddings) |
| **FRED API key** | Free at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **NewsData API key** | Free tier at [newsdata.io](https://newsdata.io/) |
| **GitHub token** | Optional — raises rate limits. [github.com/settings/tokens](https://github.com/settings/tokens) |

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd investment_signal_agent
```

### 2. Create and activate the virtual environment

```bash
# Create and install all dependencies in one step:
make setup

# Or manually:
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

> **VS Code** picks up `.venv` automatically via `.vscode/settings.json`.
> If it doesn't: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

### 3. Pull Ollama models

```bash
ollama pull granite3.3:8b      # default LLM (~5GB)
ollama pull nomic-embed-text   # embedding model (~270MB)
```

Verify Ollama is running:

```bash
ollama list
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

`.env` contents:

```bash
AGENT_LLM=granite3.3:8b
EMBEDDING_MODEL=nomic-embed-text

FRED_API_KEY=your_key_here
NEWSDATA_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here    # optional
```

---

## Verifying Your Setup

```bash
# Check all key imports resolve correctly:
make check
```

Expected output:

```
=== Environment check ===
Python 3.13.9
  langchain 1.3.4
  langchain_ollama ok
  faiss ok
  requests ok
  beautifulsoup4 ok
```

---

## Running Locally

### Run all no-dependency tests (no LLM, no API keys)

```bash
make test
```

Tests cover: signal normalisation, memory persistence, ToT dataclasses, output generation, and tool API-key guards.

### Component-by-component testing

Each module has a `__main__` block you can run directly. The Makefile provides shortcuts for all of them:

| Component | Command | Requires |
|---|---|---|
| **SEC RAG: Ingest** | `make run-ingest` | Ollama |
| **SEC RAG: Retriever** | `make run-retriever` | Ollama + FAISS index |
| **Tool: GitHub Trends** | `make run-github` | network |
| **Tool: Google Trends** | `make run-trends` | network |
| **Tool: News** | `make run-news` | `NEWSDATA_API_KEY` |
| **Tool: FRED Macro** | `make run-macro` | `FRED_API_KEY` |
| **Agent: EDGAR RAG** | `make run-edgar-agent` | Ollama + FAISS index |
| **Agent: Thought Generator** | `make run-thought-agent` | Ollama |
| **Agent: Critic** | `make run-critic-agent` | Ollama |
| **Agent: GitHub Trends** | `make run-github-agent` | Ollama + network |
| **Agent: News** | `make run-news-agent` | Ollama + `NEWSDATA_API_KEY` |
| **Agent: Macro** | `make run-macro-agent` | Ollama + `FRED_API_KEY` |
| **ToT: Beam Search** | `make run-tot` | Ollama |
| **Chain: OODA Loop** | `make run-ooda` | Ollama |
| **Chain: Standard Scoring** | `make run-scoring` | Ollama |
| **Chain: Synthesis** | `make run-synthesis` | Ollama |
| **Signals: Alignment** | `make test-signals` | none |

### Run the full pipeline

```bash
# Default: NVDA
make run-nvda

# Any ticker:
TICKER=AAPL make run

# Or directly:
.venv/bin/python main.py MSFT
```

The pipeline runs in order: signal collection → normalisation → OODA → scoring → ToT beam search → synthesis → writes a Markdown brief to `output/reports/`.

---

## VS Code Debug Panel

All 30 run/debug configurations are in `.vscode/launch.json`. Open the Debug panel with `Cmd+Shift+D` (Mac) or `Ctrl+Shift+D` (Windows/Linux).

Key configurations:

| Name | What it runs |
|---|---|
| `Run Pipeline (NVDA)` | `main.py NVDA` |
| `Run Pipeline (custom ticker)` | prompts for any ticker |
| `SEC RAG: Ingest` | `sec_rag.ingest` |
| `SEC RAG: Retriever` | `sec_rag.retriever` |
| `Agent: EDGAR RAG` | `agents.edgar_rag_agent` |
| `ToT: Beam Search` | `tot.beam_search` |
| `Chain: OODA Loop` | `chains.ooda_loop` |
| `Test: All (no-dep)` | all 5 no-dependency test suites |

Press `F5` to run the selected configuration with full debugger support.

---

## SEC RAG: Building the FAISS Index

The `sec_rag/` module downloads MD&A sections from SEC 10-K and 10-Q filings and indexes them locally with FAISS.

### Ingest filings

```bash
make run-ingest
# or
.venv/bin/python -m sec_rag.ingest
```

By default (in `sec_rag/ingest.py`) it ingests the last 10 filings per ticker for the S&P tech names: `AAPL MSFT NVDA GOOG AMZN META INTC CSCO ADBE`.

To ingest a custom list, edit the `__main__` block or call from Python:

```python
from sec_rag.ingest import ingest
ingest(["NVDA", "AMD"], form_types=["10-K"], max_filings_per_ticker=3)
```

To add tickers to an existing index without rebuilding:

```python
from sec_rag.ingest import ingest_incremental
ingest_incremental(["TSM", "ASML"])
```

Expected output:

```
Processing NVDA...
  CIK for NVDA: 0001045810
  Found 3 filings (10-K, 10-Q)
  [10-K] 2024-02-21 — 0001045810-24-000029
    MD&A extracted: ~18,400 words
    ✓ 68 chunks indexed

Building FAISS index over 204 total chunks...
✅ Index saved to 'sec_rag/faiss_index/'
```

### Query the index

```bash
make run-retriever
```

Or in Python:

```python
from sec_rag.retriever import semantic_search, rag_query

# Semantic similarity (no LLM)
semantic_search("capital expenditure guidance AI infrastructure", ticker="NVDA")

# RAG query (requires Ollama)
rag_query("What risks did management highlight?", ticker="MSFT")
```

### Configuration

Edit `sec_rag/config.py`:

```python
EMBEDDING_MODEL = "nomic-embed-text"   # Ollama embedding model
FAISS_PATH      = "./faiss_index"      # Index storage path
CHUNK_SIZE      = 1000                 # Characters per chunk
CHUNK_OVERLAP   = 150                  # Overlap between chunks
LLM             = "granite3.3:8b"      # Ollama LLM for RAG
```

---

## Signal Pipeline

### Normalisation

All raw signals are mapped to `[-1, 1]` by `signals/normalizer.py`:

| Signal source | Raw input | Scale |
|---|---|---|
| `sec_revenue_growth` | YoY growth ratio (e.g. `4.27` = 427%) | `0%`→0, `≥200%`→+1 |
| `sec_gross_margin` | Margin decimal (e.g. `0.784`) | `50%`→0, `≥80%`→+1 |
| `macro_rate` | Fed funds rate % | `0%`→0, `≥7%`→-1 (headwind) |
| `macro_yield_curve` | 10Y-2Y spread % | `-1%`→-1, `+1%`→+1 |
| `news_sentiment` | `[-1, 1]` from sentiment model | pass-through |
| `github_growth` | New repo count (90d) | `0`→0, `≥5000`→+1 |
| `google_trends` | Index 0–100 | `50`→0, `100`→+1 |

### Alignment score

`signals/alignment_scorer.py` combines all normalised signals with default weights into a composite `[-1, 1]` score:

```
STRONG BULL ≥ 0.6  |  BULL ≥ 0.2  |  MIXED ≥ -0.2  |  BEAR ≥ -0.6  |  STRONG BEAR
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-core` | LCEL chains, prompts, runnables |
| `langchain-community` | FAISS vector store |
| `langchain-ollama` | Ollama LLM + embeddings |
| `langchain-text-splitters` | Recursive character splitter |
| `faiss-cpu` | Local vector similarity index |
| `crewai` | Multi-agent orchestration |
| `requests` / `beautifulsoup4` | SEC EDGAR HTTP + HTML parsing |
| `pytrends` | Google Trends scraper |
| `python-dotenv` | `.env` file loader |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Run `make setup` to install deps, or `source .venv/bin/activate` |
| `Connection refused` (port 11434) | Ollama not running — start it: `ollama serve` |
| Red squiggles in VS Code | `Cmd+Shift+P` → **Python: Select Interpreter** → `.venv` |
| `FRED_API_KEY not set` | Add key to `.env` (copy from `.env.example`) |
| `FAISS index not found` | Run `make run-ingest` to build the index first |
| `Failed to fetch SEC filing` | Check `HEADERS` in `sec_rag/ingest.py` — SEC blocks generic User-Agent strings |
| `allow_dangerous_deserialization` | Already handled in `retriever.py` — safe to ignore |
| Ollama model not found | Run `ollama pull granite3.3:8b && ollama pull nomic-embed-text` |

---

## API Rate Limits

- **SEC EDGAR** — 10 req/s max. `ingest.py` sleeps 200ms between requests.
- **GitHub API** — 60 req/h unauthenticated, 5000 req/h with `GITHUB_TOKEN`.
- **NewsData.io** — 200 requests/day on the free tier.
- **FRED** — no hard limit on the free tier; avoid polling in tight loops.
