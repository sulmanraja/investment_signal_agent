# Investment Signal Agent

A seven-agent agentic AI system that generates a **Technology Investment Horizon Report** for engineering leaders. The system aggregates signals from SEC filings, GitHub, news, Google Trends, and macroeconomic data — then produces per-category BUY / HOLD / REDUCE stances using a structured Tree-of-Thoughts reasoning pipeline.

Built as the capstone project for the **CMU Agentic AI Certificate**.

---

## Quick Start

The system supports two modes — choose one:

| Mode | LLM | Embeddings | Entry point |
|---|---|---|---|
| **Local (Ollama)** | `granite3.3:8b` via Ollama | `nomic-embed-text` via Ollama | `make run` |
| **Claude Plugin (MCP)** | Claude API (Haiku / Sonnet) | `all-MiniLM-L6-v2` local | Claude Code |

### NOTE: Make sure to run the run-ingest make target when switching between local or anthrpoic!


```bash
make run-ingest
```

### Local mode

> Assumes Ollama is installed, models are pulled, and `.env` is configured. See [Prerequisites](#prerequisites) and [Installation](#installation) for first-time setup.

```bash
make check-live          # verify Ollama, models, API keys, FAISS index
make run-ai-ml           # run the AI / ML Infrastructure category
make run                 # run all five categories
```

Reports are written to `output/reports/horizon_report_<timestamp>.md`.

### Claude Plugin mode

See [Claude Plugin (MCP Server)](#claude-plugin-mcp-server) for setup.

---

## Claude Plugin (MCP Server)

The system can run as a **Model Context Protocol (MCP) server** inside Claude Code, giving Claude four callable tools without needing to touch the command line.

### How it works

Claude Code auto-detects `.mcp.json` at the project root and starts the server as a subprocess when you open the project. The LLM runs via the **Claude API** (no Ollama required), and embeddings run via **local HuggingFace** models (no Ollama required).

### Prerequisites

| Requirement | Notes |
|---|---|
| **Claude Code** | Desktop app or VS Code extension |
| **Anthropic API key** | [console.anthropic.com](https://console.anthropic.com/) — billed per use |
| **FRED API key** | Same free key used for local mode |
| **NewsData API key** | Same free key used for local mode |

### Setup

**1. Install dependencies** (if you haven't already run `make setup`):

```bash
make setup
```

**2. Add your Anthropic API key to `.env`:**

```ini
# .env  — add this line alongside your existing keys
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Build the FAISS index with HuggingFace embeddings:**

Set `EMBEDDING_PROVIDER=huggingface` in your `.env`, then run:

```bash
make run-ingest
```

`run-ingest` always deletes the existing index before rebuilding, so switching providers is safe. If your index was previously built with Ollama (`nomic-embed-text`, 768 dims) and the provider is now `huggingface` (`all-MiniLM-L6-v2`, 384 dims), this rebuild is required to avoid a dimension mismatch at query time.

**4. Open the project in Claude Code.**

`.mcp.json` is picked up automatically. You will see a tools indicator confirming the `investment-signal-agent` server is connected.

### Available tools

| Tool | Description | Speed |
|---|---|---|
| `list_investment_categories` | Returns the 5 sector IDs, labels, and tickers | Instant |
| `search_sec_filings` | Semantic search over SEC 10-K / 10-Q filings | ~1 s |
| `get_run_history` | Recent analysis results from long-term memory | Instant |
| `analyze_investment_categories` | Full 7-agent pipeline → Markdown report | 5–15 min |

### Example prompts

```
"What investment categories are available?"
"What did NVDA's filings say about AI infrastructure spending?"
"What was the stance on semiconductors in the last analysis?"
"Run a full investment analysis on AI infrastructure and semiconductors."
```

### What `.mcp.json` does

The server is pre-configured to use the Claude API and HuggingFace embeddings:

```json
{
  "mcpServers": {
    "investment-signal-agent": {
      "type": "stdio",
      "command": ".venv/bin/python",
      "args": ["mcp_server.py"],
      "env": {
        "AGENT_LLM_PROVIDER": "anthropic",
        "AGENT_LLM": "claude-haiku-4-5-20251001",
        "EMBEDDING_PROVIDER": "huggingface",
        "KMP_DUPLICATE_LIB_OK": "TRUE"
      }
    }
  }
}
```

`KMP_DUPLICATE_LIB_OK=TRUE` suppresses the macOS OpenMP conflict between `faiss-cpu` and PyTorch (both ship their own `libomp`). It is also exported globally in the Makefile for local runs.

To use a more capable model (slower, higher cost), change `AGENT_LLM` to `claude-sonnet-4-6`.

---

## Architecture

### Seven-Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR AGENT                                 │
│  Receives category list → routes each category through the pipeline below   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ per category (parallel async)
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
  ┌──────────────────────────────┐       ┌─────────────────────┐
  │  DATA COLLECTOR AGENT        │       │  RETRIEVAL AGENT    │
  │  5 sources, all in parallel  │       │  (SEC EDGAR/FAISS)  │
  │                              │       │  · capital_commit.  │
  │  Each source:                │       │  · platform_prior.  │
  │  API fetch →                 │       │  · forward_guidance │
  │  inline CrewAI agent →       │       │  · ≥0.72 threshold  │
  │  narrative + "Score: X/100"  │       │  → sec_score 0-100  │
  │                              │       │  → passages[]       │
  │  ┌──────────────────────┐    │       └──────────┬──────────┘
  │  │ GitHub Trend Analyst  │    │                  │
  │  │ News Sentiment Analyst│    │                  │
  │  │ Search Trend Analyst  │    │                  │
  │  │ Macro Economist       │    │                  │
  │  │ SEC Filing Analyst    │    │                  │
  │  └──────────────────────┘    │                  │
  │                              │                  │
  │  → SubScoreReport:           │                  │
  │    github_score   (0-100)    │                  │
  │    news_score     (0-100)    │                  │
  │    google_trends  (0-100)    │                  │
  │    fred_macro     (0-100)    │                  │
  │    sec_edgar      (0-100)    │                  │
  │    signal_context (narrative)│                  │
  └──────────────┬───────────────┘                  │
                 └──────────────────┬───────────────┘
                                    │
                         Orchestrator merges:
                         signal_context (DC)
                         + SEC passages (RA)
                                    │
                                    ▼
                    ┌────────────────────────┐
                    │   SIGNAL ANALYST       │
                    │  alignment_spread =    │
                    │  max(scores)-min(scores│
                    │  ALIGNED   ≤ 20 pts    │
                    │  CONTRADICTORY > 20 pts│
                    │  or hard contradiction │
                    └──────────┬─────────────┘
                               │
              ┌────────────────┴──────────────────┐
              │ ALIGNED                           │ CONTRADICTORY
              ▼                                   ▼
  ┌─────────────────────────┐     ┌───────────────────────────────────┐
  │  Weighted Score         │     │  THOUGHT GENERATOR AGENT          │
  │  sec_edgar   30%        │     │  Receives: signal_context         │
  │  github      20%        │     │  (no Critic scores — isolation)   │
  │  news        20%        │     │  4 canonical branches:            │
  │  trends      15%        │     │   A — Capital-Led                 │
  │  fred_macro  15%        │     │   B — Adoption-Led                │
  │                         │     │   C — Risk-Adjusted               │
  │  ≥65 → BUY              │     │   D — Evidence-Insufficient       │
  │  ≥40 → HOLD             │     │  + counter_evidence per branch    │
  │  <40 → REDUCE           │     └────────────────┬──────────────────┘
  └────────────┬────────────┘                      │
               │                   Orchestrator: counter_evidence gate
               │                   trivial → __TRIVIAL__ sentinel
               │                                   ▼
               │                  ┌───────────────────────────────────┐
               │                  │  CRITIC AGENT  (per-node, indep.) │
               │                  │  Receives: signal_context         │
               │                  │  (no sibling scores — isolation)  │
               │                  │  5 criteria × 0-20 = 100 max      │
               │                  │  1. Evidence Alignment             │
               │                  │     (zeroed if __TRIVIAL__)        │
               │                  │  2. Internal Consistency           │
               │                  │  3. Macro Compatibility            │
               │                  │  4. Actionability                  │
               │                  │  5. Confidence Calibration         │
               │                  └────────────────┬──────────────────┘
               │                                   │ beam prune
               │                                   │ floor=40, width=2
               │                                   │
               │                  ┌────────────────▼──────────────────┐
               │                  │  Termination check:               │
               │                  │  · clear winner ≥85 + 15pt margin │
               │                  │  · single survivor                 │
               │                  │  · max depth (2)                  │
               │                  │  · Branch D auto-promote           │
               │                  └────────────────┬──────────────────┘
               │                                   │
               └─────────────────────┬─────────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │      SYNTHESIS AGENT           │
                    │  Receives: SynthesisPackage    │
                    │  (no tool calls — pure context)│
                    │  Output guardrails applied:    │
                    │  · Confidence-stance check     │
                    │  · Citation groundedness audit │
                    │  · Scope disclaimer prepended  │
                    │  Writes Technology Investment  │
                    │  Horizon Report (Markdown)     │
                    │  output/reports/horizon_*.md   │
                    └────────────────────────────────┘
```

### Five Technology Categories

| ID | Label | Representative Tickers |
|---|---|---|
| `ai_ml_infrastructure` | AI / ML Infrastructure | NVDA, AMD, INTC, SMCI |
| `cloud_edge` | Cloud & Edge Computing | AMZN, MSFT, GOOGL, SNOW |
| `semiconductors` | Semiconductors | TSM, ASML, AMAT, KLAC |
| `developer_tooling` | Developer Tooling & Platforms | MSFT, ORCL, MDB, DDOG |
| `cybersecurity` | Cybersecurity | CRWD, PANW, ZS, OKTA |

---

## Project Structure

```
investment_signal_agent/
├── agents/
│   ├── orchestrator_agent.py       # Master coordinator (task routing, ToT loop, escalation)
│   ├── data_collector_agent.py     # 5-source collector: API fetch + inline CrewAI scoring
│   │                               #   Each source has its own Agent+Task+Crew inline
│   │                               #   → SubScoreReport (scores + signal_context narrative)
│   ├── retrieval_agent.py          # SEC EDGAR semantic search (FAISS, ≥0.72 threshold)
│   │                               #   → RetrievalReport (sec_score + passages[])
│   ├── signal_analyst_agent.py     # ALIGNED / CONTRADICTORY classifier (pure-math, no LLM)
│   ├── thought_generator_agent.py  # 4 canonical ToT branches + counter_evidence (CrewAI)
│   ├── critic_agent.py             # 5-criterion per-node scoring 0-100 (CrewAI)
│   └── synthesis_agent.py          # Isolated report writer, output guardrails (CrewAI)
├── schemas/
│   └── messages.py                 # Pydantic inter-agent message contracts
├── guardrails/
│   ├── data_integrity.py           # Source attribution, staleness flags, availability check
│   ├── escalation.py               # Three-tier escalation model (Level 1/2/3)
│   └── output_audit.py             # Confidence-stance check, citation audit, disclaimer
├── evaluation/
│   └── metrics.py                  # RunMetrics dataclass + MetricsStore (JSON-lines)
├── tot/
│   ├── thought_node.py             # ThoughtNode dataclass (branch A/B/C/D)
│   ├── branches.py                 # Generates 4 canonical stances (LLM)
│   ├── evaluator.py                # 5-criterion rubric scoring (LLM)
│   ├── pruner.py                   # Hard floor=40, beam=2, Branch D auto-promote
│   └── beam_search.py              # Beam search with 4 termination conditions
├── tools/                          # Data source integrations
│   ├── edgar_ingest.py             # Wraps sec_rag/ingest.py
│   ├── edgar_retriever.py          # Wraps sec_rag/retriever.py (called by DataCollector)
│   ├── github_tool.py              # GitHub API trending repos
│   ├── google_trends_tool.py       # Google Trends via pytrends
│   ├── newsdata_tool.py            # NewsData.io API
│   └── fred_macro_tool.py          # FRED economic indicators
├── signals/
│   ├── normalizer.py               # normalize_to_100() + legacy normalize_signal()
│   ├── alignment_scorer.py         # classify_alignment() + legacy score_alignment()
│   ├── category_registry.py        # Registry loader (weights, category IDs, signal sources)
│   └── category_registry.json      # 5 tech categories with tickers/query_terms
├── sec_rag/                        # SEC EDGAR RAG subsystem
│   ├── ingest.py                   # Fetch 10-K/10-Q → FAISS index (requires Ollama)
│   ├── retriever.py                # Semantic search + RAG queries
│   ├── config.py                   # Embedding model, paths, chunk size
│   └── faiss_index/                # Auto-created after ingestion (git-ignored)
├── memory/
│   ├── long_term.py                # JSON-backed run persistence (used by Orchestrator)
│   ├── metrics_store.json          # Auto-created on first run (git-ignored)
│   └── run_store.json              # Auto-created on first run (git-ignored)
├── output/
│   └── reports/                    # Generated Markdown reports (git-ignored)
├── faiss_index/                    # Root-level FAISS index (used by retrieval_agent)
│   ├── index.faiss
│   └── index.pkl
├── tests/
│   ├── test_signals.py
│   ├── test_tot.py
│   └── test_tools_noapi.py
├── utils/
│   └── llm_factory.py              # make_llm(temp) — routes to Ollama or Anthropic via AGENT_LLM_PROVIDER
├── main.py                         # Entry point → OrchestratorAgent (local mode)
├── mcp_server.py                   # FastMCP server — exposes 4 tools to Claude Code
├── requirements.txt
├── Makefile
├── .mcp.json                       # Claude Code project registration (auto-detected)
├── .env                            # Local secrets (git-ignored)
└── .env.example                    # Template — copy to .env and fill in keys
```

---

## Prerequisites

### Both modes

| Requirement | Version / Notes |
|---|---|
| **Python** | 3.13 required (`crewai` blocks Python ≥ 3.14) |
| **FRED API key** | Free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **NewsData API key** | Free tier at [newsdata.io](https://newsdata.io/) |
| **GitHub token** | Optional — raises rate limits from 60 to 5 000 req/h |

### Local mode only

| Requirement | Version / Notes |
|---|---|
| **Ollama** | Local LLM runtime — `ollama serve` must be running |
| **granite3.3:8b** | Default LLM (~5 GB) — `ollama pull granite3.3:8b` |
| **nomic-embed-text** | Embedding model (~270 MB) — `ollama pull nomic-embed-text` |

### Claude Plugin (MCP) mode only

| Requirement | Version / Notes |
|---|---|
| **Claude Code** | Desktop app or VS Code extension |
| **Anthropic API key** | [console.anthropic.com](https://console.anthropic.com/) — billed per use |

---

## Installation

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd investment_signal_agent
```

### 2. Create the virtual environment

```bash
make setup
```

> `make setup` uses Python 3.13 specifically (required by `crewai`). If `python3.13` is not in your PATH it falls back to `/opt/homebrew/anaconda3/bin/python3`.

### 3. Pull Ollama models

```bash
ollama pull granite3.3:8b
ollama pull nomic-embed-text
ollama list   # verify both appear
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

> For MCP / Anthropic mode, a pre-filled reference file is included:
> `.env.example.anthropic`. Copy it instead to start with the Anthropic
> and HuggingFace settings already set:
> ```bash
> cp .env.example.anthropic .env
> # Add your ANTHROPIC_API_KEY, FRED_API_KEY, and NEWSDATA_API_KEY
> ```

**Local (Ollama) mode** — minimum required keys:

```ini
# .env
AGENT_LLM_PROVIDER=ollama
AGENT_LLM=granite3.3:8b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
TOT_MAX_DEPTH=2
TOT_BEAM_WIDTH=2

FRED_API_KEY=your_key_here
NEWSDATA_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here    # optional
```

**Claude Plugin (MCP) mode** — add these instead of the Ollama settings:

```ini
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM=claude-haiku-4-5-20251001
EMBEDDING_PROVIDER=huggingface
ANTHROPIC_API_KEY=sk-ant-...
```

> `EMBEDDING_PROVIDER` controls which model is used for the FAISS index. Switching between `ollama` and `huggingface` requires rebuilding the index — `make run-ingest` handles this automatically by deleting the old index first.

### 5. Build the SEC EDGAR FAISS index

```bash
make run-ingest
```

Fetches 10-K/10-Q filings for the default tickers (`NVDA MSFT AMZN GOOGL TSM ASML CRWD PANW`), chunks the MD&A sections, and writes the FAISS index to `faiss_index/`. Always deletes any existing index before building to prevent dimension mismatches when switching embedding providers. Only needs to be run once (or when refreshing filings or changing `EMBEDDING_PROVIDER`).

---

## Verifying Your Setup

### Verify Python imports only

```bash
make check
```

### Verify everything — Ollama, models, API keys, and FAISS index

```bash
make check-live
```

Expected output when everything is ready:

```
=== Environment check ===
Python 3.13.x
  langchain       ok
  langchain_ollama ok
  faiss           ok
  crewai          ok
  pydantic        ok
  requests        ok
=== Ollama check ===
  Ollama:           running
  granite3.3:8b:    found
  nomic-embed-text: found
=== API key check ===
  FRED_API_KEY:     set
  NEWSDATA_API_KEY: set
  GITHUB_TOKEN:     set
  Keys check complete
=== FAISS index check ===
  faiss_index/:     found

All live checks passed. Ready to run the full pipeline.
```

---

## Running the Pipeline

Always run `make check-live` first to confirm all services are up.

### Run all five technology categories

```bash
make run
# equivalent to: python main.py
```

### Run a single category

```bash
make run-ai-ml
make run-cloud
make run-semiconductors
make run-devtools
make run-cybersecurity
```

### Run a custom subset

```bash
make run-category CATEGORIES="ai_ml_infrastructure semiconductors"
# equivalent to: python main.py ai_ml_infrastructure semiconductors
```

Valid category IDs: `ai_ml_infrastructure`, `cloud_edge`, `semiconductors`, `developer_tooling`, `cybersecurity`.

The report is written to `output/reports/horizon_report_<timestamp>.md`.

---

## Testing Without Ollama or API Keys

These targets run immediately with no external dependencies:

```bash
make test                # all three test suites
make run-signal-analyst  # pure-math ALIGNED/CONTRADICTORY classifier
make run-retrieval-agent # FAISS semantic search (needs FAISS index, no LLM)
make run-data-integrity  # source attribution + staleness checks
make run-escalation      # three-tier escalation evaluator
make run-output-audit    # citation audit + confidence stance check
make run-metrics         # metrics store read/write
```

---

## Individual Component Targets

Each module has a `__main__` smoke test accessible via `make`:

| Make target | Module | Requires |
|---|---|---|
| `make test-signals` | `tests/test_signals.py` | nothing |
| `make test-tot` | `tests/test_tot.py` | nothing |
| `make test-tools` | `tests/test_tools_noapi.py` | nothing |
| `make run-signal-analyst` | `agents/signal_analyst_agent.py` | nothing |
| `make run-retrieval-agent` | `agents/retrieval_agent.py` | FAISS index |
| `make run-data-integrity` | `guardrails/data_integrity.py` | nothing |
| `make run-escalation` | `guardrails/escalation.py` | nothing |
| `make run-output-audit` | `guardrails/output_audit.py` | nothing |
| `make run-metrics` | `evaluation/metrics.py` | nothing |
| `make run-github` | `tools/github_tool.py` | network |
| `make run-trends` | `tools/google_trends_tool.py` | network |
| `make run-news` | `tools/newsdata_tool.py` | `NEWSDATA_API_KEY` |
| `make run-macro` | `tools/fred_macro_tool.py` | `FRED_API_KEY` |
| `make run-edgar-tool` | `tools/edgar_retriever.py` | FAISS index |
| `make run-ingest` | `sec_rag/ingest.py` | Ollama **or** HuggingFace (set `EMBEDDING_PROVIDER` in `.env`) |
| `make run-retriever` | `sec_rag/retriever.py` | FAISS index (no LLM needed for search) |
| `make run-data-collector` | `agents/data_collector_agent.py` | LLM + API keys |
| `make run-thought-generator` | `agents/thought_generator_agent.py` | LLM |
| `make run-critic` | `agents/critic_agent.py` | LLM |
| `make run-synthesis` | `agents/synthesis_agent.py` | LLM |
| `make run-branches` | `tot/branches.py` | LLM |
| `make run-evaluator` | `tot/evaluator.py` | LLM |
| `make run-beam-search` | `tot/beam_search.py` | LLM |

---

## Signal Pipeline Details

### Sub-score collection (0-100 scale)

The `DataCollectorAgent` collects all five sources in parallel using `asyncio.gather`. For each source it: (1) fetches data via the API tool, (2) passes the raw data to an inline **CrewAI agent** (`Agent + Task + Crew`) that writes a narrative analysis ending with `Score: X/100`, (3) parses that integer as the sub-score. The math formula is the fallback used only when the CrewAI call fails.

| Dimension | API Tool | CrewAI Role | Weight |
|---|---|---|---|
| `github` | `github_tool.py` — trending repos | GitHub Trend Analyst | 20% |
| `news` | `newsdata_tool.py` — recent headlines | News Sentiment Analyst | 20% |
| `google_trends` | `google_trends_tool.py` — keyword interest | Search Trend Analyst | 15% |
| `fred_macro` | `fred_macro_tool.py` — FRED indicators | Macro Economist | 15% |
| `sec_edgar` *(DC)* | `tools/edgar_retriever.py` — FAISS docs | SEC Filing Analyst | — |
| `sec_edgar` *(RA)* | `retrieval_agent.py` — FAISS + ≥0.72 | Deterministic: avg_sim × 100 + coverage bonus | 30% |

> **Two SEC scores**: The `DataCollectorAgent` produces a lightweight `sec_edgar_score` via its inline CrewAI SEC Filing Analyst. The `RetrievalAgent` produces the authoritative `sec_score` via structured 3-query FAISS retrieval with similarity threshold and keyword confirmation. The Orchestrator uses the `RetrievalAgent` score for the `AlignmentRequest`; the DataCollector score enriches `signal_context`.

### signal_context — the narrative context string

After all five sources are scored, the `DataCollectorAgent` concatenates the per-source CrewAI narratives into a single `signal_context` string. This becomes the primary context input for the **Thought Generator** and **Critic** agents.

```
=== AI / ML Infrastructure — Signal Summary (2026-06-24) ===

GitHub (score=74/100):
Strong developer momentum with CUDA and inference frameworks dominating.
3,400+ new CUDA-related repos in 90 days indicates expanding ecosystem engagement.
Score: 74/100

News (score=58/100):
Neutral-to-positive sentiment. H100 supply constraints offset strong demand signals.
Export controls to China create headline risk but have been partially priced in.
Score: 58/100

Google Trends (score=81/100):
AI infrastructure and GPU compute search interest near peak levels.
Score: 81/100

FRED Macro (score=42/100):
Fed funds at 5.25% creates multiple compression headwind for growth assets.
Score: 42/100

SEC EDGAR (score=68/100):
NVDA 10-K highlights data-center as primary growth driver with $27B capex guidance.
Score: 68/100
```

### ALIGNED vs CONTRADICTORY routing

```
alignment_spread = max(sub_scores) − min(sub_scores)

ALIGNED:        spread ≤ 20 pts AND no hard contradiction
CONTRADICTORY:  spread > 20 pts OR hard contradiction

Hard contradictions:
  Pattern 1: sec_edgar ≥ 70 AND news ≤ 30
  Pattern 2: github ≥ 70 AND google_trends ≥ 70 AND fred_macro ≤ 30
```

**ALIGNED** → deterministic weighted score → BUY (≥65) / HOLD (≥40) / REDUCE (<40)

**CONTRADICTORY** → Tree-of-Thoughts beam search

### Tree-of-Thoughts beam search

```
Depth 0: Generate 4 canonical branches (A/B/C/D) with counter_evidence
         → Critic scores each (5 × 0-20 = 100 max)
         → Prune: hard floor = 40, beam width = 2

Termination (checked after each depth):
  · clear_winner:     top score ≥ 85 AND ≥ 15pt margin over next
  · single_survivor:  only one node above floor
  · branch_d_auto:    zero survivors → auto-promote Branch D
  · max_depth:        depth 2 reached

Depth 1 (if no early termination):
  Expand surviving branches → Critic → Prune → Select winner
```

### Counter-evidence gate

If `counter_evidence` is trivial (empty, "N/A", "none", < 20 chars), the Orchestrator marks it `__TRIVIAL__` before passing to the Critic. The Critic then forces **Evidence Alignment = 0** for that node.

---

## Safety, Reliability & Human-in-the-Loop

### Data Integrity Layer

Every source call in `DataCollectorAgent` produces a `SourceAttribution` record (tool name, query, record count, success/failure, timestamp). After collection, staleness checks run on news articles (48h threshold) and SEC filings (90-day threshold). Failed sources and stale flags are carried through to the escalation evaluator.

### Three-Tier Escalation Model

| Level | Trigger | Effect |
|---|---|---|
| **Level 1 — In-Brief Flag** | Branch D resolution, stale sub-score, LOW confidence stance | Flag appended to report; no delivery delay |
| **Level 2 — Mandatory Hold** | Source availability < 70%, ≥2 Critic variance re-evaluations | Report held; reviewer approval required |
| **Level 3 — Pipeline Halt** | Citation audit failure, ≥2 data sources fully unavailable | Pipeline stops; report not delivered |

### Output-Time Guardrails

Applied by `SynthesisAgent` after generating the report draft:

1. **Confidence-stance consistency** — assigns `ConfidenceLevel` (HIGH/MEDIUM/LOW) per category; LOW confidence on actionable stance triggers Level 1.
2. **Citation groundedness audit** — identifies claims not verifiable from source evidence; failure appends an unsupported-claims annex and triggers Level 3.
3. **Scope disclaimer** — every report opens with a disclaimer that it is decision-support, not investment advice.

### Evaluation Metrics

`RunMetrics` are recorded per run to `memory/metrics_store.json` (JSON-lines):

- **ToT quality**: Branch D promotion rate, counter-evidence completeness, Critic score variance, re-evaluation trigger count
- **Operational**: Source availability, staleness flag count, end-to-end latency
- **Escalation**: Level and triggers
- **Feedback-derived** (populated retroactively): forecast accuracy, confidence calibration

---

## SEC RAG: Building the FAISS Index

The embedding provider is selected by `EMBEDDING_PROVIDER` in `.env`:

| Value | Model | Dims | Requires |
|---|---|---|---|
| `ollama` (default) | `nomic-embed-text` | 768 | Ollama running locally |
| `huggingface` | `all-MiniLM-L6-v2` | 384 | Nothing — runs fully locally |

**The two models are incompatible.** Switching `EMBEDDING_PROVIDER` requires rebuilding the index.

### Ingest filings

```bash
make run-ingest
```

Default tickers: `NVDA MSFT AMZN GOOGL TSM ASML CRWD PANW`. Each filing's MD&A section is chunked and indexed into `faiss_index/`. The old index is always deleted first to prevent dimension mismatches.

To ingest custom tickers:

```python
from sec_rag.ingest import ingest
ingest(["NVDA", "AMD"], form_types=["10-K"], max_filings_per_ticker=3)
```

### Query the index directly

```bash
make run-retriever
```

Or in Python:

```python
from sec_rag.retriever import semantic_search, rag_query

semantic_search("capital expenditure AI infrastructure", ticker="NVDA")
rag_query("What risks did management highlight?", ticker="MSFT")
```

### Three SEC query types (RetrievalAgent)

| Query type | Keywords confirmed |
|---|---|
| `capital_commitment` | capex, investment, billion, spend, infrastructure, build |
| `platform_prioritization` | platform, priority, strategy, focus, bet, invest |
| `forward_guidance` | guidance, expect, outlook, forecast, target, quarter |

SEC score = avg_similarity × 100 + coverage bonus (5 pts per additional query type covered).

---

## Inter-Agent Message Contracts

All agents communicate via typed Pydantic models in `schemas/messages.py`:

| Schema | Direction | Key fields |
|---|---|---|
| `TaskAssignment` | Orchestrator → DC + RA | `run_id`, `category_id`, `tickers`, `query_terms` |
| `SubScoreReport` | DC → Orchestrator | `github_score`, `news_score`, `google_trends_score`, `fred_macro_score`, `sec_edgar_score`, `signal_context`, `source_attributions`, `staleness_flags`, `failed_sources` |
| `RetrievalReport` | RA → Orchestrator | `sec_score`, `passages[]`, `passages_below_threshold` |
| `AlignmentRequest` | Orchestrator → Signal Analyst | `sub_scores` (all 5 keys) |
| `AlignmentVerdict` | Signal Analyst → Orchestrator | `classification`, `alignment_spread`, `hard_contradiction` |
| `GenerationRequest` | Orchestrator → Thought Generator | `context` (= `signal_context` + SEC passages), `depth`, `survivor_branches` |
| `ThoughtNodesReport` | Thought Generator → Orchestrator | `nodes[]` (branch, content, counter_evidence) |
| `EvaluationRequest` | Orchestrator → Critic | single `node`, `context` (no sibling scores) |
| `NodeScoreReport` | Critic → Orchestrator | 5 criterion scores, `total`, `key_weakness`, `counter_evidence_zeroed` |
| `ScoreReport` | Orchestrator (internal) | `node_scores[]`, `variance_triggered` |
| `CategoryResult` | Orchestrator (internal) | `stance`, `final_score`, `winning_branch`, `pruned_nodes[]`, `confidence_level` |
| `SynthesisPackage` | Orchestrator → Synthesis | all `category_results[]`, `emerging_candidates`, `reeval_trigger_count`, `escalation` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-core` | Prompts, chains, runnables |
| `langchain-community` | FAISS vector store + HuggingFace embeddings |
| `langchain-ollama` | Ollama LLM + embeddings (local mode) |
| `langchain-text-splitters` | Recursive character splitter |
| `faiss-cpu` | Local vector similarity index |
| `crewai ≥ 1.0.0` | Multi-agent roles — DataCollector (5 inline Crews), Thought Generator, Critic, Synthesis |
| `pydantic ≥ 2.0` | Inter-agent message schemas |
| `requests` / `beautifulsoup4` | SEC EDGAR HTTP + HTML parsing |
| `pytrends` | Google Trends scraper |
| `python-dotenv` | `.env` file loader |
| `mcp[cli] ≥ 1.0.0` | FastMCP server for Claude Code plugin |
| `anthropic ≥ 0.40.0` | Claude API client (MCP mode) |
| `sentence-transformers ≥ 3.0.0` | `all-MiniLM-L6-v2` embeddings (MCP mode, no Ollama needed) |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | `make setup` or `source .venv/bin/activate` |
| `Connection refused` (port 11434) | Ollama not running — `ollama serve` |
| `FAISS index not found` | `make run-ingest` to build the SEC index |
| FAISS dimension mismatch | `EMBEDDING_PROVIDER` changed — run `make run-ingest` to rebuild the index |
| `FRED_API_KEY not set` | Add key to `.env` |
| `Failed to fetch SEC filing` | SEC blocks generic User-Agent — check `HEADERS` in `sec_rag/ingest.py` |
| Red squiggles in VS Code | `Cmd+Shift+P` → **Python: Select Interpreter** → `.venv` |
| Ollama model not found | `ollama pull granite3.3:8b && ollama pull nomic-embed-text` |
| `crewai` install fails | Ensure Python 3.13 is used (`make setup` handles this); `crewai` blocks Python ≥ 3.14 |
| `pip install` fails on `crewai` | Run `make setup` rather than plain `pip install -r requirements.txt` — the Makefile selects Python 3.13 explicitly |
| `OMP: Error #15` (libomp conflict) | `faiss-cpu` and PyTorch both ship `libomp` on macOS. `KMP_DUPLICATE_LIB_OK=TRUE` is exported by the Makefile (local runs) and set in `.mcp.json` (MCP runs) — no manual action needed |
| MCP server not connecting in Claude Code | Reload the VS Code window (`Cmd+Shift+P` → **Reload Window**) after editing `.mcp.json` or `.env` |

---

## API Rate Limits

| Source | Limit |
|---|---|
| **SEC EDGAR** | 10 req/s max; `ingest.py` sleeps 200 ms between requests |
| **GitHub API** | 60 req/h unauthenticated, 5 000 req/h with `GITHUB_TOKEN` |
| **NewsData.io** | 200 requests/day on the free tier |
| **FRED** | No hard limit on the free tier |
| **Google Trends** | No official limit; `pytrends` may be throttled at high frequency |
