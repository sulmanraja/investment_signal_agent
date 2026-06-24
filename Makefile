VENV_DIR := .venv
PYTHON   := $(VENV_DIR)/bin/python
PIP      := $(VENV_DIR)/bin/pip
ROOT     := $(shell pwd)
PYTHONPATH := $(ROOT)

# crewai requires Python <3.14 — prefer 3.13 from Anaconda if available
PYTHON3_13 := $(shell command -v python3.13 2>/dev/null || echo /opt/homebrew/anaconda3/bin/python3)

export PYTHONPATH

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	@echo "Creating virtual environment in $(VENV_DIR)/ using $(PYTHON3_13) ..."
	$(PYTHON3_13) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete."
	@echo "  Activate manually:  source $(VENV_DIR)/bin/activate"
	@echo "  VS Code picks up the interpreter automatically."

check:
	@echo "=== Environment check ==="
	@$(PYTHON) --version
	@$(PYTHON) -c "import langchain;       print('  langchain       ok')"
	@$(PYTHON) -c "import langchain_ollama; print('  langchain_ollama ok')"
	@$(PYTHON) -c "import faiss;           print('  faiss           ok')"
	@$(PYTHON) -c "import crewai;          print('  crewai          ok')"
	@$(PYTHON) -c "import pydantic;        print('  pydantic        ok')"
	@$(PYTHON) -c "import requests;        print('  requests        ok')"

# ── Live readiness checks ─────────────────────────────────────────────────────
#   Verify Ollama, models, API keys, and FAISS index are all present
#   before attempting any LLM or API-dependent target.

check-ollama:
	@echo "=== Ollama check ==="
	@curl -sf http://localhost:11434/api/tags > /dev/null \
	    && echo "  Ollama:           running" \
	    || (echo "  ERROR: Ollama not running — start it with: ollama serve" && exit 1)
	@curl -sf http://localhost:11434/api/tags \
	    | $(PYTHON) -c "import sys,json; tags=json.load(sys.stdin)['models']; \
	      names=[m['name'] for m in tags]; \
	      llm=__import__('os').getenv('AGENT_LLM','granite3.3:8b'); \
	      emb=__import__('os').getenv('EMBEDDING_MODEL','nomic-embed-text'); \
	      ok=True; \
	      [print('  '+n+':  found') or True for n in names if n.split(':')[0] in (llm.split(':')[0], emb)]; \
	      missing=[m for m in [llm,emb] if not any(t.startswith(m.split(':')[0]) for t in names)]; \
	      [print('  ERROR: missing model — run: ollama pull '+m) for m in missing]; \
	      exit(1) if missing else exit(0)" \
	    || exit 1

check-keys:
	@echo "=== API key check ==="
	@$(PYTHON) -c "\
import os; from dotenv import load_dotenv; load_dotenv(); \
keys = {'FRED_API_KEY': os.getenv('FRED_API_KEY'), \
        'NEWSDATA_API_KEY': os.getenv('NEWSDATA_API_KEY'), \
        'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN')}; \
ok = True; \
[(print('  '+k+':  set'), None) if v and not v.startswith('your_') \
 else (print('  WARNING: '+k+' not configured (optional for GitHub, required for others)'), None) \
 for k,v in keys.items()]; \
print('  Keys check complete')"

check-faiss:
	@echo "=== FAISS index check ==="
	@test -f faiss_index/index.faiss \
	    && echo "  faiss_index/:     found" \
	    || (echo "  WARNING: FAISS index missing — run: make run-ingest" && exit 1)

check-live: check check-ollama check-keys check-faiss
	@echo ""
	@echo "All live checks passed. Ready to run the full pipeline."
	@echo "  make run           — all five categories"
	@echo "  make run-ai-ml     — AI / ML Infrastructure only"

# ── No-dependency tests ────────────────────────────────────────────────────────

test-signals:
	$(PYTHON) tests/test_signals.py

test-tot:
	$(PYTHON) tests/test_tot.py

test-tools:
	$(PYTHON) tests/test_tools_noapi.py

test:
	@echo ""
	@echo "=== Running all no-dependency tests ==="
	@echo ""
	$(MAKE) test-signals
	$(MAKE) test-tot
	$(MAKE) test-tools
	@echo ""
	@echo "All tests complete."

# ── SEC RAG ───────────────────────────────────────────────────────────────────
#   run-ingest requires Ollama (nomic-embed-text for embeddings)
#   run-retriever requires Ollama + existing FAISS index

run-ingest:
	$(PYTHON) -m sec_rag.ingest

run-retriever:
	$(PYTHON) -m sec_rag.retriever

# ── Tools (require API keys unless noted) ────────────────────────────────────

run-github:
	$(PYTHON) -m tools.github_tool

run-trends:
	$(PYTHON) -m tools.google_trends_tool

run-news:
	$(PYTHON) -m tools.newsdata_tool

run-macro:
	$(PYTHON) -m tools.fred_macro_tool

run-edgar-tool:
	$(PYTHON) -m tools.edgar_retriever

# ── Agents ────────────────────────────────────────────────────────────────────

run-signal-analyst:
	$(PYTHON) -m agents.signal_analyst_agent

run-retrieval-agent:
	$(PYTHON) -m agents.retrieval_agent

run-data-collector:
	$(PYTHON) -m agents.data_collector_agent

run-thought-generator:
	$(PYTHON) -m agents.thought_generator_agent

run-critic:
	$(PYTHON) -m agents.critic_agent

run-synthesis:
	$(PYTHON) -m agents.synthesis_agent

# ── ToT modules (require Ollama) ──────────────────────────────────────────────

run-branches:
	$(PYTHON) -m tot.branches

run-evaluator:
	$(PYTHON) -m tot.evaluator

run-beam-search:
	$(PYTHON) -m tot.beam_search

# ── Guardrails & Evaluation ───────────────────────────────────────────────────

run-data-integrity:
	$(PYTHON) -m guardrails.data_integrity

run-escalation:
	$(PYTHON) -m guardrails.escalation

run-output-audit:
	$(PYTHON) -m guardrails.output_audit

run-metrics:
	$(PYTHON) -m evaluation.metrics

# ── Full pipeline ─────────────────────────────────────────────────────────────
#   All five categories (requires Ollama + API keys + FAISS index)
run:
	$(PYTHON) main.py

#   Single or subset of categories:
#     make run-category CATEGORIES="ai_ml_infrastructure semiconductors"
run-category:
	$(PYTHON) main.py $(CATEGORIES)

#   Convenience — one category each
run-ai-ml:
	$(PYTHON) main.py ai_ml_infrastructure

run-cloud:
	$(PYTHON) main.py cloud_edge

run-semiconductors:
	$(PYTHON) main.py semiconductors

run-devtools:
	$(PYTHON) main.py developer_tooling

run-cybersecurity:
	$(PYTHON) main.py cybersecurity

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Investment Signal Agent — Makefile targets"
	@echo ""
	@echo "Setup"
	@echo "  make setup                  Create venv (Python 3.13) and install all deps"
	@echo "  make check                  Verify Python imports (no LLM / no network)"
	@echo ""
	@echo "Live readiness checks"
	@echo "  make check-ollama           Verify Ollama is running and required models exist"
	@echo "  make check-keys             Verify API keys are configured in .env"
	@echo "  make check-faiss            Verify FAISS index has been built"
	@echo "  make check-live             Run all three checks above (+ make check)"
	@echo ""
	@echo "Tests  (no LLM / no API keys required)"
	@echo "  make test                   Run all no-dependency tests"
	@echo "  make test-signals           Test signals/ modules"
	@echo "  make test-tot               Test tot/ modules (ThoughtNode, pruner)"
	@echo "  make test-tools             Test tool stubs (missing-key guards)"
	@echo ""
	@echo "Guardrails & Evaluation  (no LLM / no API keys required)"
	@echo "  make run-data-integrity     Source attribution + staleness checks"
	@echo "  make run-escalation         Three-tier escalation evaluator"
	@echo "  make run-output-audit       Citation audit + confidence stance check"
	@echo "  make run-metrics            Metrics store read/write"
	@echo ""
	@echo "Agents — no LLM required"
	@echo "  make run-signal-analyst     SignalAnalystAgent (pure-math, no LLM)"
	@echo "  make run-retrieval-agent    RetrievalAgent (FAISS only, no LLM)"
	@echo ""
	@echo "Tools  (network; API key where noted)"
	@echo "  make run-github             GitHub trending repos       (no key needed)"
	@echo "  make run-trends             Google Trends interest      (no key needed)"
	@echo "  make run-news               NewsData.io headlines       [NEWSDATA_API_KEY]"
	@echo "  make run-macro              FRED macro indicators       [FRED_API_KEY]"
	@echo "  make run-edgar-tool         SEC EDGAR FAISS retriever  (needs FAISS index)"
	@echo ""
	@echo "SEC RAG  (requires Ollama + nomic-embed-text)"
	@echo "  make run-ingest             Fetch SEC filings → build FAISS index"
	@echo "  make run-retriever          Test semantic search against the index"
	@echo ""
	@echo "Agents — require Ollama"
	@echo "  make run-data-collector     DataCollectorAgent (5 sources, CrewAI)  [API keys]"
	@echo "  make run-thought-generator  ThoughtGeneratorAgent (4 ToT branches)"
	@echo "  make run-critic             CriticAgent (5-criterion scoring)"
	@echo "  make run-synthesis          SynthesisAgent (report writer)"
	@echo ""
	@echo "ToT modules  (require Ollama)"
	@echo "  make run-branches           Generate canonical branches"
	@echo "  make run-evaluator          Score a node (5-criterion rubric)"
	@echo "  make run-beam-search        Run full beam search"
	@echo ""
	@echo "Full pipeline  (requires Ollama + API keys + FAISS index)"
	@echo "  make run-ai-ml              AI / ML Infrastructure only"
	@echo "  make run-cloud              Cloud & Edge Computing only"
	@echo "  make run-semiconductors     Semiconductors only"
	@echo "  make run-devtools           Developer Tooling & Platforms only"
	@echo "  make run-cybersecurity      Cybersecurity only"
	@echo "  make run-category CATEGORIES='ai_ml_infrastructure cloud_edge'"
	@echo "  make run                    All five categories"
	@echo ""

.PHONY: setup check \
        check-ollama check-keys check-faiss check-live \
        test test-signals test-tot test-tools \
        run-ingest run-retriever \
        run-github run-trends run-news run-macro run-edgar-tool \
        run-signal-analyst run-retrieval-agent run-data-collector \
        run-thought-generator run-critic run-synthesis \
        run-branches run-evaluator run-beam-search \
        run-data-integrity run-escalation run-output-audit run-metrics \
        run run-category run-ai-ml run-cloud run-semiconductors \
        run-devtools run-cybersecurity help
