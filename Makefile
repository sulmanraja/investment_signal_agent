VENV_DIR := .venv
PYTHON   := $(VENV_DIR)/bin/python
PIP      := $(VENV_DIR)/bin/pip
ROOT     := $(shell pwd)
PYTHONPATH := $(ROOT)

export PYTHONPATH

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "Creating virtual environment in $(VENV_DIR)/ ..."
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete."
	@echo "   Activate manually:  source $(VENV_DIR)/bin/activate"
	@echo "   VSCode will pick up the interpreter automatically."

# Install only what sec_rag needs (faster iteration)
setup-rag:
	@echo "Installing sec_rag dependencies only ..."
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r sec_rag/requirements.txt
	@echo "✅ sec_rag setup complete."

# Verify the venv is set up and key imports work
check:
	@echo "=== Environment check ==="
	@$(PYTHON) --version
	@$(PYTHON) -c "import langchain; print('  langchain', langchain.__version__)"
	@$(PYTHON) -c "import langchain_ollama; print('  langchain_ollama ok')"
	@$(PYTHON) -c "import faiss; print('  faiss ok')"
	@$(PYTHON) -c "import requests; print('  requests ok')"
	@$(PYTHON) -c "import bs4; print('  beautifulsoup4 ok')"

# ── No-dependency tests (no LLM, no API keys) ─────────────────────────────────
test-signals:
	$(PYTHON) tests/test_signals.py

test-memory:
	$(PYTHON) tests/test_memory.py

test-tot:
	$(PYTHON) tests/test_tot.py

test-output:
	$(PYTHON) tests/test_output.py

test-tools:
	$(PYTHON) tests/test_tools_noapi.py

test:
	@echo ""
	@echo "=== Running all no-dependency tests ==="
	@echo ""
	$(PYTHON) tests/test_signals.py
	$(PYTHON) tests/test_memory.py
	$(PYTHON) tests/test_tot.py
	$(PYTHON) tests/test_output.py
	$(PYTHON) tests/test_tools_noapi.py
	@echo ""
	@echo "✅ All tests complete."

# ── SEC RAG (requires Ollama running locally) ─────────────────────────────────
run-ingest:
	$(PYTHON) -m sec_rag.ingest

run-retriever:
	$(PYTHON) -m sec_rag.retriever

# ── Live component tests (require Ollama) ─────────────────────────────────────
run-tot:
	$(PYTHON) -m tot.beam_search

run-ooda:
	$(PYTHON) -m chains.ooda_loop

run-scoring:
	$(PYTHON) -m chains.standard_scoring

run-alignment:
	$(PYTHON) -m chains.signal_alignment

run-synthesis:
	$(PYTHON) -m chains.synthesis_chain

run-tot-stance:
	$(PYTHON) -m chains.tot_stance_chain

# ── Live tool tests (require respective API keys) ─────────────────────────────
run-github:
	$(PYTHON) -m tools.github_tool

run-trends:
	$(PYTHON) -m tools.google_trends_tool

run-news:
	$(PYTHON) -m tools.newsdata_tool

run-macro:
	$(PYTHON) -m tools.fred_macro_tool

# ── Agent tests (require Ollama + API keys) ───────────────────────────────────
run-edgar-agent:
	$(PYTHON) -m agents.edgar_rag_agent

run-thought-agent:
	$(PYTHON) -m agents.thought_generator_agent

run-critic-agent:
	$(PYTHON) -m agents.critic_agent

run-github-agent:
	$(PYTHON) -m agents.github_trends_agent

run-news-agent:
	$(PYTHON) -m agents.news_agent

run-macro-agent:
	$(PYTHON) -m agents.macro_agent

# ── Full pipeline ─────────────────────────────────────────────────────────────
run:
	$(PYTHON) main.py $(TICKER)

run-nvda:
	$(PYTHON) main.py NVDA

.PHONY: setup setup-rag check \
        test test-signals test-memory test-tot test-output test-tools \
        run-ingest run-retriever \
        run-tot run-ooda run-scoring run-alignment run-synthesis run-tot-stance \
        run-github run-trends run-news run-macro \
        run-edgar-agent run-thought-agent run-critic-agent run-github-agent \
        run-news-agent run-macro-agent run run-nvda
