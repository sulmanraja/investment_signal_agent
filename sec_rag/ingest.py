import os
import requests
import time
import re
from bs4 import BeautifulSoup
# Alternative if langchain_community.embeddings is unresolved
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from sec_rag.config import EMBEDDING_MODEL, FAISS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

HEADERS = {"User-Agent": "SulmanRaja sulman.raja@gmail.com", 
               "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov"}

BASE = "https://data.sec.gov"

# ── Step 1: Ticker → CIK ──────────────────────────────────────────────────
def get_cik(ticker: str) -> str | None:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers={**HEADERS, "Host": "www.sec.gov"})
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"] == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            print(f"  CIK for {ticker}: {cik}")
            return cik
    print(f"  CIK not found for {ticker}")
    return None


# ── Step 2: Get 10-K / 10-Q filings ──────────────────────────────────────
def get_filings(cik: str, form_types: list[str], max_filings: int = 3) -> list[dict]:
    """Fetch recent filings of specified form types (10-K, 10-Q) for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings  = data.get("filings", {}).get("recent", {})
    forms    = filings.get("form", [])
    accnos   = filings.get("accessionNumber", [])
    dates    = filings.get("filingDate", [])
    primdocs = filings.get("primaryDocument", [])

    results = []
    for form, accno, date, doc in zip(forms, accnos, dates, primdocs):
        if form in form_types:
            results.append({
                "form":      form,
                "accession": accno,
                "date":      date,
                "primary_doc": doc,
            })
        if len(results) >= max_filings:
            break

    print(f"  Found {len(results)} filings ({', '.join(form_types)})")
    return results


# ── Step 3: Fetch and extract full filing text ────────────────────────────
def get_filing_documents(cik: str, accession: str) -> list[dict]:
    """
    Return all document entries from a filing's index page.
    Each entry has: description, url, doc_type.
    """
    accession_clean = accession.replace("-", "")
    index_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={int(cik)}"
        f"&type=10-K&dateb=&owner=include&count=1&search_text="
    )
    # Use the JSON index instead — more reliable
    json_index_url = (
        f"https://data.sec.gov/submissions/CIK{cik}.json"
    )
    filing_index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{accession_clean}-index.json"
    )
    try:
        resp = requests.get(
            filing_index_url,
            headers={**HEADERS, "Host": "www.sec.gov"}
        )
        resp.raise_for_status()
        index_data = resp.json()
        docs = []
        for item in index_data.get("documents", []):
            docs.append({
                "description": item.get("description", ""),
                "name":        item.get("name", ""),
                "type":        item.get("type", ""),
            })
        return docs
    except Exception as e:
        print(f"    Index fetch failed: {e}")
        return []


def fetch_and_clean_html(url: str) -> str:
    """Download an HTML filing and return clean plain text."""
    try:
        resp = requests.get(url, headers={**HEADERS, "Host": "www.sec.gov"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "ix:header",
                         "ix:nonfraction", "table"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text
    except Exception as e:
        print(f"    Fetch failed ({url}): {e}")
        return ""


# ── Step 4: Extract the MD&A section ─────────────────────────────────────
def extract_mda(text: str) -> str:
    """
    Extract the Management Discussion & Analysis section.
    10-K Item 7 / 10-Q Item 2.
    Targets the dense investment-signal prose (10k–30k words).
    """
    # Patterns that mark the START of MD&A
    start_patterns = [
        r"item\s*7[\.\s]*management.{0,10}discussion",
        r"item\s*2[\.\s]*management.{0,10}discussion",
        r"management.s discussion and analysis of financial condition",
    ]

    # Patterns that mark the END of MD&A (the next section)
    end_patterns = [
        r"item\s*7a[\.\s]*quantitative",
        r"item\s*8[\.\s]*financial statements",
        r"item\s*3[\.\s]*quantitative",
        r"item\s*4[\.\s]*controls",
    ]

    text_lower = text.lower()
    start_idx = None

    for pattern in start_patterns:
        match = re.search(pattern, text_lower)
        if match:
            start_idx = match.start()
            break

    if start_idx is None:
        print("    ⚠ MD&A section not found — using full document text")
        return text[:100000]

    end_idx = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text_lower[start_idx + 500:])
        if match:
            end_idx = start_idx + 500 + match.start()
            break

    mda_text = text[start_idx:end_idx].strip()
    word_count = len(mda_text.split())
    print(f"    MD&A extracted: ~{word_count:,} words")
    return mda_text


# ── Step 5: Semantic chunking ─────────────────────────────────────────────
def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    Splitter tuned for long-form financial prose.
    Larger chunks preserve more context per retrieval hit.
    Separators follow document structure: section → paragraph → sentence.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\nItem ",       # filing section boundary
            "\n\n",            # paragraph boundary
            "\n",              # line boundary
            ". ",              # sentence boundary
            " ",               # word boundary (last resort)
        ]
    )


# ── Main ingest ───────────────────────────────────────────────────────────
def ingest(tickers: list[str], form_types=["10-K", "10-Q"],
           max_filings_per_ticker: int = 3):
    #embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    splitter = make_splitter()
    all_chunks: list[Document] = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        cik = get_cik(ticker)
        if not cik:
            continue

        filings = get_filings(cik, form_types, max_filings=max_filings_per_ticker)

        for filing in filings:
            print(f"  [{filing['form']}] {filing['date']} — {filing['accession']}")
            accession_clean = filing["accession"].replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{filing['primary_doc']}"
            )

            raw_text = fetch_and_clean_html(doc_url)
            if not raw_text:
                print("    ⚠ Empty document, skipping.")
                continue

            # Extract MD&A — the high-signal prose section
            mda_text = extract_mda(raw_text)
            if not mda_text:
                continue

            chunks = splitter.create_documents(
                texts=[mda_text],
                metadatas=[{
                    "ticker":    ticker,
                    "form":      filing["form"],
                    "date":      filing["date"],
                    "accession": filing["accession"],
                    "section":   "MD&A",
                }]
            )
            all_chunks.extend(chunks)
            print(f"    ✓ {len(chunks)} chunks indexed")
            time.sleep(0.2)

    if not all_chunks:
        print("\n❌ No chunks indexed.")
        return

    print(f"\nBuilding FAISS index over {len(all_chunks)} total chunks...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ Index saved to '{FAISS_PATH}/'")


def ingest_incremental(new_tickers: list[str], form_types=["10-K", "10-Q"],
                       max_filings: int = 3):
    """Merge new tickers into an existing FAISS index."""
    #embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(
            FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("Loaded existing FAISS index.")
    else:
        vectorstore = None

    splitter = make_splitter()

    for ticker in new_tickers:
        print(f"\nProcessing {ticker}...")
        cik = get_cik(ticker)
        if not cik:
            continue

        filings = get_filings(cik, form_types, max_filings=max_filings)
        new_chunks: list[Document] = []

        for filing in filings:
            print(f"  [{filing['form']}] {filing['date']}")
            accession_clean = filing["accession"].replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{filing['primary_doc']}"
            )
            raw_text = fetch_and_clean_html(doc_url)
            if not raw_text:
                continue

            mda_text = extract_mda(raw_text)
            if not mda_text:
                continue

            chunks = splitter.create_documents(
                texts=[mda_text],
                metadatas=[{
                    "ticker":  ticker,
                    "form":    filing["form"],
                    "date":    filing["date"],
                    "section": "MD&A",
                }]
            )
            new_chunks.extend(chunks)
            time.sleep(0.2)

        if new_chunks:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(new_chunks, embeddings)
            else:
                vectorstore.add_documents(new_chunks)
            print(f"  ✓ Added {len(new_chunks)} chunks for {ticker}")

    if vectorstore:
        vectorstore.save_local(FAISS_PATH)
        print(f"\n✅ Index updated and saved to '{FAISS_PATH}/'")

if __name__ == "__main__":
    ingest(["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "INTC", "CSCO", "ADBE"], 
           form_types=["10-K", "10-Q"],
            max_filings_per_ticker=10)

