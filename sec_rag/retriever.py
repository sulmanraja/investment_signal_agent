from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_ollama import OllamaEmbeddings
from sec_rag.config import EMBEDDING_MODEL, FAISS_PATH, LLM


def load_vectorstore() -> FAISS:
    #embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(
        f"[{doc.metadata.get('ticker')} | {doc.metadata.get('date')}]\n{doc.page_content}"
        for doc in docs
    )


def semantic_search(query: str, ticker: str = "", k: int = 5):
    """Pure similarity search with optional ticker post-filter."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k * 3)

    if ticker:
        results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("ticker") == ticker
        ]

    results = results[:k]

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} | score: {score:.4f} | {doc.metadata} ---")
        print(doc.page_content[:400])

    return results


def build_rag_chain(ticker: str = ""):
    """
    Build a RAG chain using LCEL.
    Replaces the deprecated RetrievalQA.from_chain_type().
    """
    vectorstore = load_vectorstore()

    # MMR retriever — reduces repetitive boilerplate chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
    )

    prompt = PromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question using only the 
earnings call transcript excerpts provided below. If the answer isn't in 
the context, say "Not found in the provided transcripts."

Context:
{context}

Question: {question}

Answer:""")

    llm = Ollama(model=LLM)   # or "mistral", "phi3"

    # ── LCEL chain ─────────────────────────────────────────────────────────
    chain = (
        RunnableMap({
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def rag_query(query: str, ticker: str = ""):
    """Run a RAG query and print the answer + source documents."""
    chain, retriever = build_rag_chain(ticker=ticker)

    # Stream the answer token by token
    print("\nAnswer: ", end="", flush=True)
    answer = ""
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)
        answer += chunk
    print()

    # Show source documents separately
    source_docs = retriever.invoke(query)
    if ticker:
        source_docs = [d for d in source_docs if d.metadata.get("ticker") == ticker]

    print("\nSources:")
    for doc in source_docs:
        print(f"  - {doc.metadata.get('ticker')} | {doc.metadata.get('date')} | {doc.metadata.get('accession')}")

    return {"answer": answer, "source_documents": source_docs}


if __name__ == "__main__":
    # Semantic search — no LLM needed
    semantic_search("capital expenditure guidance AI infrastructure", ticker="MSFT")
    semantic_search("capital expenditure guidance AI infrastructure", ticker="NVDA")
    semantic_search("capital expenditure guidance AI infrastructure", ticker="AAPL")

    # RAG query — requires Ollama running locally
    #rag_query("What risks did management highlight?", ticker="NVDA")
    rag_query("What was said about gross margins?", ticker="MSFT")
    rag_query("What risks did management highlight?", ticker="MSFT")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="MSFT")

    rag_query("What was said about gross margins?", ticker="NVDA")
    rag_query("What risks did management highlight?", ticker="NVDA")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="NVDA")

    rag_query("What was said about gross margins?", ticker="AAPL")
    rag_query("What risks did management highlight?", ticker="AAPL")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="AAPL")

    rag_query("What was said about gross margins?", ticker="GOOG")
    rag_query("What risks did management highlight?", ticker="GOOG")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="GOOG")

    rag_query("What was said about gross margins?", ticker="AMZN")
    rag_query("What risks did management highlight?", ticker="AMZN")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="AMZN")

    rag_query("What was said about gross margins?", ticker="META")
    rag_query("What risks did management highlight?", ticker="META")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="META")

    rag_query("What was said about gross margins?", ticker="INTC")
    rag_query("What risks did management highlight?", ticker="INTC")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="INTC")

    rag_query("What was said about gross margins?", ticker="CSCO")
    rag_query("What risks did management highlight?", ticker="CSCO")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="CSCO")

    rag_query("What was said about gross margins?", ticker="ADBE")
    rag_query("What risks did management highlight?", ticker="ADBE")
    rag_query("What are the company announcements related to AI infrastructure?", ticker="ADBE")

