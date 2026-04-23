# src/index.py
# Technical Plan §4: Embedding & Indexing
import os
import sys

# Bootstrap path to allow 'from src...' imports when running from root
sys.path.append(os.getcwd())

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    print("⚠️ Warning: langchain.retrievers not found. Falling back to single retriever.")
    EnsembleRetriever = None
from src.ingest import ingest_medical_textbook

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight for laptop CPU

def get_embedding_function():
    """Initializes the lightweight embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_hybrid_retriever(vector_store, chunks):
    """
    Enhanced Hybrid Search from ACADOC2: Vector + BM25 + Fuzzy Match.
    """
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    if EnsembleRetriever is None:
        return vector_retriever
        
    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 2
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    except Exception as e:
        print(f"⚠️ BM25 Init failed: {e}. Using vector only.")
        return vector_retriever

def medical_fuzzy_boost(query: str, chunks: list):
    """
    Logic from ACADOC2: Boosts chunks containing authoritative definitions.
    """
    boosted = []
    query_norm = query.lower()
    for doc in chunks:
        content = doc.page_content.lower()
        score = 1.0
        # Definition phrase boost (§9)
        if any(p in content for p in ["defined as", "medicine is", "deals with", "refers to"]):
            score += 2.0
        # Exact keyword match boost
        if query_norm in content:
            score += 1.5
        boosted.append((doc, score))
    
    boosted.sort(key=lambda x: x[1], reverse=True)
    return [b[0] for b in boosted]

def build_index(pdf_path: str):
    """
    Ingests a PDF and stores chunks in a local ChromaDB instance.
    """
    chunks = ingest_medical_textbook(pdf_path)
    if not chunks:
        print("⚠️ No chunks to index.")
        return

    print(f"📦 Indexing {len(chunks)} chunks into ChromaDB...")
    db = Chroma.from_documents(
        chunks,
        get_embedding_function(),
        persist_directory=CHROMA_PATH
    )
    # db.persist()  # Chroma 0.4.x handles this automatically
    print("✅ Indexing successful.")

def get_vector_store():
    """Returns a connection to the existing vector store."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Index not found at {CHROMA_PATH}.")
        
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

if __name__ == "__main__":
    data_dir = "./data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if pdf_files:
        build_index(os.path.join(data_dir, pdf_files[0]))
