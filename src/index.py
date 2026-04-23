# src/index.py
# Technical Plan §4: Embedding & Indexing
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_vector_store(persist_dir: str = "./chroma_db"):
    """Returns initialized ChromaDB with medical-friendly embeddings"""
    # Technical Plan §10: CPU-friendly model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def get_hybrid_retriever(vector_store, chunks):
    """POC: Simple similarity search (expand to hybrid in Phase 2)"""
    return vector_store.as_retriever(search_kwargs={"k": 4})

def medical_fuzzy_boost(query: str, docs: list):
    """POC: No-op boost (preserve order). Expand with BM25 in Phase 2."""
    return docs
