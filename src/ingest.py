# src/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_medical_textbook(pdf_path: str):
    """Technical Plan §3: Heading-aware PDF chunking."""
    # Issue 5: Graceful fallback if no PDF
    if not os.path.exists(pdf_path):
        print(f"⚠️ Warning: PDF not found at {pdf_path}")
        return []

    print(f"📖 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_documents(pages)
    
    # Metadata preservation (§3)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = os.path.basename(pdf_path)
        
    return chunks
