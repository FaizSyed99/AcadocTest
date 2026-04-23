# src/ingest.py
# Technical Plan §3: Textbook Grounding & §4: Preprocessing
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def ingest_medical_textbook(file_path: str):
    """
    Loads a medical PDF and splits it into heading-aware chunks.
    Preserves page numbers and source metadata for traceability.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return []

    print(f"📖 Loading textbook: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Technical Plan §4: Heading-aware splitting
    # We prioritize splitting at headings to keep section context intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n# ", "\n\n", "\n", " "]
    )

    chunks = []
    print(f"✂️ Splitting into semantic chunks...")  # noqa: F541
    
    for page in pages:
        # Extract basic metadata
        source = os.path.basename(file_path)
        page_num = page.metadata.get("page", 0) + 1
        
        # Split text while keeping metadata
        page_chunks = text_splitter.split_text(page.page_content)
        
        for i, chunk_text in enumerate(page_chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"{source}_p{page_num}_{i}",
                    "project": "AcaDoc AI"
                }
            )
            chunks.append(doc)

    print(f"✅ Ingestion complete. Generated {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # POC Test: Look for any PDF in the /data folder
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if pdf_files:
        sample_pdf = os.path.join(data_dir, pdf_files[0])
        ingest_medical_textbook(sample_pdf)
    else:
        print("ℹ️ No PDF found in /data. Please place a medical textbook PDF there to test.")
