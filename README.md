# AcaDoc AI: Medical Education RAG (POC v1.2)

**AcaDoc AI** is a specialized, textbook-grounded AI system designed specifically for the Indian medical curriculum (MBBS/PG/BDS). Unlike generic LLMs, AcaDoc AI prioritizes **factual accuracy** and **source traceability** over generative fluency.

## 🩺 Core USP (Technical Plan §9)
- **Zero Hallucination Pipeline:** Every claim is validated against ingested textbooks before delivery.
- **Curriculum Aligned:** Structured to answer based on standard references (e.g., KD Tripathi, Shanbhag).
- **Source-Backed:** Every response includes page-level citations and a confidence score.

## 🛠️ POC Architecture (§8)
1. **Ingest:** Heading-aware PDF chunking with metadata preservation.
2. **Index:** Local ChromaDB vector storage.
3. **Agentic Pipeline:** 
   - **Retriever:** Fetches context from local index.
   - **Validator:** Gates output based on context sufficiency.
   - **Generator:** Produces exam-oriented, determinisitic answers (Temp=0.0).

## 🚀 Running Locally (§10)
Requires [Ollama](https://ollama.com/) with `gemma2:9b` or `llama3` installed.
1. `pip install -r requirements.txt`
2. Place medical PDF in `./data/`
3. `python src/ingest.py`
4. `python api/index.py`
