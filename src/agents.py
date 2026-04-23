# src/agents.py
# Technical Plan §8: Agent-Driven Architecture & §9: Near-Zero Hallucination
import os
import sys
import json
import ollama
from typing import List, Dict, Any, TypedDict

# Bootstrap path to allow 'from src...' imports
sys.path.append(os.getcwd())

from src.index import get_vector_store

# Technical Plan §4: Controlled Generation - Strictly Deterministic
LLM_MODEL = os.getenv("ACADOC_MODEL", "gemma2:9b") 
TEMPERATURE = 0.0 

class AgentState(TypedDict):
    query: str
    context_chunks: List[Any]
    is_sufficient: bool
    confidence: float
    answer: str
    citations: List[str]

def expand_medical_query(query: str) -> list[str]:
    """
    Logic from ACADOC2: Expands medical queries to match textbook phrasing (§4).
    """
    expansions = [query]
    q_lower = query.lower().strip()
    
    # Detect definitional intent
    if any(t in q_lower for t in ["what is", "define", "concept", "explain"]):
        expansions.append(f"Definition of {query}")
        expansions.append(f"Introduction to {query}")
        
    return list(set(expansions))

def retrieve_agent(state: AgentState) -> AgentState:
    """Fetches textbook chunks using Smart Hybrid Search + Query Expansion."""
    expanded_queries = expand_medical_query(state['query'])
    print(f"🔍 [Retriever] Expanded Queries: {expanded_queries}")
    
    try:
        vector_store = get_vector_store()
        from src.ingest import ingest_medical_textbook
        data_dir = "./data"
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
        if not pdf_files:
            state['context_chunks'] = []
            return state
            
        chunks = ingest_medical_textbook(os.path.join(data_dir, pdf_files[0]))
        
        from src.index import get_hybrid_retriever, medical_fuzzy_boost
        retriever = get_hybrid_retriever(vector_store, chunks)
        
        # Retrieve for each expanded query
        all_docs = []
        for q in expanded_queries:
            all_docs.extend(retriever.get_relevant_documents(q))
            
        # Deduplicate and Boost (§9)
        unique_docs = {d.page_content: d for d in all_docs}.values()
        state['context_chunks'] = medical_fuzzy_boost(state['query'], list(unique_docs))[:4]
        
    except Exception as e:
        print(f"❌ [Retriever] Error: {e}")
        state['context_chunks'] = []
    return state

def validate_agent(state: AgentState) -> AgentState:
    """Strict gatekeeper: Checks if context contains the answer."""
    if not state['context_chunks']:
        state['is_sufficient'] = False
        return state

    context_text = "\n---\n".join([d.page_content for d in state['context_chunks']])
    
    prompt = f"""
    SYSTEM: You are a Medical Validator for AcaDoc AI. 
    TASK: Determine if the following TEXTBOOK CONTEXT is sufficient to answer the USER QUERY.
    RULES:
    1. If the context does not explicitly contain the answer, set is_sufficient to false.
    2. Be extremely conservative. Do not use outside knowledge.
    
    USER QUERY: {state['query']}
    TEXTBOOK CONTEXT: {context_text}
    
    RESPONSE FORMAT (STRICT JSON):
    {{
        "is_sufficient": bool,
        "confidence": float (0.0 to 1.0),
        "reason": "Brief explanation"
    }}
    """
    
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt, format='json', options={'temperature': 0})
        result = json.loads(response['response'])
        state['is_sufficient'] = result.get('is_sufficient', False)
        state['confidence'] = result.get('confidence', 0.0)
    except Exception as e:
        print(f"❌ [Validator] Error: {e}")
        state['is_sufficient'] = False
        state['confidence'] = 0.0
        
    print(f"🛡️ [Validator] Sufficient: {state['is_sufficient']} (Score: {state['confidence']})")
    return state

def generate_agent(state: AgentState) -> AgentState:
    """Final generation or graceful fallback."""
    if not state['is_sufficient']:
        state['answer'] = "I'm sorry, but the current textbook context does not contain sufficient information to answer this accurately. AcaDoc AI prioritizes medical safety and refuses to speculate."
        state['citations'] = []
        return state

    context_text = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(state['context_chunks'])])
    
    prompt = f"""
    SYSTEM: You are AcaDoc AI, a medical tutor. 
    TASK: Answer the query using ONLY the provided context. 
    STYLE: Professional, exam-aligned, structured.
    
    CONTEXT:
    {context_text}
    
    USER QUERY: {state['query']}
    """
    
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt, options={'temperature': TEMPERATURE})
        state['answer'] = response['response']
        
        # Extract citations from metadata
        state['citations'] = list(set([
            f"{d.metadata.get('source')} (Page {d.metadata.get('page')})" 
            for d in state['context_chunks']
        ]))
    except Exception as e:
        print(f"❌ [Generator] Error: {e}")
        state['answer'] = "Error generating response."
        
    print("✍️ [Generator] Response generated with citations.")
    return state

def run_pipeline(query: str) -> AgentState:
    """Coordinates the 3-agent flow."""
    state = AgentState(
        query=query, 
        context_chunks=[], 
        is_sufficient=False, 
        confidence=0.0, 
        answer="", 
        citations=[]
    )
    
    state = retrieve_agent(state)
    state = validate_agent(state)
    state = generate_agent(state)
    
    return state

if __name__ == "__main__":
    test_query = "What are the common clinical features mentioned in the text?"
    res = run_pipeline(test_query)
    print(f"\nFINAL ANSWER:\n{res['answer']}")
    print(f"\nSOURCES:\n{res['citations']}")
