# src/agents.py
# Technical Plan §8: Agent-Driven Architecture & §9: Near-Zero Hallucination
import os
import sys
import json
import ollama
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from src.index import get_vector_store, medical_fuzzy_boost

# Load environment variables (API Keys)
load_dotenv()

# Technical Plan §10: Sync pipeline for POC simplicity
sys.path.append(os.getcwd())

# Technical Plan §9: Strictly Deterministic for medical safety
LLM_MODEL = os.getenv("ACADOC_MODEL", "gpt-4o-mini") 
TEMPERATURE = 0.0 

def get_llm():
    """Model Factory: Uses Cloud if API key is present, otherwise fails gracefully."""
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if api_key:
        from langchain_openai import ChatOpenAI
        # Default to OpenRouter if using OLLAMA_API_KEY style keys
        base_url = os.getenv("CLOUD_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("ACADOC_MODEL", "openai/gpt-4o-mini")
        
        print(f"☁️ [LLM] Connecting to Cloud: {model_name} via {base_url}")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=TEMPERATURE,
            base_url=base_url
        )
    else:
        print("🏠 [LLM] No API Key found. Falling back to local Ollama.")
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="gemma2:9b", temperature=TEMPERATURE)

class AgentState(TypedDict):
    query: str
    context_chunks: List[Any]
    is_sufficient: bool
    confidence: float
    answer: str
    citations: List[str]

def retrieve_agent(state: AgentState) -> AgentState:
    """Technical Plan §4: Fetches top-4 relevant textbook chunks."""
    print(f"🔍 [Retriever] Searching for: {state['query']}")
    try:
        vs = get_vector_store()
        # Issue 2: Direct similarity search for POC stability
        docs = vs.similarity_search(state['query'], k=4)
        state['context_chunks'] = medical_fuzzy_boost(state['query'], docs)
        
        # Issue 5: Immediate fallback if no context found
        if not state['context_chunks']:
            state['is_sufficient'] = False
            
    except Exception as e:
        print(f"❌ [Retriever] Error: {e}")
        state['context_chunks'] = []
    return state

def validate_agent(state: AgentState) -> AgentState:
    """Technical Plan §9: Strict gatekeeper logic."""
    if not state['context_chunks']:
        state['is_sufficient'] = False
        return state

    context_text = "\n---\n".join([d.page_content for d in state['context_chunks']])
    
    prompt = f"""
    SYSTEM: You are a Medical Validator. Determine if the TEXTBOOK CONTEXT is sufficient to answer the QUERY.
    QUERY: {state['query']}
    CONTEXT: {context_text}
    
    STRICT JSON RESPONSE:
    {{
        "is_sufficient": bool,
        "confidence": float,
        "reason": "explanation"
    }}
    """
    
    try:
        # Issue 3: Sync generation via Factory
        llm = get_llm()
        response = llm.invoke(prompt)
        # Handle both string and Message objects
        res_text = response.content if hasattr(response, 'content') else str(response)
        result = json.loads(res_text)
        state['is_sufficient'] = result.get('is_sufficient', False)
        state['confidence'] = result.get('confidence', 0.0)
    except Exception as e:
        print(f"❌ [Validator] Error: {e}")
        state['is_sufficient'] = False
        
    return state

def generate_agent(state: AgentState) -> AgentState:
    """Technical Plan §9: Grounded generation or fallback."""
    if not state['is_sufficient']:
        state['answer'] = "I'm sorry, but the current textbook context does not contain sufficient information. AcaDoc AI refuses to speculate on medical topics."
        return state

    context_text = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(state['context_chunks'])])
    prompt = f"Using ONLY this context: {context_text}\n\nAnswer this: {state['query']}"
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        state['answer'] = response.content if hasattr(response, 'content') else str(response)
        state['citations'] = [f"{d.metadata.get('source')} (Page {d.metadata.get('page')})" for d in state['context_chunks']]
    except Exception as e:
        state['answer'] = f"Error in grounded generation: {e}"
        
    return state

def run_pipeline(query: str) -> AgentState:
    """Technical Plan §8: Coordinated sync flow."""
    state = AgentState(query=query, context_chunks=[], is_sufficient=False, confidence=0.0, answer="", citations=[])
    state = retrieve_agent(state)
    state = validate_agent(state)
    state = generate_agent(state)
    return state
