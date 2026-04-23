# api/index.py
# Technical Plan §10: Start Small, Iterate
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Bootstrap path to allow 'from src...' imports
sys.path.append(os.getcwd())

from src.agents import run_pipeline

app = FastAPI(title="AcaDoc AI API", version="1.2.0")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    is_sufficient: bool
    confidence: float

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "project": "AcaDoc AI"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Triggers the 3-agent pipeline: Retrieve -> Validate -> Generate.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    print(f"🚀 Incoming Medical Query: {request.query}")
    
    try:
        # Run the agentic pipeline (§8)
        result = run_pipeline(request.query)
        
        return ChatResponse(
            answer=result['answer'],
            citations=result['citations'],
            is_sufficient=result['is_sufficient'],
            confidence=result['confidence']
        )
    except Exception as e:
        print(f"🔥 API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Technical Plan §10: POC runs locally on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
