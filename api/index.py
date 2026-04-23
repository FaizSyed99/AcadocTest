# api/index.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents import run_pipeline
import uvicorn

app = FastAPI(title="AcaDoc AI API")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
def chat(request: QueryRequest):
    try:
        result = run_pipeline(request.query)
        # Ensure we return a clean JSON-serializable dict
        return {
            "answer": str(result.get("answer", "")),
            "citations": list(result.get("citations", []))
        }
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Issue 4: Port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
