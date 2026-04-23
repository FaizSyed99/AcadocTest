@echo off
echo 🩺 Initializing AcaDoc AI POC v1.2...

:: 1. Cleanup stale files
echo 🧹 Cleaning up stale files...
if exist api\__pycache__ rmdir /s /q api\__pycache__
if exist src\__pycache__ rmdir /s /q src\__pycache__

:: 2. Install Dependencies
echo 🛠️ Ensuring dependencies are present (Skip-Build Mode)...
pip install --prefer-binary fastapi uvicorn pypdf langchain langchain-community chromadb sentence-transformers ollama pydantic python-multipart streamlit requests rank_bm25

:: 3. Start Backend
echo 🚀 Starting Backend API...
set PYTHONPATH=%CD%
start "AcaDoc-API" cmd /k "set PYTHONPATH=%CD% && python -m api.index"

:: 4. Start Frontend
echo 🎨 Starting Frontend UI...
:: Use python for delay if ping is missing
python -c "import time; time.sleep(5)"
python -m streamlit run src/app.py --browser.gatherUsageStats false

echo ✅ System is running!
