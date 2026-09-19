"""Backend runner for University Regulation RAG Assistant"""
import sys
import os
import uvicorn

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    print(f"Starting University Regulation RAG Assistant on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
