"""Quick automated test for backend RAG engine and endpoints"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.models import QueryRequest
from app.vector_store import VectorStore
from app.document_processor import FastSemanticEmbedder
from app.rag import UniversityRAGEngine

def test_rag():
    print("[Test] Initializing vector store and RAG engine...")
    vstore = VectorStore(db_dir=os.path.join(BASE_DIR, "data", "vector_db"))
    embedder = FastSemanticEmbedder()
    rag = UniversityRAGEngine(vector_store=vstore, embedder=embedder)

    print(f"[Test] Chunks in vector store: {len(vstore.metadata)}")
    assert len(vstore.metadata) > 0, "Vector store is empty!"

    # Test 1: Minimum Attendance Query
    print("\n--- Test 1: Attendance Question ---")
    req1 = QueryRequest(query="What is the minimum attendance required for examinations?")
    res1 = rag.query(req1)
    print(f"Confidence: {res1.confidence}")
    print(f"Retrieved Chunks: {res1.retrieved_chunks_count}")
    print(f"Processing Time: {res1.processing_time_ms}ms")
    print(f"Citations ({len(res1.citations)}): {[c.source + ' p.' + str(c.page) for c in res1.citations]}")
    print(f"Answer Preview:\n{res1.answer[:300].encode('ascii', 'replace').decode('ascii')}...\n")
    assert res1.retrieved_chunks_count > 0, "Failed to retrieve attendance chunks"

    # Test 2: Irrelevant Question (Threshold Fallback)
    print("--- Test 2: Irrelevant Question (Fallback) ---")
    req2 = QueryRequest(query="How do I bake chocolate chip cookies with butter?", threshold=0.35)
    res2 = rag.query(req2)
    print(f"Confidence: {res2.confidence}")
    print(f"Retrieved Chunks: {res2.retrieved_chunks_count}")
    print(f"Answer:\n{res2.answer}\n")
    assert res2.confidence == "Not Found" or res2.retrieved_chunks_count == 0, "Irrelevant query should not find matches"

    # Test 3: Grading & CGPA
    print("--- Test 3: Grading Scale and CGPA ---")
    req3 = QueryRequest(query="How is CGPA calculated and what is the minimum CGPA to graduate?")
    res3 = rag.query(req3)
    print(f"Confidence: {res3.confidence}")
    print(f"Citations: {[c.source + ' p.' + str(c.page) for c in res3.citations]}")
    assert len(res3.citations) > 0, "Failed to retrieve academic grading chunks"

    print("[SUCCESS] All backend RAG unit tests passed successfully!")

if __name__ == "__main__":
    test_rag()
