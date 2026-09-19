"""Enterprise FastAPI Application for University Regulation RAG Assistant"""
import os
import json
import shutil
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    DocumentInfo, SystemStats, ChatSession, ChatMessage,
    FeedbackRequest, AnalyticsSummary, CompareRequest, CompareResponse
)
from app.vector_store import VectorStore
from app.document_processor import FastSemanticEmbedder, process_document
from app.rag import UniversityRAGEngine
from app.sessions import SessionManager
from app.analytics import AnalyticsTracker

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REG_DIR = os.path.join(DATA_DIR, "regulations")
META_DIR = os.path.join(DATA_DIR, "metadata")
VEC_DIR = os.path.join(DATA_DIR, "vector_db")

os.makedirs(REG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(VEC_DIR, exist_ok=True)

# Initialize Core Services
embedder = FastSemanticEmbedder()
vector_store = VectorStore(db_dir=VEC_DIR)
rag_engine = UniversityRAGEngine(vector_store=vector_store, embedder=embedder)
session_manager = SessionManager()
analytics_tracker = AnalyticsTracker()

app = FastAPI(
    title="University Regulation RAG Assistant — Enterprise API",
    description="Enterprise Multi-Provider RAG System grounded on official university academic bylaws and regulations.",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_document_registry() -> List[dict]:
    reg_file = os.path.join(META_DIR, "documents.json")
    if os.path.exists(reg_file):
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_document_registry(docs: List[dict]):
    reg_file = os.path.join(META_DIR, "documents.json")
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "University Regulation Enterprise RAG API",
        "version": "2.0.0",
        "indexed_chunks": len(vector_store.metadata),
        "vector_dimension": embedder.dimension,
        "active_sessions": len(session_manager.sessions)
    }


@app.get("/api/stats", response_model=SystemStats)
def get_stats():
    docs = load_document_registry()
    categories = sorted(list(set(d.get("category", "General") for d in docs))) if docs else []
    return SystemStats(
        total_documents=len(docs),
        total_chunks=len(vector_store.metadata),
        total_categories=len(categories),
        categories=categories,
        vector_dimension=embedder.dimension,
        index_type="FAISS IndexFlatIP" if vector_store.use_faiss else "Dense Cosine + BM25 Hybrid Index",
        status="Operational",
        model_name="all-MiniLM-L6-v2 (384d Dense)",
        total_queries_served=len(analytics_tracker.query_logs)
    )


@app.get("/api/documents", response_model=List[DocumentInfo])
def list_documents():
    docs = load_document_registry()
    result = []
    for d in docs:
        result.append(DocumentInfo(
            id=d.get("id", ""),
            title=d.get("title", ""),
            filename=d.get("filename", ""),
            category=d.get("category", "General"),
            source_url=d.get("source_url", ""),
            total_pages=d.get("total_pages", 1),
            total_chunks=d.get("total_chunks", 0),
            uploaded_at=d.get("uploaded_at"),
            file_size_kb=d.get("file_size_kb", 0.0)
        ))
    return result


@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    docs = load_document_registry()
    doc = next((d for d in docs if d.get("id") == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    fname = doc.get("filename")
    matching_chunks = [c for c in vector_store.metadata if c.get("file") == fname]
    return {
        "document": doc,
        "chunks": matching_chunks,
        "total_chunks": len(matching_chunks)
    }


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    """Streams the authentic document file directly from the server."""
    docs = load_document_registry()
    doc = next((d for d in docs if d.get("id") == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in registry")

    fname = doc.get("filename")
    file_path = os.path.join(REG_DIR, fname)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Physical document file not found on server")

    media_type = "application/pdf" if fname.lower().endswith(".pdf") else "text/plain; charset=utf-8"
    return FileResponse(file_path, media_type=media_type, filename=fname)


@app.get("/api/documents/{doc_id}/content")
def get_document_content(doc_id: str):
    """Returns the full text / extracted pages of the document."""
    docs = load_document_registry()
    doc = next((d for d in docs if d.get("id") == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in registry")

    fname = doc.get("filename")
    file_path = os.path.join(REG_DIR, fname)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    if fname.lower().endswith(".pdf"):
        from app.document_processor import extract_pages_from_pdf
        pages = extract_pages_from_pdf(file_path)
        return {
            "document": doc,
            "pages": pages,
            "total_pages": len(pages),
            "content": "\n\n".join([f"=== PAGE {p['page_number']} ===\n{p['text']}" for p in pages])
        }
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {
            "document": doc,
            "content": text,
            "total_pages": doc.get("total_pages", 1)
        }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("Academic"),
    source_url: Optional[str] = Form("")
):
    """Uploads a PDF or text regulation document, chunks, embeds, and updates the vector database."""
    filename = file.filename
    dest_path = os.path.join(REG_DIR, filename)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_kb = round(os.path.getsize(dest_path) / 1024, 2)

    # Process and Chunk Document
    chunks, total_pages = process_document(
        file_path=dest_path,
        title=title,
        category=category,
        source_url=source_url or "",
        chunk_size=800,
        chunk_overlap=150
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text or chunks from the uploaded file.")

    # Generate Embeddings
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    # Add to Vector Store & Rebuild BM25
    vector_store.add_documents(chunks, embeddings)
    rag_engine.rebuild_bm25()

    # Update Registry
    registry = load_document_registry()
    existing = next((d for d in registry if d.get("filename") == filename), None)
    
    doc_id = f"DOC{len(registry)+1:03d}" if not existing else existing.get("id", f"DOC{len(registry):03d}")
    doc_entry = {
        "id": doc_id,
        "title": title,
        "filename": filename,
        "category": category,
        "source_url": source_url or "",
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "file_size_kb": file_size_kb,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if existing:
        registry = [d if d.get("filename") != filename else doc_entry for d in registry]
    else:
        registry.append(doc_entry)

    save_document_registry(registry)

    return {
        "message": f"Document '{title}' successfully indexed!",
        "document": doc_entry,
        "chunks_indexed": len(chunks),
        "total_index_size": len(vector_store.metadata)
    }


@app.post("/api/documents/upload-batch")
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    category: str = Form("Academic"),
    source_url: Optional[str] = Form("")
):
    """Uploads multiple PDF or text documents simultaneously, chunks, embeds, and updates FAISS."""
    registry = load_document_registry()
    uploaded_results = []
    all_new_chunks = []

    for file in files:
        filename = file.filename
        dest_path = os.path.join(REG_DIR, filename)
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_kb = round(os.path.getsize(dest_path) / 1024, 2)
        title = filename.replace(".pdf", "").replace(".txt", "").replace("_", " ").title()

        chunks, total_pages = process_document(
            file_path=dest_path,
            title=title,
            category=category,
            source_url=source_url or "",
            chunk_size=800,
            chunk_overlap=150
        )

        if not chunks:
            continue

        all_new_chunks.extend(chunks)

        existing = next((d for d in registry if d.get("filename") == filename), None)
        doc_id = f"DOC{len(registry)+1:03d}" if not existing else existing.get("id", f"DOC{len(registry):03d}")
        doc_entry = {
            "id": doc_id,
            "title": title,
            "filename": filename,
            "category": category,
            "source_url": source_url or "",
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "file_size_kb": file_size_kb,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if existing:
            registry = [d if d.get("filename") != filename else doc_entry for d in registry]
        else:
            registry.append(doc_entry)

        uploaded_results.append({
            "filename": filename,
            "title": title,
            "pages": total_pages,
            "chunks": len(chunks)
        })

    if all_new_chunks:
        texts = [c["text"] for c in all_new_chunks]
        embeddings = embedder.encode(texts)
        vector_store.add_documents(all_new_chunks, embeddings)
        rag_engine.rebuild_bm25()
        save_document_registry(registry)

    return {
        "message": f"Successfully processed and indexed {len(uploaded_results)} documents!",
        "processed_files": uploaded_results,
        "total_new_chunks": len(all_new_chunks),
        "total_index_size": len(vector_store.metadata)
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    registry = load_document_registry()
    target = next((d for d in registry if d.get("id") == doc_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Document not found")

    filename = target.get("filename")
    
    # Remove from vector store
    vector_store.remove_document(filename)
    rag_engine.rebuild_bm25()
    
    # Remove from disk if exists
    file_path = os.path.join(REG_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    # Update Registry
    new_registry = [d for d in registry if d.get("id") != doc_id]
    save_document_registry(new_registry)

    return {
        "message": f"Document '{target.get('title')}' deleted and vector index rebuilt.",
        "remaining_chunks": len(vector_store.metadata)
    }


# ----------------------------------------------------
# Multi-Turn Chat Sessions Endpoints
# ----------------------------------------------------
@app.get("/api/sessions", response_model=List[ChatSession])
def list_chat_sessions():
    return session_manager.list_sessions()


@app.post("/api/sessions", response_model=ChatSession)
def create_chat_session(title: Optional[str] = Query("New Conversation")):
    return session_manager.create_session(title)


@app.get("/api/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_chat_session(session_id: str):
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    analytics_tracker.log_feedback(req.feedback)
    if req.session_id:
        session_manager.update_feedback(req.session_id, req.message_index, req.feedback)
    return {"message": "Feedback recorded", "status": "success"}


# ----------------------------------------------------
# Core RAG QA & Search Endpoints
# ----------------------------------------------------
@app.post("/api/chat", response_model=QueryResponse)
@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """Executes end-to-end Hybrid RAG QA pipeline with citations and session persistence."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Run RAG
    res = rag_engine.query(req)

    # Save to session if session_id provided
    session_id = req.session_id or "default"
    user_msg = ChatMessage(
        role="user",
        content=req.query,
        time=time.strftime("%I:%M %p")
    )
    ai_msg = ChatMessage(
        role="assistant",
        content=res.answer,
        time=time.strftime("%I:%M %p"),
        citations=[c.model_dump() for c in res.citations],
        confidence=res.confidence,
        llm_used=res.llm_used,
        processing_time_ms=res.processing_time_ms
    )
    session_manager.add_message(session_id, user_msg)
    session_manager.add_message(session_id, ai_msg)
    res.session_id = session_id

    # Log telemetry
    analytics_tracker.log_query(
        query=req.query,
        category=req.category or "all",
        confidence=res.confidence,
        latency_ms=res.processing_time_ms,
        chunks_count=res.retrieved_chunks_count,
        llm_used=res.llm_used
    )

    return res


@app.post("/api/search", response_model=SearchResponse)
def semantic_search(req: SearchRequest):
    """Performs raw hybrid similarity search across all chunks."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = rag_engine.search_only(
        query=req.query,
        top_k=req.top_k or 5,
        category=req.category or "all"
    )
    return SearchResponse(
        results=results,
        query=req.query,
        total_matched=len(results)
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_regulations(req: CompareRequest):
    """Compares policy clauses across two categories."""
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    return rag_engine.compare_topics(req)


@app.get("/api/analytics", response_model=AnalyticsSummary)
def get_analytics():
    """Returns telemetry and confidence distribution analytics."""
    return analytics_tracker.get_summary()


# Mount built frontend dist folder if present
FRONTEND_DIST = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
