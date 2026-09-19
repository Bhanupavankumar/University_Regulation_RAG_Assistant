"""Expanded Data Models for Enterprise University Regulation RAG System"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    text: str
    source: str
    file: str
    page: int
    category: str
    url: Optional[str] = ""

class DocumentInfo(BaseModel):
    id: str
    title: str
    filename: str
    category: str
    source_url: Optional[str] = ""
    total_pages: Optional[int] = 1
    total_chunks: Optional[int] = 0
    uploaded_at: Optional[str] = None
    file_size_kb: Optional[float] = 0.0

class ChatMessage(BaseModel):
    role: str # "user" | "assistant" | "system"
    content: str
    time: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = []
    confidence: Optional[str] = None
    llm_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    feedback: Optional[str] = None # "like" | "dislike"

class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = []

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 4
    category: Optional[str] = "all"
    threshold: Optional[float] = 0.35
    llm_provider: Optional[str] = "local" # local, gemini, groq, openai, ollama
    api_key: Optional[str] = None
    temperature: Optional[float] = 0.2
    hybrid_weight: Optional[float] = 0.65 # Dense vector vs BM25 weight

class Citation(BaseModel):
    id: int
    source: str
    page: int
    file: str
    category: str
    snippet: str
    score: float
    verified: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str # "High", "Medium", "Low", "Not Found"
    retrieved_chunks_count: int
    llm_used: str
    processing_time_ms: float
    query: str
    follow_up_questions: List[str] = []
    session_id: Optional[str] = None
    hallucination_check: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    category: Optional[str] = "all"

class SearchResultChunk(BaseModel):
    text: str
    source: str
    file: str
    page: int
    category: str
    url: Optional[str] = ""
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResultChunk]
    query: str
    total_matched: int

class CompareRequest(BaseModel):
    topic: str
    category_a: Optional[str] = None
    category_b: Optional[str] = None
    doc_id_a: Optional[str] = None
    doc_id_b: Optional[str] = None

class CompareResponse(BaseModel):
    topic: str
    perspective_a: Dict[str, Any]
    perspective_b: Dict[str, Any]
    comparison_summary: str

class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    message_index: int
    feedback: str # "like" | "dislike"
    comment: Optional[str] = None

class AnalyticsSummary(BaseModel):
    total_queries: int
    average_latency_ms: float
    confidence_breakdown: Dict[str, int]
    top_queried_categories: Dict[str, int]
    feedback_stats: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_categories: int
    categories: List[str]
    vector_dimension: int
    index_type: str
    status: str
    model_name: str
    total_queries_served: int
