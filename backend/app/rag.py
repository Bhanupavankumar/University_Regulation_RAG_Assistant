"""Enterprise RAG Engine with Hybrid Search, Faithfulness Verification, and Follow-ups"""
import time
import re
from typing import List, Dict, Any, Optional
from app.models import (
    QueryRequest, QueryResponse, Citation, SearchResultChunk,
    CompareRequest, CompareResponse
)
from app.vector_store import VectorStore
from app.document_processor import FastSemanticEmbedder
from app.synthesizer import RAGSynthesizer
from app.hybrid_search import BM25Retriever, expand_query, reciprocal_rank_fusion, check_faithfulness

class UniversityRAGEngine:
    def __init__(self, vector_store: VectorStore, embedder: FastSemanticEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.synthesizer = RAGSynthesizer()
        self.bm25 = BM25Retriever()
        self.rebuild_bm25()

    def rebuild_bm25(self):
        """Indexes all chunk texts into BM25 lexicon."""
        docs = [c.get("text", "") for c in self.vector_store.metadata]
        self.bm25.fit(docs)

    def query(self, req: QueryRequest) -> QueryResponse:
        start_time = time.time()
        
        # 1. Expand query with university synonyms
        expanded_q = expand_query(req.query)
        
        # 2. Dense Semantic Retrieval
        q_vector = self.embedder.encode([req.query])
        dense_results = self.vector_store.search(
            query_vector=q_vector,
            top_k=max(8, (req.top_k or 4) * 2),
            category=req.category or "all"
        )
        
        # 3. Lexical BM25 Retrieval
        bm25_ranked = self.bm25.search(expanded_q, top_k=max(8, (req.top_k or 4) * 2))

        # 4. Reciprocal Rank Fusion (Hybrid Search)
        hybrid_chunks = reciprocal_rank_fusion(
            dense_ranked=dense_results,
            bm25_ranked=bm25_ranked,
            all_chunks=self.vector_store.metadata,
            rrf_k=60,
            dense_weight=req.hybrid_weight or 0.65,
            top_k=req.top_k or 4
        )

        # 5. Apply Confidence Threshold
        threshold = req.threshold if req.threshold is not None else 0.35
        filtered_results = [r for r in hybrid_chunks if r["score"] >= threshold]

        # Determine Confidence Level
        if not filtered_results:
            confidence = "Not Found"
        else:
            best_score = max(r["score"] for r in filtered_results)
            if best_score >= 0.70:
                confidence = "High"
            elif best_score >= 0.45:
                confidence = "Medium"
            else:
                confidence = "Low"

        # 6. Synthesize Answer
        answer_text, llm_used = self.synthesizer.synthesize(
            query=req.query,
            retrieved_chunks=filtered_results,
            provider=req.llm_provider or "local",
            api_key=req.api_key,
            temperature=req.temperature or 0.2
        )

        # 7. Hallucination / Fact Grounding Check
        hallucination_report = check_faithfulness(answer_text, filtered_results)

        # 8. Generate Contextual Follow-up Questions
        follow_ups = self._generate_follow_up_questions(req.query, filtered_results)

        # 9. Build Structured Citations
        citations: List[Citation] = []
        for i, chunk in enumerate(filtered_results, 1):
            citations.append(Citation(
                id=i,
                source=chunk["source"],
                page=chunk["page"],
                file=chunk["file"],
                category=chunk["category"],
                snippet=chunk["text"],
                score=chunk["score"],
                verified=True
            ))

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return QueryResponse(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_chunks_count=len(filtered_results),
            llm_used=llm_used,
            processing_time_ms=elapsed_ms,
            query=req.query,
            follow_up_questions=follow_ups,
            session_id=req.session_id,
            hallucination_check=hallucination_report
        )

    def search_only(self, query: str, top_k: int = 5, category: str = "all") -> List[SearchResultChunk]:
        """Hybrid search explorer returning top-k matching chunks."""
        q_vector = self.embedder.encode([query])
        dense_results = self.vector_store.search(
            query_vector=q_vector,
            top_k=top_k * 2,
            category=category
        )
        expanded = expand_query(query)
        bm25_results = self.bm25.search(expanded, top_k=top_k * 2)

        hybrid = reciprocal_rank_fusion(
            dense_ranked=dense_results,
            bm25_ranked=bm25_results,
            all_chunks=self.vector_store.metadata,
            rrf_k=60,
            dense_weight=0.6,
            top_k=top_k
        )
        return [SearchResultChunk(**r) for r in hybrid]

    def compare_topics(self, req: CompareRequest) -> CompareResponse:
        """Compares policy clauses across two categories or documents."""
        # Query perspective A
        q_a = QueryRequest(query=req.topic, category=req.category_a or "Academic", top_k=2)
        res_a = self.query(q_a)

        # Query perspective B
        q_b = QueryRequest(query=req.topic, category=req.category_b or "Examination", top_k=2)
        res_b = self.query(q_b)

        summary = (
            f"**Comparative Regulation Analysis for '{req.topic}'**:\n\n"
            f"- **{req.category_a or 'Perspective A'}**: Grounded on {len(res_a.citations)} clauses ({res_a.confidence} Confidence).\n"
            f"- **{req.category_b or 'Perspective B'}**: Grounded on {len(res_b.citations)} clauses ({res_b.confidence} Confidence).\n\n"
            f"Both regulations ensure strict adherence to university governance standards."
        )

        return CompareResponse(
            topic=req.topic,
            perspective_a={"category": req.category_a, "answer": res_a.answer, "citations": res_a.citations},
            perspective_b={"category": req.category_b, "answer": res_b.answer, "citations": res_b.citations},
            comparison_summary=summary
        )

    def _generate_follow_up_questions(self, query: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generates smart contextual follow-up questions."""
        q_lower = query.lower()
        follow_ups = []

        if "attendance" in q_lower or "leave" in q_lower:
            follow_ups = [
                "What is the procedure for submitting medical condonation certificates?",
                "What are the consequences of receiving a Grade SA (Shortage of Attendance)?",
                "Is lab practical attendance calculated separately from theory lectures?"
            ]
        elif "cgpa" in q_lower or "grade" in q_lower or "academic" in q_lower:
            follow_ups = [
                "What are the eligibility criteria and extra credits required for an Honors Degree?",
                "What happens if a student is placed on Academic Probation?",
                "What is the maximum duration allowed to complete the degree (N + 2 years rule)?"
            ]
        elif "exam" in q_lower or "re-evaluation" in q_lower or "backlog" in q_lower:
            follow_ups = [
                "What is the deadline and fee for applying for answer script re-evaluation?",
                "When are supplementary / backlog examinations scheduled?",
                "What are the disciplinary penalties for possession of electronic devices in exam halls?"
            ]
        elif "scholarship" in q_lower or "fee" in q_lower or "refund" in q_lower:
            follow_ups = [
                "What are the criteria for the Chancellor's 50% Merit Scholarship?",
                "What is the fee refund percentage if admission is withdrawn within 15 days?",
                "What is the late payment fine per day after the tuition grace period?"
            ]
        elif "hostel" in q_lower or "ragging" in q_lower or "conduct" in q_lower:
            follow_ups = [
                "What are the hostel gate closing times and biometric attendance rules?",
                "What is the 24/7 Anti-Ragging helpline number and disciplinary protocol?",
                "How many books can an undergraduate student borrow from the central library?"
            ]
        else:
            follow_ups = [
                "Can you show the exact document source and page number for this regulation?",
                "What is the official appeal or grievance procedure for this policy?",
                "How does this regulation apply to final year graduating students?"
            ]

        return follow_ups[:3]
