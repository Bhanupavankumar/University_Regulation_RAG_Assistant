"""Analytics and Telemetry tracker for RAG System"""
import time
from collections import Counter
from typing import List, Dict, Any
from app.models import AnalyticsSummary

class AnalyticsTracker:
    def __init__(self):
        self.query_logs: List[Dict[str, Any]] = []
        self.feedback_counts = Counter()
        self.category_counts = Counter()
        self.confidence_counts = Counter({"High": 0, "Medium": 0, "Low": 0, "Not Found": 0})

    def log_query(
        self,
        query: str,
        category: str,
        confidence: str,
        latency_ms: float,
        chunks_count: int,
        llm_used: str
    ):
        self.confidence_counts[confidence] += 1
        self.category_counts[category] += 1
        
        log_entry = {
            "query": query,
            "category": category,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "chunks_count": chunks_count,
            "llm_used": llm_used,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.query_logs.append(log_entry)
        if len(self.query_logs) > 500:
            self.query_logs.pop(0)

    def log_feedback(self, feedback: str):
        if feedback in ["like", "dislike"]:
            self.feedback_counts[feedback] += 1

    def get_summary(self) -> AnalyticsSummary:
        total_queries = len(self.query_logs)
        avg_latency = (
            sum(q["latency_ms"] for q in self.query_logs) / max(1, total_queries)
            if total_queries > 0 else 0.0
        )

        return AnalyticsSummary(
            total_queries=total_queries,
            average_latency_ms=round(avg_latency, 2),
            confidence_breakdown=dict(self.confidence_counts),
            top_queried_categories=dict(self.category_counts),
            feedback_stats=dict(self.feedback_counts),
            recent_activity=list(reversed(self.query_logs[-10:]))
        )
