"""Hybrid Search Engine combining Dense Vector Cosine Similarity and Lexical BM25 with Reciprocal Rank Fusion"""
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

# University colloquialism and synonym mapping for intelligent query expansion
UNIVERSITY_SYNONYMS = {
    "backlog": ["arrears", "supplementary examination", "failed course", "re-appearance", "Grade F"],
    "hall ticket": ["admit card", "examination pass", "SEE eligibility", "exam registration"],
    "attendance": ["minimum attendance", "aggregate attendance", "condonation", "shortage", "Grade SA"],
    "cgpa": ["cumulative grade point average", "SGPA", "grade points", "10-point scale", "grading system"],
    "honors": ["honours degree", "minor degree", "specialized credits", "interdisciplinary"],
    "probation": ["academic probation", "mentoring", "credit restriction", "minimum CGPA"],
    "ragging": ["anti-ragging", "harassment", "disciplinary committee", "zero tolerance", "expulsion"],
    "refund": ["fee refund", "tuition cancellation", "admission withdrawal", "deduction"],
    "curfew": ["hostel gate", "resident guidelines", "in-time", "biometric attendance"],
    "plagiarism": ["academic integrity", "similarity index", "thesis ethics", "turnitin"],
    "internship": ["industrial training", "summer internship", "credits", "placement"]
}

class BM25Retriever:
    """Fast in-memory BM25 retrieval for regulation documents."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.doc_tokens: List[List[str]] = []

    def fit(self, documents: List[str]):
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_tokens = []
        self.doc_lens = []
        total_len = 0
        df = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            
            # Count unique terms in document
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] += 1

        self.avgdl = total_len / max(1, self.corpus_size)
        self.doc_freqs = dict(df)

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # idf = log((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        clean = text.lower()
        return re.findall(r'\b[a-z0-9_]{2,}\b', clean)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, score) ranked descending."""
        if self.corpus_size == 0:
            return []

        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for idx, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(term)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[idx] += term_idf * (numerator / denominator)

        # Normalize BM25 scores to [0.0, 1.0]
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        normalized = [(i, scores[i] / max_score) for i in range(self.corpus_size)]
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized[:top_k]


def expand_query(query: str) -> str:
    """Expands student query with university policy synonyms."""
    expanded = query
    q_lower = query.lower()
    for key, syns in UNIVERSITY_SYNONYMS.items():
        if key in q_lower:
            extra = " ".join(syns[:3])
            expanded += f" {extra}"
    return expanded


def reciprocal_rank_fusion(
    dense_ranked: List[Dict[str, Any]],
    bm25_ranked: List[Tuple[int, float]],
    all_chunks: List[Dict[str, Any]],
    rrf_k: int = 60,
    dense_weight: float = 0.65,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Fuses dense semantic similarity rankings with BM25 lexical rankings.
    Computes grounded similarity score so out-of-scope queries naturally fall below threshold.
    """
    scores: Dict[int, float] = {}
    dense_score_map: Dict[int, float] = {}
    bm25_score_map: Dict[int, float] = {doc_idx: score for doc_idx, score in bm25_ranked}

    # 1. Dense rankings
    for rank, chunk_res in enumerate(dense_ranked, 1):
        idx = next((i for i, c in enumerate(all_chunks) if c["text"] == chunk_res["text"] and c["page"] == chunk_res["page"]), None)
        if idx is not None:
            dense_score_map[idx] = chunk_res["score"]
            rrf_rank_boost = (1.0 / (rrf_k + rank))
            scores[idx] = scores.get(idx, 0.0) + (chunk_res["score"] * dense_weight) + rrf_rank_boost

    # 2. BM25 rankings
    bm25_weight = 1.0 - dense_weight
    for rank, (doc_idx, bm25_score) in enumerate(bm25_ranked, 1):
        rrf_rank_boost = (1.0 / (rrf_k + rank))
        scores[doc_idx] = scores.get(doc_idx, 0.0) + (bm25_score * bm25_weight) + rrf_rank_boost

    # Sort combined
    sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for doc_idx, _ in sorted_indices[:top_k]:
        chunk = all_chunks[doc_idx]
        d_score = dense_score_map.get(doc_idx, 0.0)
        b_score = bm25_score_map.get(doc_idx, 0.0)
        
        # Grounded blended score in [0.0, 1.0]
        final_sim = (d_score * dense_weight) + (b_score * bm25_weight)
        
        results.append({
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "file": chunk.get("file", ""),
            "page": chunk.get("page", 1),
            "category": chunk.get("category", "General"),
            "url": chunk.get("url", ""),
            "score": round(final_sim, 4)
        })

    return results


def check_faithfulness(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Checks if numerical numbers, percentages, grade letters, and policy thresholds
    in the answer are grounded verbatim in the retrieved chunks.
    """
    if not chunks:
        return {"grounded": False, "score": 0.0, "unverified_facts": []}

    combined_context = " ".join(c["text"] for c in chunks).lower()
    
    # Extract numbers, percentages, grade letters like "75%", "5.0", "160 credits", "$25", "Grade A"
    facts_in_answer = re.findall(r'\b(?:\d+(?:\.\d+)?%?|\$\d+|Grade\s+[A-Z\+]+|\d+\s+credits|\d+\s+years|\d+\s+days)\b', answer)
    
    if not facts_in_answer:
        return {"grounded": True, "score": 1.0, "verified_facts": ["General contextual alignment"]}

    verified = []
    unverified = []

    for fact in facts_in_answer:
        clean_fact = fact.lower().strip()
        if clean_fact in combined_context:
            verified.append(fact)
        else:
            unverified.append(fact)

    grounded_score = len(verified) / max(1, len(facts_in_answer))
    return {
        "grounded": grounded_score >= 0.75,
        "score": round(grounded_score, 2),
        "verified_facts": list(set(verified)),
        "unverified_facts": list(set(unverified))
    }
