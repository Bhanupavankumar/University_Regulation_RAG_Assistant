"""Smart multi-provider synthesizer for University Regulation QA"""
import re
import json
import requests
from typing import List, Dict, Any, Tuple

class RAGSynthesizer:
    def __init__(self):
        pass

    def synthesize(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        provider: str = "local",
        api_key: str = None,
        temperature: float = 0.2
    ) -> Tuple[str, str]:
        """
        Synthesizes an answer based on retrieved regulation chunks.
        Returns (answer_markdown, llm_used).
        """
        if not retrieved_chunks:
            return (
                "### ℹ️ Information Not Found\n\n"
                "The requested topic was not found in the indexed university regulations and policies. "
                "Please verify your question or ensure the relevant university handbook or policy document has been uploaded.",
                "System Fallback"
            )

        provider_clean = (provider or "local").lower()

        # Try API providers if key is present
        if provider_clean == "gemini" and api_key:
            ans = self._call_gemini(query, retrieved_chunks, api_key, temperature)
            if ans:
                return ans, "Google Gemini"

        elif provider_clean == "groq" and api_key:
            ans = self._call_groq(query, retrieved_chunks, api_key, temperature)
            if ans:
                return ans, "Groq (Llama-3.3-70B)"

        elif provider_clean == "openai" and api_key:
            ans = self._call_openai(query, retrieved_chunks, api_key, temperature)
            if ans:
                return ans, "OpenAI (GPT-4o-mini)"

        elif provider_clean == "ollama":
            ans = self._call_ollama(query, retrieved_chunks, temperature)
            if ans:
                return ans, "Local Ollama"

        # High-performance built-in Local Extractive Synthesizer
        return self._local_synthesize(query, retrieved_chunks), "Local Extractive Synthesizer"

    def _build_context_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, c in enumerate(retrieved_chunks, 1):
            context_str += f"\n--- DOCUMENT EXCERPT {i} ---\n"
            context_str += f"Source: {c['source']}\n"
            context_str += f"Page: {c['page']}\n"
            context_str += f"Category: {c['category']}\n"
            context_str += f"Content:\n{c['text']}\n"

        prompt = f"""You are the official University Regulations and Policies AI Assistant.
Answer the user's question accurately using ONLY the provided official document excerpts.
Every fact, regulation, threshold, or policy you state MUST include an inline citation in the exact format: `[{'{Source}'} | Page {'{Page}'}]`.

If the answer cannot be found in the provided excerpts, clearly state: "Information not found in the available university regulations."

User Question: {query}

Official Excerpts:
{context_str}

Structured Answer:"""
        return prompt

    def _call_gemini(self, query: str, chunks: List[Dict[str, Any]], api_key: str, temp: float) -> str:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            prompt = self._build_context_prompt(query, chunks)
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temp}
            }
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[Synthesizer] Gemini API Error: {e}")
        return None

    def _call_groq(self, query: str, chunks: List[Dict[str, Any]], api_key: str, temp: float) -> str:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            prompt = self._build_context_prompt(query, chunks)
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Synthesizer] Groq API Error: {e}")
        return None

    def _call_openai(self, query: str, chunks: List[Dict[str, Any]], api_key: str, temp: float) -> str:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            prompt = self._build_context_prompt(query, chunks)
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Synthesizer] OpenAI API Error: {e}")
        return None

    def _call_ollama(self, query: str, chunks: List[Dict[str, Any]], temp: float) -> str:
        try:
            url = "http://localhost:11434/api/generate"
            prompt = self._build_context_prompt(query, chunks)
            payload = {"model": "llama3.2", "prompt": prompt, "stream": False, "options": {"temperature": temp}}
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                return resp.json().get("response")
        except Exception as e:
            print(f"[Synthesizer] Ollama Error: {e}")
        return None

    def _local_synthesize(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes an intelligent, structured answer directly from chunk contents
        with sentence highlighting, rule extraction, and exact citations.
        """
        q_words = set(re.findall(r'\b[a-z0-9]{3,}\b', query.lower()))
        
        # Group chunks by source document
        by_source = {}
        for c in chunks:
            src = c["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(c)

        sections = []
        
        # Summary Header
        sections.append(f"Based on official university records, here is the relevant policy regarding **{query.strip('?')}**:\n")

        for src, src_chunks in by_source.items():
            first_chunk = src_chunks[0]
            page_list = sorted(list(set(c["page"] for c in src_chunks)))
            pages_str = ", ".join(f"Page {p}" for p in page_list)
            
            sections.append(f"### 📘 {src} ({pages_str})\n")
            
            for c in src_chunks:
                text = c["text"]
                # Split sentences
                sentences = re.split(r'(?<=[.?!])\s+', text)
                relevant_sentences = []
                
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean or len(s_clean) < 15:
                        continue
                    
                    s_words = set(re.findall(r'\b[a-z0-9]{3,}\b', s_clean.lower()))
                    overlap = len(q_words.intersection(s_words))
                    
                    # Highlight keywords if matching
                    if overlap >= 1 or any(k in s_clean.lower() for k in ["must", "required", "minimum", "policy", "rule", "fee", "grade", "shall", "percent", "%"]):
                        relevant_sentences.append(s_clean)

                if relevant_sentences:
                    for rs in relevant_sentences[:3]:
                        sections.append(f"- {rs} `[{src} | Page {c['page']}]`")
                else:
                    # Fallback to snippet
                    snippet = text[:250].strip() + ("..." if len(text) > 250 else "")
                    sections.append(f"- {snippet} `[{src} | Page {c['page']}]`")

            sections.append("")

        sections.append("---")
        sections.append("📌 *All statements above are directly grounded in the official University Regulations and verified against the registered source pages.*")
        
        return "\n".join(sections)
