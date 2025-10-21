"""LLM service using Groq API for generating answers."""

from __future__ import annotations

import logging
from typing import Sequence

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self) -> None:
        self.client = None
        if settings.groq_api_key and OpenAI is not None:
            try:
                self.client = OpenAI(
                    api_key=settings.groq_api_key,
                    base_url=settings.api_base_url
                )
                logger.info(f"Groq LLM client initialized: {settings.llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            if not settings.groq_api_key:
                logger.warning("GROQ_API_KEY not set. Configure it in .env file or Streamlit secrets.")
            if OpenAI is None:
                logger.warning("openai package not installed. Install it with: pip install openai")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_chunks(self, chunks: Sequence[str], max_tokens: int = 2500) -> list[str]:
        truncated = []
        total_tokens = 0
        max_chunk_length = 400
        max_chunks = 4
        
        for i, chunk in enumerate(chunks):
            if i >= max_chunks or total_tokens >= max_tokens:
                break
            
            chunk_text = str(chunk)
            if len(chunk_text) > max_chunk_length:
                chunk_text = chunk_text[:max_chunk_length] + "..."
            
            chunk_tokens = self._estimate_tokens(chunk_text)
            if total_tokens + chunk_tokens > max_tokens:
                remaining = max_tokens - total_tokens
                if remaining > 100:
                    chunk_text = chunk_text[:remaining * 4]
                    truncated.append(chunk_text)
                break
            
            truncated.append(chunk_text)
            total_tokens += chunk_tokens
        
        return truncated

    def generate_answer(self, query: str, context_chunks: Sequence[str]) -> str:
        if not context_chunks:
            return "No relevant information found in the documents."

        if self.client:
            try:
                truncated_chunks = self._truncate_chunks(context_chunks, max_tokens=3000)
                
                prompt = (
                    "Answer the question using only the provided context. "
                    "If the answer isn't in the context, say you don't know.\n\n"
                    "Context:\n" + "\n---\n".join(truncated_chunks) + f"\n\nQuestion: {query}"
                )
                
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful document assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Groq API error: {e}")
                if '429' in error_str or 'quota' in error_str or 'rate_limit' in error_str or '413' in error_str:
                    if '413' in error_str or 'too large' in error_str:
                        return "Request too large. Try a more specific question or wait a moment."
                    return "Rate limit exceeded. Wait a few seconds and try again."
                elif '401' in error_str or 'unauthorized' in error_str:
                    return "Authentication failed. Check your GROQ_API_KEY in .env or Streamlit secrets"
                else:
                    return f"API error: {str(e)}"

        query_lower = query.lower()
        query_words = {w.strip('.,!?;:') for w in query_lower.split() if len(w) > 2}
        
        scored = []
        for chunk in context_chunks:
            chunk_text = str(chunk)
            chunk_lower = chunk_text.lower()
            matches = sum(1 for word in query_words if word in chunk_lower)
            early_bonus = 2.0 if any(word in chunk_lower[:200] for word in query_words) else 1.0
            scored.append((matches * early_bonus, chunk_text))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored[:3] if score > 0] or context_chunks[:1]
        
        combined = " ".join(top_chunks)
        sentences = combined.split('. ')
        relevant = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if any(word in sent_lower for word in query_words):
                relevant.append(sent.strip())
            elif len(relevant) < 2:
                relevant.append(sent.strip())
            if len(relevant) >= 4:
                break
        
        if not relevant:
            relevant = sentences[:3]
        
        summary = '. '.join(relevant)
        if not summary.endswith('.'):
            summary += '.'
        
        return f"{summary}\n\n(Offline mode - set GROQ_API_KEY for AI answers)"
