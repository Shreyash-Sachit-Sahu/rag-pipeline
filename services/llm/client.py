"""
services/llm/client.py

Thin wrapper around the local llama.cpp HTTP server.
Formats retrieved chunks into a prompt context window and returns the LLM response.
No external API calls — all inference is local.

Prompt template follows PRD §4.2.
All parameters (temperature, max_tokens, etc.) come from LLMSettings (env vars).
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import httpx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_llm_settings

logger = logging.getLogger(__name__)
settings = get_llm_settings()

# ── Prompt template (PRD §4.2) ───────────────────────────────
_SYSTEM_PROMPT = (
    "You are a precise assistant. "
    "Answer using ONLY the context below. "
    "If the answer is not in the context, say 'I don't know.'"
)

_PROMPT_TEMPLATE = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:"
)


def _build_context(chunks: List[dict], max_chars: Optional[int] = None) -> str:
    """
    Join chunk texts into a context string.
    Optionally truncate to max_chars to stay within context window.
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        segment = f"[{i}] ({source})\n{text}"
        if max_chars and total + len(segment) > max_chars:
            break
        parts.append(segment)
        total += len(segment)
    return "\n\n".join(parts)


class LLMClient:
    """
    HTTP client for the llama.cpp OpenAI-compatible server.
    """

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(120.0),
        )

    async def generate(
        self,
        question: str,
        chunks: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Generate an answer from the local LLM.

        Returns:
            {
                "answer": str,
                "latency_ms": float,
                "tokens_generated": int,
                "tokens_per_second": float,
            }
        """
        context = _build_context(
            chunks,
            max_chars=settings.context_size * 4,  # rough char → token ratio
        )

        user_content = _PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature if temperature is not None else settings.temperature,
            "stream": settings.stream,
        }

        t0 = time.perf_counter()
        response = await self._http.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        data = response.json()
        answer_text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        tokens_generated = usage.get("completion_tokens", 0)
        tokens_per_second = (
            tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0.0
        )

        logger.info(
            f"LLM response: {tokens_generated} tokens, "
            f"{latency_ms:.0f}ms, {tokens_per_second:.1f} tok/s"
        )

        return {
            "answer": answer_text,
            "latency_ms": round(latency_ms, 2),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_second, 2),
        }

    async def health(self) -> bool:
        """Check if the llama.cpp server is responsive."""
        try:
            r = await self._http.get("/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
