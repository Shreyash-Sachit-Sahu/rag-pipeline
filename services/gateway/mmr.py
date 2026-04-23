"""
services/gateway/mmr.py

Maximal Marginal Relevance (MMR) re-ranking (PRD §5.2).

Selects a diverse subset of retrieved chunks by balancing:
  - Relevance: cosine similarity to the query
  - Diversity: dissimilarity to already-selected chunks

MMR score = lambda * sim(chunk, query) - (1 - lambda) * max(sim(chunk, selected))

lambda (MMR_LAMBDA env var, default 0.7) controls the relevance/diversity trade-off.
Higher lambda → more relevant but potentially redundant.
Lower lambda  → more diverse but possibly less relevant.
"""
from __future__ import annotations

import math
import os
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_gateway_settings

settings = get_gateway_settings()


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (assumes they may not be normalised)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def mmr_rerank(
    query_embedding: List[float],
    chunks: List[dict],
    embeddings: List[List[float]],
    top_k: int,
    mmr_lambda: float = settings.mmr_lambda,
) -> List[dict]:
    """
    Re-rank `chunks` using MMR to reduce redundancy.

    Args:
        query_embedding: the query vector
        chunks:          list of chunk dicts (must have same ordering as embeddings)
        embeddings:      list of chunk embedding vectors
        top_k:           number of chunks to select
        mmr_lambda:      relevance/diversity trade-off (from env var MMR_LAMBDA)

    Returns:
        Ordered list of selected chunk dicts, length <= top_k
    """
    if not chunks:
        return []

    top_k = min(top_k, len(chunks))
    remaining = list(range(len(chunks)))
    selected_indices: List[int] = []

    # Pre-compute query similarities
    query_sims = [_cosine(query_embedding, emb) for emb in embeddings]

    while len(selected_indices) < top_k and remaining:
        if not selected_indices:
            # First selection: pick highest query similarity
            best_idx = max(remaining, key=lambda i: query_sims[i])
        else:
            best_score = float("-inf")
            best_idx = remaining[0]

            for i in remaining:
                # Max similarity to any already-selected chunk
                max_selected_sim = max(
                    _cosine(embeddings[i], embeddings[s])
                    for s in selected_indices
                )
                mmr_score = (
                    mmr_lambda * query_sims[i]
                    - (1.0 - mmr_lambda) * max_selected_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected_indices]
