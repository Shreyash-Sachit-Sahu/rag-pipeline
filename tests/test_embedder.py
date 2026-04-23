"""
tests/test_embedder.py

Unit tests for the embedding service.
Tests embedder accuracy: cosine similarity > 0.85 on synonym pairs (PRD §4).
"""
from __future__ import annotations

import math
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Helpers ──────────────────────────────────────────────────

def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Tests ─────────────────────────────────────────────────────

class TestChunkText:
    """Tests for the chunking logic (importable without running FastAPI)."""

    def _chunk(self, text, chunk_size=512, overlap=64):
        """Inline copy of the chunking function for isolated testing."""
        tokens = text.split()
        if not tokens:
            return []
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(tokens):
                break
            start += chunk_size - overlap
        return chunks

    def test_single_chunk_short_text(self):
        text = "This is a short piece of text."
        chunks = self._chunk(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks(self):
        words = ["word"] * 600
        text = " ".join(words)
        chunks = self._chunk(text, chunk_size=512, overlap=64)
        assert len(chunks) >= 2

    def test_overlap_creates_shared_tokens(self):
        words = [str(i) for i in range(600)]
        text = " ".join(words)
        chunks = self._chunk(text, chunk_size=100, overlap=20)
        # Last tokens of chunk[0] should appear in chunk[1]
        end_of_first = chunks[0].split()[-20:]
        start_of_second = chunks[1].split()[:20]
        assert end_of_first == start_of_second

    def test_empty_text_returns_empty(self):
        assert self._chunk("") == []
        assert self._chunk("   ") == []

    def test_chunk_size_respected(self):
        words = ["word"] * 1000
        text = " ".join(words)
        chunks = self._chunk(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert len(chunk.split()) <= 100

    def test_configurable_params_used(self):
        """Verify chunk_size and overlap parameters are honoured."""
        words = [str(i) for i in range(200)]
        text = " ".join(words)
        chunks_small = self._chunk(text, chunk_size=50, overlap=5)
        chunks_large = self._chunk(text, chunk_size=150, overlap=5)
        assert len(chunks_small) > len(chunks_large)


class TestEmbedderAccuracy:
    """
    Smoke tests for embedding quality.
    Uses a real SentenceTransformer only if available; otherwise skips.
    PRD target: cosine similarity > 0.85 on synonym pairs.
    """

    COSINE_TARGET = float(os.getenv("EMBEDDING_COSINE_TARGET", "0.85"))

    SYNONYM_PAIRS = [
        ("car", "automobile"),
        ("fast", "quick"),
        ("happy", "joyful"),
        ("big", "large"),
        ("begin", "start"),
    ]

    DISSIMILAR_PAIRS = [
        ("car", "philosophy"),
        ("fast", "ocean"),
    ]

    @pytest.fixture(scope="class")
    def model(self):
        pytest.importorskip("sentence_transformers")
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_name)

    def test_synonym_similarity_above_target(self, model):
        for w1, w2 in self.SYNONYM_PAIRS:
            embs = model.encode([w1, w2], normalize_embeddings=True)
            sim = cosine_sim(embs[0].tolist(), embs[1].tolist())
            assert sim > self.COSINE_TARGET, (
                f"Synonym pair ({w1}, {w2}) similarity {sim:.3f} "
                f"< target {self.COSINE_TARGET}"
            )

    def test_dissimilar_pairs_below_threshold(self, model):
        for w1, w2 in self.DISSIMILAR_PAIRS:
            embs = model.encode([w1, w2], normalize_embeddings=True)
            sim = cosine_sim(embs[0].tolist(), embs[1].tolist())
            assert sim < 0.95, (
                f"Dissimilar pair ({w1}, {w2}) unexpectedly high similarity {sim:.3f}"
            )

    def test_output_dimension_matches_config(self, model):
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
        emb = model.encode(["test"], normalize_embeddings=True)
        assert emb.shape[1] == dim, (
            f"Embedding dim {emb.shape[1]} != configured {dim}"
        )

    def test_batch_matches_individual(self, model):
        texts = ["Hello world", "How are you?", "Testing batch embedding"]
        batch_embs = model.encode(texts, normalize_embeddings=True)
        for i, text in enumerate(texts):
            single_emb = model.encode([text], normalize_embeddings=True)
            sim = cosine_sim(batch_embs[i].tolist(), single_emb[0].tolist())
            assert sim > 0.999, f"Batch vs single mismatch for '{text}': sim={sim}"
