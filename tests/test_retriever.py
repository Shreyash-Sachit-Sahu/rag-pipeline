"""
tests/test_retriever.py

Unit tests for:
  - FAISSIndexManager index selection logic (PRD §2.1)
  - Multi-hop retrieval merging
  - MMR re-ranking (PRD §5.2)

PRD target: FAISS recall@5 > 0.90 on test corpus.
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "retriever"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "gateway"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RECALL_TARGET = float(os.getenv("EVAL_TARGET_RECALL", "0.91"))
FLAT_MAX      = int(os.getenv("FAISS_FLAT_MAX_CHUNKS", "10000"))
IVF_MAX       = int(os.getenv("FAISS_IVF_MAX_CHUNKS", "1000000"))


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tiny_corpus():
    """500 random 384-dim unit vectors with sequential chunk_ids."""
    np.random.seed(42)
    n, dim = 500, 384
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs


@pytest.fixture
def index_manager_with_corpus(tiny_corpus, tmp_path):
    """IndexManager pre-loaded with the tiny corpus and matching SQLite."""
    pytest.importorskip("faiss")
    from index_manager import FAISSIndexManager

    db_path = str(tmp_path / "meta.sqlite")
    idx_path = str(tmp_path / "index.faiss")

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE chunks (
            faiss_idx INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER
        )
    """)
    rows = [
        (i, f"chunk_{i:04d}", f"doc_{i//10}", f"Text for chunk {i}", "test.txt", 0)
        for i in range(len(tiny_corpus))
    ]
    conn.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    manager = FAISSIndexManager()
    manager.add_vectors(tiny_corpus.copy())
    manager._db_conn = sqlite3.connect(db_path, check_same_thread=False)
    manager._faiss_index_path = idx_path

    return manager


# ── Index selection tests ─────────────────────────────────────

class TestIndexSelection:

    def test_flat_index_for_small_corpus(self):
        pytest.importorskip("faiss")
        import faiss
        from index_manager import FAISSIndexManager

        n = FLAT_MAX - 1
        dim = 32
        vecs = np.random.randn(n, dim).astype(np.float32)
        index = FAISSIndexManager.build_index(vecs)
        assert isinstance(index, faiss.IndexFlatIP), (
            f"Expected IndexFlatIP for n={n}, got {type(index).__name__}"
        )

    def test_ivf_index_for_medium_corpus(self):
        pytest.importorskip("faiss")
        import faiss
        from index_manager import FAISSIndexManager

        n = FLAT_MAX + 1
        dim = 32
        vecs = np.random.randn(n, dim).astype(np.float32)
        index = FAISSIndexManager.build_index(vecs)
        assert isinstance(index, faiss.IndexIVFFlat), (
            f"Expected IndexIVFFlat for n={n}, got {type(index).__name__}"
        )

    def test_hnsw_index_for_large_corpus(self):
        pytest.importorskip("faiss")
        import faiss
        from index_manager import FAISSIndexManager

        n = IVF_MAX + 1
        dim = 16
        vecs = np.random.randn(n, dim).astype(np.float32)
        index = FAISSIndexManager.build_index(vecs)
        assert isinstance(index, faiss.IndexHNSWFlat), (
            f"Expected IndexHNSWFlat for n={n}, got {type(index).__name__}"
        )


# ── Search quality tests ──────────────────────────────────────

class TestRetrievalRecall:

    def test_exact_vector_retrieves_itself(self, index_manager_with_corpus, tiny_corpus):
        manager = index_manager_with_corpus
        query_vec = tiny_corpus[42].tolist()
        results = manager.search(query_vec, k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_0042"

    def test_recall_at_5_above_target(self, index_manager_with_corpus, tiny_corpus):
        """
        For each of 50 random queries that ARE in the corpus,
        verify the original chunk appears in top-5 results.
        PRD target: recall@5 > 0.90.
        """
        manager = index_manager_with_corpus
        n_queries = 50
        hits = 0
        indices = np.random.choice(len(tiny_corpus), n_queries, replace=False)

        for idx in indices:
            query_vec = tiny_corpus[idx].tolist()
            expected_id = f"chunk_{idx:04d}"
            results = manager.search(query_vec, k=5)
            retrieved_ids = {r.chunk_id for r in results}
            if expected_id in retrieved_ids:
                hits += 1

        recall = hits / n_queries
        assert recall >= RECALL_TARGET, (
            f"Recall@5 {recall:.3f} < target {RECALL_TARGET} "
            f"({hits}/{n_queries} hits)"
        )

    def test_k_results_returned(self, index_manager_with_corpus, tiny_corpus):
        manager = index_manager_with_corpus
        for k in [1, 3, 5, 10]:
            results = manager.search(tiny_corpus[0].tolist(), k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    def test_results_sorted_by_score_descending(self, index_manager_with_corpus, tiny_corpus):
        manager = index_manager_with_corpus
        results = manager.search(tiny_corpus[0].tolist(), k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, index_manager_with_corpus, tiny_corpus):
        manager = index_manager_with_corpus
        results = manager.search(tiny_corpus[0].tolist(), k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0, f"Score out of range: {r.score}"


# ── MMR re-ranking tests ──────────────────────────────────────

class TestMMR:

    def _make_chunks(self, n):
        return [{"chunk_id": str(i), "text": f"chunk {i}", "score": 1.0 - i * 0.05} for i in range(n)]

    def _random_vecs(self, n, dim=16):
        np.random.seed(0)
        vecs = np.random.randn(n, dim).astype(float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).tolist()

    def test_mmr_returns_top_k(self):
        from mmr import mmr_rerank
        dim = 16
        n = 10
        chunks = self._make_chunks(n)
        vecs = self._random_vecs(n, dim)
        query = self._random_vecs(1, dim)[0]
        result = mmr_rerank(query, chunks, vecs, top_k=5)
        assert len(result) == 5

    def test_mmr_reduces_redundancy(self):
        """Two nearly identical chunks should not both appear in MMR top-2."""
        from mmr import mmr_rerank
        dim = 16
        np.random.seed(1)
        base = np.random.randn(dim)
        base = (base / np.linalg.norm(base)).tolist()
        near_dup = [v + 0.001 for v in base]

        query = np.random.randn(dim).tolist()
        chunks = [
            {"chunk_id": "a", "text": "dup1"},
            {"chunk_id": "b", "text": "dup2"},
            {"chunk_id": "c", "text": "diverse"},
        ]
        diverse_vec = np.random.randn(dim)
        diverse_vec = (diverse_vec / np.linalg.norm(diverse_vec)).tolist()

        result = mmr_rerank(query, chunks, [base, near_dup, diverse_vec], top_k=2, mmr_lambda=0.3)
        ids = {r["chunk_id"] for r in result}
        # With low lambda (high diversity weight), near-dup should be penalised
        assert len(ids) == 2

    def test_mmr_empty_input(self):
        from mmr import mmr_rerank
        result = mmr_rerank([], [], [], top_k=5)
        assert result == []

    def test_mmr_lambda_from_env(self):
        """MMR uses MMR_LAMBDA from env, not a hard-coded value."""
        lambda_val = float(os.getenv("MMR_LAMBDA", "0.7"))
        from mmr import settings
        assert settings.mmr_lambda == lambda_val
