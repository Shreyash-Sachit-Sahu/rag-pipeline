"""
services/retriever/index_manager.py

Dynamically selects and manages the FAISS index type based on corpus size.
All thresholds and parameters are read from RetrieverSettings (env vars).

Index strategy (from PRD §2.1):
  < flat_max_chunks  → IndexFlatIP      (exact, always accurate)
  < ivf_max_chunks   → IndexIVFFlat     (nprobe-controlled ANN, ~10x faster)
  >= ivf_max_chunks  → IndexHNSWFlat    (graph ANN, best for very large corpora)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_retriever_settings

logger = logging.getLogger(__name__)
settings = get_retriever_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    score: float


class FAISSIndexManager:
    """
    Manages the FAISS index lifecycle:
    - load / create index from disk
    - select index type based on corpus size
    - search with automatic multi-hop support
    """

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._db_conn: Optional[sqlite3.Connection] = None
        self._n_vectors: int = 0
        self._index_type: str = "unloaded"

    # ── Lifecycle ────────────────────────────────────────────

    def load(self, index_path: str = settings.faiss_index_path,
             sqlite_path: str = settings.sqlite_path) -> None:
        """Load index from disk. Called once at service startup."""
        if not os.path.exists(index_path):
            logger.warning(f"FAISS index not found at {index_path}. Starting empty.")
            self._index = None
            self._n_vectors = 0
        else:
            self._index = faiss.read_index(index_path)
            self._n_vectors = self._index.ntotal
            self._index_type = self._detect_index_type()
            logger.info(
                f"Loaded FAISS index: {self._index_type}, "
                f"vectors={self._n_vectors}, path={index_path}"
            )

        if os.path.exists(sqlite_path):
            self._db_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            logger.info(f"Connected to metadata DB: {sqlite_path}")
        else:
            logger.warning(f"SQLite metadata not found at {sqlite_path}.")

    def save(self, index_path: str = settings.faiss_index_path) -> None:
        if self._index is not None:
            faiss.write_index(self._index, index_path)
            logger.info(f"FAISS index saved to {index_path}")

    # ── Index factory ────────────────────────────────────────

    @staticmethod
    def build_index(embeddings: np.ndarray) -> faiss.Index:
        """
        Build the appropriate FAISS index based on corpus size.
        All thresholds are read from environment variables.
        """
        n, dim = embeddings.shape
        flat_max = settings.flat_max_chunks
        ivf_max = settings.ivf_max_chunks

        if n < flat_max:
            logger.info(f"Corpus size {n} < {flat_max}: using IndexFlatIP (exact search)")
            index = faiss.IndexFlatIP(dim)

        elif n < ivf_max:
            logger.info(
                f"Corpus size {n} in [{flat_max}, {ivf_max}): "
                f"using IndexIVFFlat (nprobe={settings.ivf_nprobe})"
            )
            quantiser = faiss.IndexFlatIP(dim)
            n_lists = max(1, int(n ** 0.5))  # sqrt heuristic for IVF
            index = faiss.IndexIVFFlat(quantiser, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings.astype(np.float32))
            index.nprobe = settings.ivf_nprobe

        else:
            logger.info(
                f"Corpus size {n} >= {ivf_max}: "
                f"using IndexHNSWFlat (M={settings.hnsw_m}, "
                f"ef_search={settings.hnsw_ef_search})"
            )
            index = faiss.IndexHNSWFlat(dim, settings.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = settings.hnsw_ef_search

        # Normalised embeddings + inner product ≡ cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        return index

    def add_vectors(self, embeddings: np.ndarray, rebuild: bool = False) -> None:
        """
        Add new vectors to the index. If rebuild=True, recreates the index
        with the optimal type for the new corpus size.
        """
        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)

        if self._index is None or rebuild:
            self._index = self.build_index(vecs)
        else:
            if not self._index.is_trained:
                self._index.train(vecs)
            self._index.add(vecs)

        self._n_vectors = self._index.ntotal
        self._index_type = self._detect_index_type()

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        k: int,
    ) -> List[RetrievedChunk]:
        """Single-hop search. Returns k nearest chunks."""
        if self._index is None or self._n_vectors == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(k, self._n_vectors)
        scores, indices = self._index.search(vec, k)

        return self._indices_to_chunks(indices[0], scores[0])

    def search_multi_hop(
        self,
        query_embedding: List[float],
        enriched_embedding: List[float],
        k1: int = settings.multihop_k1,
        k2: int = settings.multihop_k2,
    ) -> List[RetrievedChunk]:
        """
        Two-round multi-hop retrieval (PRD §2.2):
          Round 1: retrieve k1 chunks with original query embedding
          Round 2: retrieve k2 new chunks with enriched embedding
          Merge + deduplicate by chunk_id, rank by max score
        """
        round1 = self.search(query_embedding, k1)
        round2 = self.search(enriched_embedding, k2)

        # Merge, deduplicate, keep highest score per chunk_id
        seen: dict[str, RetrievedChunk] = {}
        for chunk in round1 + round2:
            if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
                seen[chunk.chunk_id] = chunk

        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return merged

    # ── Helpers ──────────────────────────────────────────────

    def _indices_to_chunks(
        self, indices: np.ndarray, scores: np.ndarray
    ) -> List[RetrievedChunk]:
        chunks = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue
            meta = self._fetch_metadata(int(idx))
            if meta:
                chunk_id, text, source, page = meta
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        source=source,
                        page=page,
                        score=float(score),
                    )
                )
        return chunks

    def _fetch_metadata(
        self, faiss_idx: int
    ) -> Optional[Tuple[str, str, str, Optional[int]]]:
        if self._db_conn is None:
            return None
        cursor = self._db_conn.execute(
            "SELECT chunk_id, text, source, page FROM chunks WHERE faiss_idx = ?",
            (faiss_idx,),
        )
        row = cursor.fetchone()
        return tuple(row) if row else None

    def _detect_index_type(self) -> str:
        if self._index is None:
            return "none"
        t = type(self._index).__name__
        return t

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @property
    def index_type(self) -> str:
        return self._index_type
