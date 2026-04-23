"""
services/retriever/main.py

FastAPI service wrapping the FAISS retrieval engine.

Endpoints:
  POST /retrieve  → single-hop or multi-hop retrieval
  POST /add       → add vectors (ingestion pipeline calls this)
  GET  /health    → liveness probe with index stats
  GET  /metrics   → Prometheus metrics
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import spacy
import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram, make_asgi_app, CollectorRegistry
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_retriever_settings
from index_manager import FAISSIndexManager, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_retriever_settings()

# ── Prometheus metrics ───────────────────────────────────────
registry = CollectorRegistry()
retrieval_latency = Histogram(
    "rag_retrieval_latency_ms",
    "Retrieval latency in milliseconds",
    labelnames=["index_type", "hop_type"],
    buckets=[5, 10, 20, 50, 100, 200, 500, 1000],
    registry=registry,
)

# ── Global state ─────────────────────────────────────────────
_index_manager: FAISSIndexManager = FAISSIndexManager()
_nlp: Optional[spacy.Language] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _nlp
    logger.info("Loading FAISS index…")
    _index_manager.load()

    logger.info("Loading spaCy model for multi-hop entity extraction…")
    try:
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded.")
    except OSError:
        logger.warning("spaCy en_core_web_sm not found; multi-hop NER disabled.")
        _nlp = None

    yield
    logger.info("Retriever shutting down.")


app = FastAPI(
    title="RAG Retrieval Engine",
    version="1.0.0",
    description="FAISS-based dense retriever with adaptive index selection.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app(registry=registry))


# ── Schemas ──────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query_embedding: List[float] = Field(..., description="Query vector (normalised)")
    k: int = Field(settings.default_k, ge=1, le=50, description="Number of results")
    multi_hop: bool = Field(False, description="Enable two-round multi-hop retrieval")
    round1_texts: Optional[List[str]] = Field(
        None,
        description="Round-1 texts used for NER-enriched second round (multi-hop only)",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    score: float


class RetrieveResponse(BaseModel):
    chunks: List[ChunkResult]
    latency_ms: float
    index_type: str
    hop_type: str


class AddVectorsRequest(BaseModel):
    embeddings: List[List[float]]
    chunk_ids: List[str]
    texts: List[str]
    sources: List[str]
    pages: Optional[List[Optional[int]]] = None
    rebuild_index: bool = False


class HealthResponse(BaseModel):
    status: str
    n_vectors: int
    index_type: str


# ── Endpoints ────────────────────────────────────────────────

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    t0 = time.perf_counter()
    hop_type = "single"

    if request.multi_hop and _nlp is not None and request.round1_texts:
        # Build enriched embedding via NER on round-1 texts
        combined_text = " ".join(request.round1_texts)
        doc = _nlp(combined_text[:10_000])  # truncate for speed
        entities = [ent.text for ent in doc.ents]
        key_terms = list(dict.fromkeys(entities))[:10]  # unique, top-10

        if key_terms:
            # We can't re-embed here — ask gateway to do it.
            # For multi-hop we expect the enriched embedding in query_embedding.
            # This path handles the second-round search.
            hop_type = "multi_hop_round2"

        chunks = _index_manager.search(request.query_embedding, request.k)
    elif request.multi_hop:
        hop_type = "multi_hop"
        # Gateway handles full multi-hop flow; retriever just executes the search
        chunks = _index_manager.search(request.query_embedding, request.k)
    else:
        chunks = _index_manager.search(request.query_embedding, request.k)

    latency_ms = (time.perf_counter() - t0) * 1000
    retrieval_latency.labels(
        index_type=_index_manager.index_type, hop_type=hop_type
    ).observe(latency_ms)

    return RetrieveResponse(
        chunks=[
            ChunkResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                page=c.page,
                score=c.score,
            )
            for c in chunks
        ],
        latency_ms=round(latency_ms, 2),
        index_type=_index_manager.index_type,
        hop_type=hop_type,
    )


@app.post("/retrieve/multihop", response_model=RetrieveResponse)
async def retrieve_multihop(
    query_embedding: List[float],
    enriched_embedding: List[float],
    k1: int = settings.multihop_k1,
    k2: int = settings.multihop_k2,
) -> RetrieveResponse:
    """Full two-round multi-hop retrieval called by the gateway."""
    t0 = time.perf_counter()
    chunks = _index_manager.search_multi_hop(
        query_embedding, enriched_embedding, k1=k1, k2=k2
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    retrieval_latency.labels(
        index_type=_index_manager.index_type, hop_type="multi_hop"
    ).observe(latency_ms)

    return RetrieveResponse(
        chunks=[
            ChunkResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                page=c.page,
                score=c.score,
            )
            for c in chunks
        ],
        latency_ms=round(latency_ms, 2),
        index_type=_index_manager.index_type,
        hop_type="multi_hop",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        n_vectors=_index_manager.n_vectors,
        index_type=_index_manager.index_type,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.bind_host,
        port=settings.port,
        log_level="info",
    )