"""
services/embedder/main.py

FastAPI service that serves sentence-transformer embeddings.
- POST /embed  → single or batch embedding
- GET  /health → liveness probe

All configuration is read from environment variables via EmbedderSettings.
No values are hard-coded.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_embedder_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_embedder_settings()

# ── Global model state (loaded once at startup) ──────────────
_model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info(f"Loading embedding model: {settings.model_name}")
    _model = SentenceTransformer(settings.model_name)
    logger.info(
        f"Model loaded. Embedding dim={settings.embedding_dim}, "
        f"device={_model.device}"
    )
    yield
    _model = None
    logger.info("Embedder shut down.")


# Use a fresh registry so restarts don't hit duplicate-registration errors
REGISTRY = CollectorRegistry(auto_describe=True)
embed_latency = Histogram(
    "rag_embed_latency_ms",
    "Embedding latency in milliseconds",
    labelnames=["batch_size"],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
    registry=REGISTRY,
)

app = FastAPI(
    title="RAG Embedding Service",
    version="1.0.0",
    description="Local sentence-transformer embedding server. No external API calls.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app(registry=REGISTRY))


# ── Request / Response schemas ───────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Texts to embed")

    model_config = {"json_schema_extra": {"example": {"texts": ["What is RAG?"]}}}


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_dim: int


# ── Endpoints ────────────────────────────────────────────────

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(request.texts) > settings.batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.texts)} exceeds maximum {settings.batch_size}",
        )

    t0 = time.perf_counter()
    embeddings = _model.encode(
        request.texts,
        batch_size=min(len(request.texts), 64),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Emit latency warning if targets are breached
    if len(request.texts) == 1 and latency_ms > settings.single_latency_target_ms:
        logger.warning(
            f"Single-text embedding latency {latency_ms:.1f}ms "
            f"exceeded target {settings.single_latency_target_ms}ms"
        )
    elif len(request.texts) >= 64 and latency_ms > settings.batch_latency_target_ms:
        logger.warning(
            f"Batch embedding latency {latency_ms:.1f}ms "
            f"exceeded target {settings.batch_latency_target_ms}ms"
        )

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dim=embeddings.shape[1],
        latency_ms=round(latency_ms, 2),
        model=settings.model_name,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model=settings.model_name,
        embedding_dim=settings.embedding_dim,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.bind_host,
        port=settings.port,
        log_level="info",
        reload=False,
    )