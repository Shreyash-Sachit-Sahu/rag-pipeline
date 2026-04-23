"""
services/gateway/main.py

The single entry point for all user queries.
Implements the full routing decision tree from PRD §5.1:

  Step 1: Check Redis semantic cache → if HIT, return immediately (< 5ms target)
  Step 2: Classify query complexity → simple | complex
  Step 3a (simple):  single-hop FAISS top-k → LLM with short context
  Step 3b (complex): multi-hop retrieve top-k → MMR re-rank → LLM with full context
  Step 4: Store result in Redis cache
  Step 5: Emit Prometheus metrics

All configuration (thresholds, URLs, targets) comes from GatewaySettings (env vars).
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app,CollectorRegistry
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_gateway_settings
from classifier import ComplexityClassifier
from mmr import mmr_rerank

# cache_client.py and llm_client.py are copied into /app/ by the Dockerfile
from cache_client import SemanticCache
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_gateway_settings()
registry = CollectorRegistry()

# ── Prometheus metrics (PRD §6.1) ────────────────────────────
query_latency = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency in seconds",
    labelnames=["route"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5],
    registry=registry,
)
cache_hit_counter = Counter(
    "rag_cache_hit_total",
    "Semantic cache hit/miss counter",
    labelnames=["result"],
    registry=registry,
)
llm_tokens_per_second = Gauge(
    "rag_llm_tokens_per_second",
    "LLM inference throughput (tokens/second)",
    registry=registry,
)

# ── Global service clients ────────────────────────────────────
_classifier: ComplexityClassifier
_cache: SemanticCache
_llm_client: LLMClient
_http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _cache, _llm_client, _http

    logger.info("Initialising gateway services…")
    _classifier = ComplexityClassifier()

    _cache = SemanticCache()
    from shared.config import get_embedder_settings
    dim = get_embedder_settings().embedding_dim
    _cache.connect(dim=dim)

    _llm_client = LLMClient()

    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
    )
    logger.info("Gateway ready.")
    yield

    await _llm_client.close()
    await _http.aclose()
    logger.info("Gateway shut down.")


app = FastAPI(
    title="RAG Query Gateway",
    version="1.0.0",
    description="Latency-aware RAG gateway with adaptive routing and semantic caching.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app())


# ── Schemas ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    model_config = {
        "json_schema_extra": {"example": {"query": "What is retrieval-augmented generation?"}}
    }


class QueryResponse(BaseModel):
    answer: str
    route: str           # "cache_hit" | "simple" | "complex"
    latency_ms: float
    complexity_score: float
    complexity_features: dict
    cache_hit: bool
    chunks_used: int
    llm_tokens_per_second: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    cache_connected: bool
    services: dict


# ── Helper: call embedder service ────────────────────────────

async def _embed(texts: List[str]) -> List[List[float]]:
    url = f"{settings.embedder_url}/embed"
    resp = await _http.post(url, json={"texts": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]


# ── Helper: call retriever service ───────────────────────────

async def _retrieve(
    embedding: List[float],
    k: int,
    multi_hop: bool = False,
    round1_texts: Optional[List[str]] = None,
) -> dict:
    url = f"{settings.retriever_url}/retrieve"
    payload = {
        "query_embedding": embedding,
        "k": k,
        "multi_hop": multi_hop,
        "round1_texts": round1_texts,
    }
    resp = await _http.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def _retrieve_multihop(
    query_embedding: List[float],
    enriched_embedding: List[float],
) -> dict:
    url = f"{settings.retriever_url}/retrieve/multihop"
    resp = await _http.post(
        url,
        params={
            "k1": settings.complex_k if hasattr(settings, "complex_k") else 8,
            "k2": 4,
        },
        json={
            "query_embedding": query_embedding,
            "enriched_embedding": enriched_embedding,
        },
    )
    resp.raise_for_status()
    return resp.json()


# ── Main query endpoint ───────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    t_start = time.perf_counter()

    # ── Step 1: Embed query ──────────────────────────────────
    try:
        embeddings = await _embed([request.query])
        q_vec = embeddings[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedder error: {exc}")

    # ── Step 2: Redis semantic cache check ───────────────────
    t_cache_start = time.perf_counter()
    cached_response = _cache.lookup(q_vec)
    cache_latency_ms = (time.perf_counter() - t_cache_start) * 1000

    if cached_response is not None:
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        cache_hit_counter.labels(result="hit").inc()
        query_latency.labels(route="cache_hit").observe(total_latency_ms / 1000)
        logger.info(f"Cache HIT: {total_latency_ms:.1f}ms")
        return QueryResponse(
            answer=cached_response,
            route="cache_hit",
            latency_ms=round(total_latency_ms, 2),
            complexity_score=0.0,
            complexity_features={},
            cache_hit=True,
            chunks_used=0,
        )

    cache_hit_counter.labels(result="miss").inc()

    # ── Step 3: Classify query complexity ───────────────────
    complexity = _classifier.score(request.query)
    logger.info(
        f"Query complexity: score={complexity.score:.3f}, "
        f"is_complex={complexity.is_complex}, "
        f"features={complexity.features}"
    )

    # ── Step 4: Retrieve chunks ──────────────────────────────
    try:
        if complexity.is_complex:
            # Multi-hop: round 1 → extract entities → re-embed → round 2
            round1_result = await _retrieve(q_vec, k=8, multi_hop=False)
            round1_texts = [c["text"] for c in round1_result["chunks"]]

            key_terms = _classifier.extract_key_terms(round1_texts)
            if key_terms:
                enriched_query = request.query + " " + " ".join(key_terms)
                enriched_embeddings = await _embed([enriched_query])
                enriched_vec = enriched_embeddings[0]
            else:
                enriched_vec = q_vec

            round2_result = await _retrieve(enriched_vec, k=4, multi_hop=True)

            # Merge and deduplicate by chunk_id
            all_chunks = {
                c["chunk_id"]: c for c in round1_result["chunks"] + round2_result["chunks"]
            }.values()
            chunks = sorted(all_chunks, key=lambda c: c["score"], reverse=True)[
                :settings.complex_k if hasattr(settings, "complex_k") else 12
            ]

            # MMR re-rank to reduce redundancy (PRD §5.2)
            chunk_list = list(chunks)
            if len(chunk_list) > 1:
                # We don't have stored embeddings per chunk here; use score as proxy
                # For full MMR with embeddings, the retriever would need to return them
                chunk_embeddings_proxy = [[c["score"]] + [0.0] * (len(q_vec) - 1)
                                           for c in chunk_list]
                chunk_list = mmr_rerank(
                    query_embedding=q_vec,
                    chunks=chunk_list,
                    embeddings=chunk_embeddings_proxy,
                    top_k=min(len(chunk_list), settings.complex_k if hasattr(settings, "complex_k") else 12),
                    mmr_lambda=settings.mmr_lambda,
                )
            route = "complex"
        else:
            result = await _retrieve(q_vec, k=4)
            chunk_list = result["chunks"]
            route = "simple"

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retriever error: {exc}")

    # ── Step 5: LLM generation ───────────────────────────────
    try:
        llm_result = await _llm_client.generate(
            question=request.query,
            chunks=chunk_list,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    answer = llm_result["answer"]
    if llm_result["tokens_per_second"]:
        llm_tokens_per_second.set(llm_result["tokens_per_second"])

    # ── Step 6: Store in cache ───────────────────────────────
    try:
        _cache.store(q_vec, answer, source=route)
    except Exception as exc:
        logger.warning(f"Cache store failed: {exc}")

    total_latency_ms = (time.perf_counter() - t_start) * 1000
    query_latency.labels(route=route).observe(total_latency_ms / 1000)

    logger.info(
        f"Query complete: route={route}, "
        f"chunks={len(chunk_list)}, "
        f"latency={total_latency_ms:.1f}ms"
    )

    return QueryResponse(
        answer=answer,
        route=route,
        latency_ms=round(total_latency_ms, 2),
        complexity_score=complexity.score,
        complexity_features=complexity.features,
        cache_hit=False,
        chunks_used=len(chunk_list),
        llm_tokens_per_second=llm_result.get("tokens_per_second"),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    services = {}

    # Check embedder
    try:
        r = await _http.get(f"{settings.embedder_url}/health", timeout=3.0)
        services["embedder"] = r.json().get("status", "unknown")
    except Exception:
        services["embedder"] = "unreachable"

    # Check retriever
    try:
        r = await _http.get(f"{settings.retriever_url}/health", timeout=3.0)
        services["retriever"] = r.json().get("status", "unknown")
    except Exception:
        services["retriever"] = "unreachable"

    # Check LLM
    services["llm"] = "ok" if await _llm_client.health() else "unreachable"

    return HealthResponse(
        status="ok",
        cache_connected=_cache.is_connected,
        services=services,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.bind_host,
        port=settings.port,
        log_level="info",
    )