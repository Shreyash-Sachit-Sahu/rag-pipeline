"""
services/cache/cache_client.py

Redis Stack semantic cache using RediSearch vector similarity.

Flow (from PRD §3.1):
  1. Embed query → q_vec
  2. KNN search in Redis vector index
  3. If cosine similarity > CACHE_SIMILARITY_THRESHOLD → cache HIT
  4. Else → cache MISS; after LLM response, store {q_vec, response} with TTL

All thresholds and connection params come from CacheSettings (env vars).
"""
from __future__ import annotations

import json
import logging
import os
import struct
import uuid
from typing import Optional

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_cache_settings

logger = logging.getLogger(__name__)
settings = get_cache_settings()


def _vec_to_bytes(vec: list[float]) -> bytes:
    """Encode float list as little-endian float32 bytes for Redis."""
    return struct.pack(f"{len(vec)}f", *vec)


class SemanticCache:
    """
    Redis Stack semantic cache.
    Thread-safe — one instance per process.
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._dim: int = 0

    def connect(self, dim: int) -> None:
        """Connect to Redis and ensure the vector index exists."""
        self._dim = dim
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=False,
        )
        self._ensure_index()
        logger.info(
            f"Connected to Redis at {settings.redis_host}:{settings.redis_port}. "
            f"Vector dim={dim}, threshold={settings.similarity_threshold}"
        )

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it doesn't exist."""
        try:
            self._client.ft(settings.index_name).info()
            logger.info(f"Redis index '{settings.index_name}' already exists.")
        except Exception:
            logger.info(f"Creating Redis index '{settings.index_name}'…")
            schema = (
                TagField("$.source", as_name="source"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "BLOCK_SIZE": 1000,
                    },
                    as_name="embedding",
                ),
            )
            self._client.ft(settings.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[settings.key_prefix], index_type=IndexType.JSON
                ),
            )
            logger.info("Redis index created.")

    # ── Public API ───────────────────────────────────────────

    def lookup(self, query_embedding: list[float]) -> Optional[str]:
        """
        Return cached response string if similarity > threshold, else None.
        Uses FT.SEARCH KNN-1 to find the nearest stored embedding.
        """
        if self._client is None:
            raise RuntimeError("SemanticCache.connect() must be called first.")

        vec_bytes = _vec_to_bytes(query_embedding)

        q = (
            Query(f"*=>[KNN 1 @embedding $vec AS score]")
            .return_fields("score", "$.response")
            .sort_by("score")
            .dialect(2)
        )

        try:
            results = self._client.ft(settings.index_name).search(
                q, query_params={"vec": vec_bytes}
            )
        except Exception as exc:
            logger.error(f"Redis search error: {exc}")
            return None

        if not results.docs:
            return None

        doc = results.docs[0]
        # Redis COSINE distance: 0 = identical, 1 = orthogonal
        # Convert to similarity: similarity = 1 - distance
        distance = float(getattr(doc, "score", 1.0))
        similarity = 1.0 - distance

        if similarity >= settings.similarity_threshold:
            response_raw = getattr(doc, "$.response", None)
            if response_raw:
                logger.debug(f"Cache HIT (similarity={similarity:.4f})")
                return json.loads(response_raw) if isinstance(response_raw, (bytes, str)) else response_raw
        else:
            logger.debug(f"Cache MISS (best similarity={similarity:.4f})")

        return None

    def store(
        self,
        query_embedding: list[float],
        response: str,
        source: str = "gateway",
    ) -> str:
        """
        Store a query embedding + response in Redis with TTL.
        Returns the cache key.
        """
        if self._client is None:
            raise RuntimeError("SemanticCache.connect() must be called first.")

        key = f"{settings.key_prefix}{uuid.uuid4().hex}"
        payload = {
            "embedding": query_embedding,
            "response": response,
            "source": source,
        }
        self._client.json().set(key, "$", payload)
        self._client.expire(key, settings.ttl_seconds)
        logger.debug(f"Stored cache entry key={key}, TTL={settings.ttl_seconds}s")
        return key

    def flush(self) -> int:
        """Delete all cache entries (useful for testing)."""
        if self._client is None:
            return 0
        keys = self._client.keys(f"{settings.key_prefix}*")
        if keys:
            return self._client.delete(*keys)
        return 0

    @property
    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False