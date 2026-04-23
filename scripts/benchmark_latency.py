"""
scripts/benchmark_latency.py

Latency benchmark runner (PRD §6.3).
Fires N_QUERIES against the gateway and asserts P99 targets.
Used in CI to gate merges — fails with exit code 1 if targets are breached.

All targets and config come from environment variables.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from typing import List, Optional

import httpx
import numpy as np

# ── Config from env (no hard-coded values) ───────────────────
GATEWAY_HOST   = os.getenv("GATEWAY_HOST", "localhost")
GATEWAY_PORT   = int(os.getenv("GATEWAY_PORT", "8000"))
N_QUERIES      = int(os.getenv("BENCHMARK_N_QUERIES", "500"))
P99_CACHED_MAX = float(os.getenv("BENCHMARK_P99_CACHED_MAX_MS", "100"))
P99_COMPLEX_MAX= float(os.getenv("BENCHMARK_P99_COMPLEX_MAX_MS", "800"))
CONCURRENCY    = int(os.getenv("BENCHMARK_CONCURRENCY", "10"))
TIMEOUT_S      = float(os.getenv("BENCHMARK_TIMEOUT_S", "30"))
GATEWAY_URL    = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")

# ── Synthetic query bank ─────────────────────────────────────
_SIMPLE_QUERIES = [
    "What is retrieval-augmented generation?",
    "Explain how FAISS works.",
    "What is a vector embedding?",
    "How does semantic search differ from keyword search?",
    "What is Redis Stack?",
    "Define cosine similarity.",
    "What is sentence-transformers?",
    "How does IVF indexing work?",
    "What is quantisation in LLMs?",
    "What is HNSW?",
]

_COMPLEX_QUERIES = [
    "Compare IVF and HNSW indices and explain when to use each, considering trade-offs in recall and latency for corpora over 1M vectors.",
    "How does multi-hop retrieval improve recall for complex queries and what are the latency implications compared to single-hop?",
    "Explain the full pipeline from user query to LLM response, including cache lookup, complexity classification, and MMR re-ranking.",
    "What are the advantages of using a local LLM like Mistral-7B over OpenAI API in a production RAG system, considering cost, latency, and privacy?",
    "Describe how semantic caching works and how to tune the similarity threshold to balance hit rate against answer staleness.",
]


def generate_query_bank(n: int) -> List[str]:
    """Generate a mix of simple and complex queries, with cache-warming duplicates."""
    bank = []
    # First 20% are repeated (to warm cache and measure hit latency)
    warm_queries = _SIMPLE_QUERIES[:5]
    for _ in range(max(1, n // 5)):
        bank.append(random.choice(warm_queries))
    # Fill rest with shuffled mix
    pool = _SIMPLE_QUERIES * 20 + _COMPLEX_QUERIES * 5
    random.shuffle(pool)
    bank.extend(pool[: n - len(bank)])
    random.shuffle(bank)
    return bank[:n]


# ── Async runner ─────────────────────────────────────────────

async def run_query(
    client: httpx.AsyncClient,
    query: str,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{GATEWAY_URL}/query",
                json={"query": query},
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "latency_ms": elapsed_ms,
                "route": data.get("route", "unknown"),
                "cache_hit": data.get("cache_hit", False),
                "status": "ok",
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(f"Query failed after {elapsed_ms:.0f}ms: {exc}")
            return {"latency_ms": elapsed_ms, "route": "error", "cache_hit": False, "status": "error"}


async def run_benchmark(
    n_queries: int = N_QUERIES,
    concurrency: int = CONCURRENCY,
    assert_p99_cached: float = P99_CACHED_MAX,
    assert_p99_complex: float = P99_COMPLEX_MAX,
) -> dict:
    queries = generate_query_bank(n_queries)
    sem = asyncio.Semaphore(concurrency)
    results = []

    logger.info(
        f"Starting benchmark: {n_queries} queries, "
        f"concurrency={concurrency}, gateway={GATEWAY_URL}"
    )

    async with httpx.AsyncClient() as client:
        tasks = [run_query(client, q, sem) for q in queries]
        t_total_start = time.perf_counter()
        raw_results = await asyncio.gather(*tasks)
        total_elapsed_s = time.perf_counter() - t_total_start

    results = [r for r in raw_results if r is not None]

    # ── Partition by route ───────────────────────────────────
    cached    = [r["latency_ms"] for r in results if r["route"] == "cache_hit"]
    simple    = [r["latency_ms"] for r in results if r["route"] == "simple"]
    complex_  = [r["latency_ms"] for r in results if r["route"] == "complex"]
    errors    = [r for r in results if r["status"] == "error"]

    def pct(data: list, p: float) -> float:
        return float(np.percentile(data, p)) if data else 0.0

    stats = {
        "total_queries":      n_queries,
        "completed":          len(results),
        "errors":             len(errors),
        "error_rate_pct":     round(100 * len(errors) / max(len(results), 1), 2),
        "total_elapsed_s":    round(total_elapsed_s, 2),
        "qps":                round(len(results) / total_elapsed_s, 1),
        "cache": {
            "n":   len(cached),
            "p50": round(pct(cached, 50), 2),
            "p95": round(pct(cached, 95), 2),
            "p99": round(pct(cached, 99), 2),
        },
        "simple": {
            "n":   len(simple),
            "p50": round(pct(simple, 50), 2),
            "p95": round(pct(simple, 95), 2),
            "p99": round(pct(simple, 99), 2),
        },
        "complex": {
            "n":   len(complex_),
            "p50": round(pct(complex_, 50), 2),
            "p95": round(pct(complex_, 95), 2),
            "p99": round(pct(complex_, 99), 2),
        },
    }

    # ── Print report ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(json.dumps(stats, indent=2))

    # ── Assert targets ────────────────────────────────────────
    failures = []

    if cached and stats["cache"]["p99"] > assert_p99_cached:
        failures.append(
            f"Cache-hit P99 {stats['cache']['p99']}ms > target {assert_p99_cached}ms"
        )

    if complex_ and stats["complex"]["p99"] > assert_p99_complex:
        failures.append(
            f"Complex P99 {stats['complex']['p99']}ms > target {assert_p99_complex}ms"
        )

    if stats["error_rate_pct"] > 5.0:
        failures.append(f"Error rate {stats['error_rate_pct']}% > 5%")

    if failures:
        logger.error("\nBENCHMARK FAILED:")
        for f in failures:
            logger.error(f"  ✗ {f}")
        return {**stats, "passed": False, "failures": failures}
    else:
        logger.info("\n✓ All benchmark targets met.")
        return {**stats, "passed": True, "failures": []}


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Latency Benchmark")
    parser.add_argument("--n-queries",        type=int,   default=N_QUERIES)
    parser.add_argument("--concurrency",      type=int,   default=CONCURRENCY)
    parser.add_argument("--assert-p99-lt",    type=float, default=P99_CACHED_MAX,
                        help="Max allowed P99 latency (ms) for cached queries")
    parser.add_argument("--assert-complex-p99-lt", type=float, default=P99_COMPLEX_MAX,
                        help="Max allowed P99 latency (ms) for complex queries")
    parser.add_argument("--output-json",      type=str,   default=None,
                        help="Write results JSON to file")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            n_queries=args.n_queries,
            concurrency=args.concurrency,
            assert_p99_cached=args.assert_p99_lt,
            assert_p99_complex=args.assert_complex_p99_lt,
        )
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {args.output_json}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
