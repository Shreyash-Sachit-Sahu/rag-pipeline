"""
tests/locustfile.py

Locust load-test definition for the RAG gateway (PRD §4 — latency benchmark).
Run with:
  locust -f tests/locustfile.py --host http://localhost:8000
  locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 --run-time 60s

All config comes from environment variables.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

SIMPLE_QUERIES = [
    "What is FAISS?",
    "Explain vector embeddings.",
    "What is Redis Stack?",
    "Define cosine similarity.",
    "What is sentence-transformers?",
    "How does semantic search work?",
    "What is retrieval-augmented generation?",
    "Explain HNSW indexing.",
    "What is a context window in LLMs?",
    "How does MMR re-ranking reduce redundancy?",
]

COMPLEX_QUERIES = [
    (
        "Compare IVF and HNSW indices for corpora over 1M vectors, "
        "discussing recall-latency trade-offs and nprobe tuning."
    ),
    (
        "Explain how multi-hop retrieval improves recall for complex queries "
        "and what latency overhead it adds versus single-hop retrieval."
    ),
    (
        "Describe the full RAG pipeline from user query to LLM response, "
        "including semantic caching, complexity classification, and MMR re-ranking."
    ),
]

WAIT_MIN = float(os.getenv("LOCUST_WAIT_MIN", "0.5"))
WAIT_MAX = float(os.getenv("LOCUST_WAIT_MAX", "2.0"))
SIMPLE_WEIGHT = int(os.getenv("LOCUST_SIMPLE_WEIGHT", "8"))
COMPLEX_WEIGHT = int(os.getenv("LOCUST_COMPLEX_WEIGHT", "2"))


class RAGUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task(SIMPLE_WEIGHT)
    def simple_query(self):
        query = random.choice(SIMPLE_QUERIES)
        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True,
            name="/query [simple]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.json().get("route") not in ("simple", "cache_hit"):
                resp.success()

    @task(COMPLEX_WEIGHT)
    def complex_query(self):
        query = random.choice(COMPLEX_QUERIES)
        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True,
            name="/query [complex]",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
