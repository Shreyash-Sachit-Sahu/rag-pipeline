"""
scripts/eval_recall.py

Offline retrieval recall evaluation (PRD §5 / §6.1).

Measures Recall@k against a ground-truth QA set and pushes the result
to the Prometheus pushgateway (or prints it) so Grafana can display it.

All config comes from environment variables.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_recall")

GATEWAY_HOST   = os.getenv("GATEWAY_HOST", "localhost")
GATEWAY_PORT   = int(os.getenv("GATEWAY_PORT", "8000"))
EMBEDDER_HOST  = os.getenv("EMBEDDER_HOST", "localhost")
EMBEDDER_PORT  = int(os.getenv("EMBEDDER_PORT", "8001"))
RETRIEVER_HOST = os.getenv("RETRIEVER_HOST", "localhost")
RETRIEVER_PORT = int(os.getenv("RETRIEVER_PORT", "8002"))
RECALL_K       = int(os.getenv("EVAL_RECALL_K", "5"))
TARGET_RECALL  = float(os.getenv("EVAL_TARGET_RECALL", "0.91"))

EMBEDDER_URL   = f"http://{EMBEDDER_HOST}:{EMBEDDER_PORT}"
RETRIEVER_URL  = f"http://{RETRIEVER_HOST}:{RETRIEVER_PORT}"

# ── Ground-truth QA pairs ─────────────────────────────────────
# In production these come from a JSON file; defaults are illustrative.
DEFAULT_QA_PATH = os.getenv(
    "EVAL_QA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "qa_pairs.json"),
)


def load_qa_pairs(path: str) -> List[dict]:
    """
    Load QA pairs from JSON file.
    Expected format: [{"question": str, "relevant_chunk_ids": [str, ...]}]
    """
    if not os.path.exists(path):
        logger.warning(f"QA pairs file not found: {path}. Using built-in sample.")
        return [
            {
                "question": "What is FAISS?",
                "relevant_chunk_ids": ["sample_chunk_001"],
            },
            {
                "question": "How does semantic caching work?",
                "relevant_chunk_ids": ["sample_chunk_002"],
            },
        ]
    with open(path) as f:
        return json.load(f)


def embed(texts: List[str]) -> List[List[float]]:
    resp = httpx.post(f"{EMBEDDER_URL}/embed", json={"texts": texts}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def retrieve(embedding: List[float], k: int) -> List[str]:
    resp = httpx.post(
        f"{RETRIEVER_URL}/retrieve",
        json={"query_embedding": embedding, "k": k, "multi_hop": False},
        timeout=30.0,
    )
    resp.raise_for_status()
    return [c["chunk_id"] for c in resp.json()["chunks"]]


def compute_recall(qa_pairs: List[dict], k: int) -> float:
    hits = 0
    total = 0
    for pair in qa_pairs:
        question = pair["question"]
        relevant_ids = set(pair["relevant_chunk_ids"])
        total += 1

        try:
            embeddings = embed([question])
            retrieved_ids = retrieve(embeddings[0], k=k)
            if relevant_ids & set(retrieved_ids):
                hits += 1
        except Exception as exc:
            logger.warning(f"Eval failed for '{question[:50]}': {exc}")

    return hits / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval recall@k")
    parser.add_argument("--k",            type=int,   default=RECALL_K)
    parser.add_argument("--target",       type=float, default=TARGET_RECALL)
    parser.add_argument("--qa-path",      type=str,   default=DEFAULT_QA_PATH)
    parser.add_argument("--output-json",  type=str,   default=None)
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_path)
    logger.info(f"Evaluating recall@{args.k} on {len(qa_pairs)} QA pairs…")

    recall = compute_recall(qa_pairs, k=args.k)
    result = {
        "recall_at_k": round(recall, 4),
        "k": args.k,
        "n_pairs": len(qa_pairs),
        "target": args.target,
        "passed": recall >= args.target,
    }

    logger.info(f"Recall@{args.k} = {recall:.4f} (target: {args.target})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    if not result["passed"]:
        logger.error(f"FAIL: recall@{args.k} {recall:.4f} < target {args.target}")
        sys.exit(1)
    else:
        logger.info(f"PASS: recall@{args.k} {recall:.4f} >= target {args.target}")
        sys.exit(0)


if __name__ == "__main__":
    main()
