"""
services/ingestion/ingest.py

Offline ingestion pipeline (PRD §1.3).

Reads documents from INGEST_DATA_DIR, chunks at CHUNK_SIZE tokens with
CHUNK_OVERLAP overlap, calls the embedder service, writes to FAISS index
and SQLite metadata store.

Supported formats (configurable via INGEST_SUPPORTED_EXTENSIONS):
  .pdf   — via langchain-community PyPDFLoader
  .md    — via langchain-community UnstructuredMarkdownLoader
  .txt   — native Python

No values are hard-coded; all config from IngestionSettings (env vars).
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Generator, List, Tuple

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_ingestion_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

settings = get_ingestion_settings()

# ── Document loading ─────────────────────────────────────────

def load_documents(data_dir: str, extensions: List[str]) -> Generator[dict, None, None]:
    """
    Walk `data_dir` and yield {text, source, page} dicts for each supported file.
    Uses langchain-community loaders for PDF and Markdown; native read for .txt.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return

    for ext in extensions:
        for file_path in data_path.rglob(f"*{ext}"):
            logger.info(f"Loading: {file_path}")
            try:
                if ext == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(file_path))
                    for doc in loader.load():
                        yield {
                            "text": doc.page_content,
                            "source": str(file_path.relative_to(data_path)),
                            "page": doc.metadata.get("page", 0),
                        }

                elif ext in (".md", ".markdown"):
                    from langchain_community.document_loaders import UnstructuredMarkdownLoader
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    for doc in loader.load():
                        yield {
                            "text": doc.page_content,
                            "source": str(file_path.relative_to(data_path)),
                            "page": 0,
                        }

                elif ext == ".txt":
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    yield {
                        "text": text,
                        "source": str(file_path.relative_to(data_path)),
                        "page": 0,
                    }

            except Exception as exc:
                logger.error(f"Failed to load {file_path}: {exc}")


# ── Chunking ─────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> List[str]:
    """
    Split text into overlapping token-level chunks.
    Uses whitespace tokenisation (fast, no dependency).
    chunk_size and overlap are in tokens (words).
    """
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
        start += chunk_size - overlap  # sliding window with overlap

    return chunks


# ── Embedding via HTTP ────────────────────────────────────────

def embed_texts(
    texts: List[str],
    embedder_url: str = settings.embedder_url,
    batch_size: int = 64,
) -> np.ndarray:
    """Call the embedder service and return a 2D float32 array."""
    all_embeddings = []
    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.post(f"{embedder_url}/embed", json={"texts": batch})
            resp.raise_for_status()
            all_embeddings.extend(resp.json()["embeddings"])
            logger.info(
                f"Embedded batch {i // batch_size + 1}: "
                f"{i + len(batch)}/{len(texts)} texts"
            )
    return np.array(all_embeddings, dtype=np.float32)


# ── SQLite metadata store ────────────────────────────────────

def init_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            faiss_idx   INTEGER PRIMARY KEY,
            chunk_id    TEXT NOT NULL UNIQUE,
            doc_id      TEXT NOT NULL,
            text        TEXT NOT NULL,
            source      TEXT NOT NULL,
            page        INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id   ON chunks(doc_id)")
    conn.commit()
    return conn


def insert_chunks(
    conn: sqlite3.Connection,
    faiss_start_idx: int,
    chunk_records: List[dict],
) -> None:
    rows = [
        (
            faiss_start_idx + i,
            rec["chunk_id"],
            rec["doc_id"],
            rec["text"],
            rec["source"],
            rec.get("page"),
        )
        for i, rec in enumerate(chunk_records)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO chunks (faiss_idx, chunk_id, doc_id, text, source, page) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ── FAISS index management ────────────────────────────────────

def build_or_append_faiss(
    embeddings: np.ndarray,
    index_path: str,
) -> Tuple[int, int]:
    """
    Build a new FAISS index or append to existing.
    Returns (start_idx, end_idx) of the inserted vectors.
    """
    import faiss

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    if os.path.exists(index_path):
        logger.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        logger.info(f"Creating new FAISS IndexFlatIP (dim={dim})")
        index = faiss.IndexFlatIP(dim)

    start_idx = index.ntotal
    index.add(embeddings)
    end_idx = index.ntotal

    faiss.write_index(index, index_path)
    logger.info(
        f"FAISS index saved: {end_idx} total vectors (+{end_idx - start_idx} new)"
    )
    return start_idx, end_idx


# ── Main ingestion entry point ────────────────────────────────

def run_ingestion(
    data_dir: str = settings.data_dir,
    index_path: str = settings.faiss_index_path,
    sqlite_path: str = settings.sqlite_path,
    embedder_url: str = settings.embedder_url,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    supported_extensions: List[str] = settings.supported_extensions,
    dry_run: bool = False,
) -> dict:
    """
    Full ingestion pipeline. Returns stats dict.
    All parameters default from env vars.
    """
    t0 = time.perf_counter()
    stats = {
        "docs_processed": 0,
        "chunks_total": 0,
        "embeddings_computed": 0,
        "elapsed_seconds": 0.0,
    }

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)

    # ── Collect all chunks from all documents ────────────────
    all_chunk_records: List[dict] = []
    doc_count = 0

    for page_doc in load_documents(data_dir, supported_extensions):
        chunks = chunk_text(page_doc["text"], chunk_size, chunk_overlap)
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        for chunk_text_str in chunks:
            all_chunk_records.append(
                {
                    "chunk_id": f"{doc_id}_c{len(all_chunk_records)}",
                    "doc_id": doc_id,
                    "text": chunk_text_str,
                    "source": page_doc["source"],
                    "page": page_doc.get("page"),
                }
            )
        doc_count += 1
        logger.info(
            f"Doc {doc_count}: {page_doc['source']} → {len(chunks)} chunks "
            f"(total so far: {len(all_chunk_records)})"
        )

    stats["docs_processed"] = doc_count
    stats["chunks_total"] = len(all_chunk_records)
    logger.info(f"Total: {doc_count} docs, {len(all_chunk_records)} chunks")

    if not all_chunk_records:
        logger.warning("No chunks produced. Check data directory and file formats.")
        return stats

    if dry_run:
        logger.info("Dry run — skipping embedding and indexing.")
        return stats

    # ── Embed all chunks ─────────────────────────────────────
    texts = [r["text"] for r in all_chunk_records]
    embeddings = embed_texts(texts, embedder_url=embedder_url)
    stats["embeddings_computed"] = len(embeddings)

    # ── Write FAISS index ────────────────────────────────────
    start_idx, end_idx = build_or_append_faiss(embeddings, index_path)

    # ── Write SQLite metadata ────────────────────────────────
    conn = init_sqlite(sqlite_path)
    insert_chunks(conn, start_idx, all_chunk_records)
    conn.close()

    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    logger.info(f"Ingestion complete: {stats}")
    return stats


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--index-path", default=settings.faiss_index_path)
    parser.add_argument("--sqlite-path", default=settings.sqlite_path)
    parser.add_argument("--embedder-url", default=settings.embedder_url)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_ingestion(
        data_dir=args.data_dir,
        index_path=args.index_path,
        sqlite_path=args.sqlite_path,
        embedder_url=args.embedder_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dry_run=args.dry_run,
    )
