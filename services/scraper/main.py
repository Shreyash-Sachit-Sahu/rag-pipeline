"""
services/scraper/main.py

FastAPI service wrapping the web scraper.
Exposes endpoints to start, monitor, and retrieve scraping jobs.
Optionally triggers the ingestion pipeline automatically on completion.

Endpoints:
  POST /scrape          — start a scrape job (async, returns job_id)
  GET  /scrape/{job_id} — poll job status and results
  GET  /scrape          — list all jobs
  POST /scrape/sync     — scrape + ingest synchronously (for CLI use)
  GET  /health          — liveness probe
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_scraper_settings
from scraper import CrawlStats, WebScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper.api")

settings = get_scraper_settings()


# ── Job state ─────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"


class ScrapeJob(BaseModel):
    job_id: str
    status: JobStatus
    seed_urls: List[str]
    output_dir: str
    stats: Optional[dict] = None
    error: Optional[str] = None
    ingest_triggered: bool = False


_jobs: Dict[str, ScrapeJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Scraper service ready. "
        f"Default seeds: {settings.seed_urls or '(none — pass in request)'}, "
        f"output_dir: {settings.output_dir}"
    )
    yield


app = FastAPI(
    title="RAG Web Scraper",
    version="1.0.0",
    description="Crawl websites and feed content into the RAG ingestion pipeline.",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    seed_urls: Optional[List[str]] = Field(
        None,
        description="Override SCRAPER_SEED_URLS env var for this job",
        examples=[["https://docs.python.org/3/library/"]],
    )
    max_depth: Optional[int] = Field(None, description="Override SCRAPER_MAX_DEPTH")
    max_pages_per_seed: Optional[int] = Field(None, description="Override SCRAPER_MAX_PAGES_PER_SEED")
    output_dir: Optional[str] = Field(None, description="Override SCRAPER_OUTPUT_DIR")
    auto_ingest: Optional[bool] = Field(None, description="Override SCRAPER_AUTO_INGEST")


class ScrapeResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


# ── Background job runner ─────────────────────────────────────

async def _run_job(job_id: str, request: ScrapeRequest) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING

    try:
        # Build override settings if request contains overrides
        override = None
        if any(v is not None for v in [
            request.max_depth,
            request.max_pages_per_seed,
            request.output_dir,
        ]):
            from shared.config import ScraperSettings
            override = ScraperSettings(
                **{
                    "SCRAPER_SEED_URLS": ",".join(request.seed_urls or settings.seed_urls),
                    "SCRAPER_MAX_DEPTH": str(request.max_depth or settings.max_depth),
                    "SCRAPER_MAX_PAGES_PER_SEED": str(
                        request.max_pages_per_seed or settings.max_pages_per_seed
                    ),
                    "SCRAPER_OUTPUT_DIR": request.output_dir or settings.output_dir,
                    "SCRAPER_AUTO_INGEST": str(
                        request.auto_ingest if request.auto_ingest is not None
                        else settings.auto_ingest
                    ).lower(),
                }
            )

        scraper = WebScraper(override_settings=override)
        stats: CrawlStats = await scraper.scrape(
            seed_urls=request.seed_urls,
            output_dir=request.output_dir,
        )

        job.stats = {
            "pages_crawled":    stats.pages_crawled,
            "pages_saved":      stats.pages_saved,
            "pages_skipped":    stats.pages_skipped,
            "pages_failed":     stats.pages_failed,
            "elapsed_s":        stats.elapsed_s,
            "saved_files":      stats.saved_files,
        }

        # Auto-trigger ingestion if enabled and there are saved files
        should_ingest = (
            request.auto_ingest if request.auto_ingest is not None else settings.auto_ingest
        )
        if should_ingest and stats.saved_files:
            logger.info(f"Job {job_id}: triggering ingestion for {len(stats.saved_files)} files…")
            await _trigger_ingest(request.output_dir or settings.output_dir)
            job.ingest_triggered = True

        job.status = JobStatus.DONE
        logger.info(f"Job {job_id} complete: {job.stats}")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        job.status = JobStatus.FAILED
        job.error = str(exc)


async def _trigger_ingest(output_dir: str) -> None:
    """
    Call the ingestion service (or run inline if not running as separate container).
    Tries the ingestion HTTP endpoint first; falls back to in-process execution.
    """
    ingest_url = os.getenv("INGEST_URL", "")
    if ingest_url:
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{ingest_url}/ingest",
                    json={"data_dir": output_dir},
                )
                resp.raise_for_status()
                logger.info(f"Ingestion triggered via HTTP: {resp.json()}")
                return
        except Exception as exc:
            logger.warning(f"HTTP ingest trigger failed ({exc}), falling back to in-process…")

    # In-process fallback
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion/"))
    from ingest import run_ingestion
    from shared.config import get_ingestion_settings

    ingest_settings = get_ingestion_settings()
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: run_ingestion(data_dir=output_dir),
    )
    logger.info(f"In-process ingestion complete: {stats}")


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/scrape", response_model=ScrapeResponse, status_code=202)
async def start_scrape(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
) -> ScrapeResponse:
    """Start an async scrape job. Poll /scrape/{job_id} for status."""
    seeds = request.seed_urls or settings.seed_urls
    if not seeds:
        raise HTTPException(
            status_code=400,
            detail=(
                "No seed_urls provided and SCRAPER_SEED_URLS env var is not set. "
                "Pass seed_urls in the request body."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    job = ScrapeJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        seed_urls=seeds,
        output_dir=request.output_dir or settings.output_dir,
    )
    _jobs[job_id] = job
    background_tasks.add_task(_run_job, job_id, request)

    return ScrapeResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Scrape job started for {len(seeds)} seed URL(s). Poll /scrape/{job_id} for progress.",
    )


@app.get("/scrape/{job_id}", response_model=ScrapeJob)
async def get_job(job_id: str) -> ScrapeJob:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


@app.get("/scrape", response_model=List[ScrapeJob])
async def list_jobs() -> List[ScrapeJob]:
    return list(_jobs.values())


@app.post("/scrape/sync", response_model=ScrapeJob)
async def scrape_sync(request: ScrapeRequest) -> ScrapeJob:
    """
    Synchronous scrape + ingest. Blocks until complete.
    Use for simple CLI / one-shot workflows.
    """
    seeds = request.seed_urls or settings.seed_urls
    if not seeds:
        raise HTTPException(status_code=400, detail="No seed_urls provided.")

    job_id = uuid.uuid4().hex[:12]
    job = ScrapeJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        seed_urls=seeds,
        output_dir=request.output_dir or settings.output_dir,
    )
    _jobs[job_id] = job
    await _run_job(job_id, request)
    return _jobs[job_id]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "seed_urls_configured": len(settings.seed_urls),
        "output_dir": settings.output_dir,
        "auto_ingest": settings.auto_ingest,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("SCRAPER_PORT", "8003")),
        log_level="info",
    )
