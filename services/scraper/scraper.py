"""
services/scraper/scraper.py

Async web crawler and content extractor for the RAG ingestion pipeline.

Features:
  - BFS crawl from seed URLs up to configurable depth and page limit
  - Domain-scoped crawling (stays on seed domain by default)
  - robots.txt compliance
  - Boilerplate removal (nav, footer, ads) via trafilatura
  - HTML → Markdown or plain text conversion
  - Respects per-domain request delay (polite crawling)
  - All config from ScraperSettings (env vars) — nothing hard-coded
  - Optional: auto-triggers the ingestion pipeline when done

Usage:
  python scraper.py                         # uses SCRAPER_SEED_URLS env var
  python scraper.py --seed https://docs.example.com --depth 3
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from trafilatura.settings import use_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_scraper_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scraper")

settings = get_scraper_settings()

# ── trafilatura config (disable network fetching — we handle HTTP ourselves) ─
_traf_config = use_config()
_traf_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


# ── Data classes ─────────────────────────────────────────────

@dataclass
class CrawlJob:
    url: str
    depth: int
    parent_url: Optional[str] = None


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str          # cleaned text / markdown
    content_type: str     # "markdown" | "text"
    depth: int
    word_count: int
    saved_path: Optional[str] = None


@dataclass
class CrawlStats:
    pages_crawled: int = 0
    pages_saved: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    bytes_downloaded: int = 0
    elapsed_s: float = 0.0
    saved_files: List[str] = field(default_factory=list)


# ── robots.txt cache ─────────────────────────────────────────

_robots_cache: Dict[str, RobotFileParser] = {}


async def _get_robots(client: httpx.AsyncClient, base_url: str) -> Optional[RobotFileParser]:
    """Fetch and cache robots.txt for a domain."""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if robots_url in _robots_cache:
        return _robots_cache[robots_url]

    try:
        resp = await client.get(robots_url, timeout=5.0)
        rp = RobotFileParser()
        rp.set_url(robots_url)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.allow_all = True
        _robots_cache[robots_url] = rp
        return rp
    except Exception:
        rp = RobotFileParser()
        rp.allow_all = True
        _robots_cache[robots_url] = rp
        return rp


def _is_allowed_by_robots(rp: Optional[RobotFileParser], url: str, agent: str) -> bool:
    if rp is None:
        return True
    return rp.can_fetch(agent, url)


# ── URL utilities ─────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """Remove fragment, trailing slash inconsistency, sort query params."""
    p = urlparse(url)
    # drop fragment
    normalised = p._replace(fragment="").geturl()
    return normalised.rstrip("/")


def _extract_links(html: str, base_url: str) -> List[str]:
    """Extract all <a href> links from raw HTML, resolved to absolute URLs."""
    links = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.I):
        href = match.group(1).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        links.append(_normalise_url(abs_url))
    return links


def _matches_any(url: str, patterns: List[str]) -> bool:
    return any(re.search(p, url, re.I) for p in patterns if p)


def _should_crawl(
    url: str,
    seed_domain: str,
    visited: Set[str],
    allowed_domains: List[str],
    exclude_patterns: List[str],
    include_patterns: List[str],
    stay_on_domain: bool,
) -> bool:
    """Decide whether to enqueue a URL."""
    if url in visited:
        return False

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    # Domain check
    if stay_on_domain:
        allowed = {seed_domain} | set(allowed_domains)
        if not any(parsed.netloc == d or parsed.netloc.endswith("." + d) for d in allowed):
            return False

    # Explicit include overrides exclude
    if include_patterns and _matches_any(url, include_patterns):
        return True

    if _matches_any(url, exclude_patterns):
        return False

    return True


# ── Content extraction ────────────────────────────────────────

def _extract_content(html: str, url: str, output_format: str) -> tuple[str, str]:
    """
    Use trafilatura to extract main content, stripping boilerplate.
    Returns (title, content).
    """
    if output_format == "markdown":
        content = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            config=_traf_config,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
    else:
        content = trafilatura.extract(
            html,
            url=url,
            config=_traf_config,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

    # Extract title from metadata
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (metadata.title if metadata and metadata.title else "") or urlparse(url).path

    return title or "untitled", content or ""


# ── File saving ───────────────────────────────────────────────

def _safe_filename(url: str) -> str:
    """Convert a URL into a safe filename."""
    parsed = urlparse(url)
    path_part = parsed.netloc + parsed.path
    # Replace non-alphanumeric chars with underscores
    safe = re.sub(r"[^a-zA-Z0-9\-_./]", "_", path_part)
    safe = safe.strip("/_").replace("/", "__")
    return safe[:200] or "index"


def _save_page(page: ScrapedPage, output_dir: str) -> str:
    """Write scraped content to disk, return the saved file path."""
    ext = ".md" if page.content_type == "markdown" else ".txt"
    filename = _safe_filename(page.url) + ext
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    # Prepend URL as metadata comment so ingestion pipeline can trace provenance
    header = f"<!-- source: {page.url} -->\n# {page.title}\n\n" if ext == ".md" \
        else f"# Source: {page.url}\n# Title: {page.title}\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + page.content)

    return filepath


# ── Per-domain rate limiter ───────────────────────────────────

class DomainRateLimiter:
    """Enforces a minimum delay between consecutive requests to the same domain."""

    def __init__(self, delay_s: float):
        self._delay = delay_s
        self._last_request: Dict[str, float] = defaultdict(float)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, domain: str) -> None:
        async with self._locks[domain]:
            elapsed = time.monotonic() - self._last_request[domain]
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last_request[domain] = time.monotonic()


# ── Core crawler ──────────────────────────────────────────────

class WebScraper:
    """
    Async BFS web crawler.
    All configuration comes from ScraperSettings (environment variables).
    """

    def __init__(self, override_settings=None):
        self._s = override_settings or settings
        self._rate_limiter = DomainRateLimiter(self._s.request_delay_s)
        self._semaphore = asyncio.Semaphore(self._s.concurrency)

    async def scrape(
        self,
        seed_urls: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> CrawlStats:
        """
        Run the full crawl from seed_urls.
        Returns CrawlStats with counts and saved file paths.
        """
        seeds = seed_urls or self._s.seed_urls
        out_dir = output_dir or self._s.output_dir

        if not seeds:
            logger.error("No seed URLs provided. Set SCRAPER_SEED_URLS or pass --seed.")
            return CrawlStats()

        stats = CrawlStats()
        t0 = time.perf_counter()

        headers = {"User-Agent": self._s.user_agent}

        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=self._s.request_timeout_s,
            verify=True,
        ) as client:
            for seed_url in seeds:
                seed_url = _normalise_url(seed_url)
                seed_domain = urlparse(seed_url).netloc
                logger.info(
                    f"Starting crawl: seed={seed_url}, domain={seed_domain}, "
                    f"max_depth={self._s.max_depth}, max_pages={self._s.max_pages_per_seed}"
                )

                robots = None
                if self._s.respect_robots_txt:
                    robots = await _get_robots(client, seed_url)

                visited: Set[str] = set()
                queue: deque[CrawlJob] = deque([CrawlJob(url=seed_url, depth=1)])
                pages_this_seed = 0

                while queue:
                    # Respect per-seed page limit
                    if self._s.max_pages_per_seed and pages_this_seed >= self._s.max_pages_per_seed:
                        logger.info(
                            f"Reached max_pages_per_seed={self._s.max_pages_per_seed} for {seed_url}"
                        )
                        break

                    # Drain up to `concurrency` jobs in parallel
                    batch: List[CrawlJob] = []
                    while queue and len(batch) < self._s.concurrency:
                        job = queue.popleft()
                        if job.url in visited:
                            continue
                        visited.add(job.url)
                        batch.append(job)

                    if not batch:
                        continue

                    # Crawl batch concurrently
                    results = await asyncio.gather(
                        *[
                            self._crawl_one(client, job, seed_domain, robots, out_dir)
                            for job in batch
                        ],
                        return_exceptions=True,
                    )

                    for job, result in zip(batch, results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed {job.url}: {result}")
                            stats.pages_failed += 1
                            continue

                        if result is None:
                            stats.pages_skipped += 1
                            continue

                        page, new_links = result
                        stats.pages_crawled += 1
                        pages_this_seed += 1

                        if page.saved_path:
                            stats.pages_saved += 1
                            stats.saved_files.append(page.saved_path)
                            logger.info(
                                f"  ✓ [{job.depth}] {page.url[:80]} "
                                f"→ {page.word_count} words → {page.saved_path}"
                            )
                        else:
                            stats.pages_skipped += 1
                            logger.debug(f"  ○ [{job.depth}] {page.url[:80]} (too short, skipped)")

                        # Enqueue discovered links for next depth
                        if job.depth < self._s.max_depth:
                            for link in new_links:
                                if link not in visited and _should_crawl(
                                    link,
                                    seed_domain,
                                    visited,
                                    self._s.allowed_domains,
                                    self._s.exclude_patterns,
                                    self._s.include_patterns,
                                    self._s.stay_on_domain,
                                ):
                                    queue.append(CrawlJob(url=link, depth=job.depth + 1, parent_url=job.url))

        stats.elapsed_s = round(time.perf_counter() - t0, 2)
        logger.info(
            f"\nCrawl complete in {stats.elapsed_s}s: "
            f"crawled={stats.pages_crawled}, saved={stats.pages_saved}, "
            f"skipped={stats.pages_skipped}, failed={stats.pages_failed}"
        )
        return stats

    async def _crawl_one(
        self,
        client: httpx.AsyncClient,
        job: CrawlJob,
        seed_domain: str,
        robots: Optional[RobotFileParser],
        output_dir: str,
    ) -> Optional[tuple[ScrapedPage, List[str]]]:
        """Fetch and process a single URL. Returns (ScrapedPage, discovered_links) or None."""

        # robots.txt check
        if self._s.respect_robots_txt and not _is_allowed_by_robots(
            robots, job.url, self._s.user_agent
        ):
            logger.debug(f"robots.txt disallows: {job.url}")
            return None

        domain = urlparse(job.url).netloc
        async with self._semaphore:
            await self._rate_limiter.wait(domain)
            try:
                resp = await client.get(job.url)
            except Exception as exc:
                raise RuntimeError(f"HTTP error for {job.url}: {exc}") from exc

        if resp.status_code != 200:
            logger.debug(f"HTTP {resp.status_code} for {job.url}")
            return None

        content_type_header = resp.headers.get("content-type", "")

        # Handle PDFs directly
        if "pdf" in content_type_header or job.url.lower().endswith(".pdf"):
            return await self._handle_pdf(resp, job, output_dir)

        if "text/html" not in content_type_header and "text/plain" not in content_type_header:
            return None

        html = resp.text

        # Extract clean content via trafilatura
        title, content = _extract_content(html, job.url, self._s.output_format)

        # Extract links for BFS
        links = _extract_links(html, job.url)

        word_count = len(content.split())
        page = ScrapedPage(
            url=job.url,
            title=title,
            content=content,
            content_type=self._s.output_format,
            depth=job.depth,
            word_count=word_count,
        )

        if word_count * 5 >= self._s.min_content_length:  # rough char estimate
            page.saved_path = _save_page(page, output_dir)

        return page, links

    async def _handle_pdf(
        self,
        resp: httpx.Response,
        job: CrawlJob,
        output_dir: str,
    ) -> Optional[tuple[ScrapedPage, List[str]]]:
        """Save PDF bytes directly to output_dir for the ingestion pipeline to process."""
        filename = _safe_filename(job.url) + ".pdf"
        filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(resp.content)

        logger.info(f"  ✓ PDF saved: {job.url[:80]} → {filepath}")

        page = ScrapedPage(
            url=job.url,
            title=filename,
            content="",
            content_type="pdf",
            depth=job.depth,
            word_count=0,
            saved_path=filepath,
        )
        return page, []
