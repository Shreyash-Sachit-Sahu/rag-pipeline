"""
services/scraper/run.py

CLI entrypoint for the web scraper + auto-ingest pipeline.
Can be used standalone (no Docker needed) or called from the FastAPI service.

Examples:
  # Scrape a single site, ingest immediately
  python run.py --seed https://docs.python.org/3/library/ --depth 2

  # Scrape multiple sites
  python run.py --seed https://docs.example.com --seed https://blog.example.com

  # Scrape without ingesting (save files only)
  python run.py --seed https://docs.example.com --no-ingest

  # Use env vars only (no CLI args)
  SCRAPER_SEED_URLS=https://docs.example.com python run.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from shared.config import get_scraper_settings
from scraper import WebScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("scraper.cli")

settings = get_scraper_settings()


async def main(args: argparse.Namespace) -> int:
    seed_urls = args.seed or settings.seed_urls
    if not seed_urls:
        logger.error(
            "No seed URLs provided. Use --seed <url> or set SCRAPER_SEED_URLS env var."
        )
        return 1

    output_dir = args.output_dir or settings.output_dir
    auto_ingest = not args.no_ingest and settings.auto_ingest

    logger.info(f"Seeds       : {seed_urls}")
    logger.info(f"Max depth   : {args.depth or settings.max_depth}")
    logger.info(f"Max pages   : {args.max_pages or settings.max_pages_per_seed}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Auto-ingest : {auto_ingest}")

    # Build settings override if CLI args differ from env
    from shared.config import ScraperSettings
    override = ScraperSettings(
        **{
            "SCRAPER_SEED_URLS": ",".join(seed_urls),
            "SCRAPER_MAX_DEPTH": str(args.depth or settings.max_depth),
            "SCRAPER_MAX_PAGES_PER_SEED": str(args.max_pages or settings.max_pages_per_seed),
            "SCRAPER_OUTPUT_DIR": output_dir,
            "SCRAPER_STAY_ON_DOMAIN": str(not args.allow_external).lower(),
            "SCRAPER_REQUEST_DELAY_S": str(args.delay or settings.request_delay_s),
            "SCRAPER_CONCURRENCY": str(args.concurrency or settings.concurrency),
            "SCRAPER_AUTO_INGEST": str(auto_ingest).lower(),
            "SCRAPER_RESPECT_ROBOTS_TXT": str(not args.ignore_robots).lower(),
            "SCRAPER_MIN_CONTENT_LENGTH": str(args.min_length or settings.min_content_length),
            "SCRAPER_OUTPUT_FORMAT": args.format or settings.output_format,
        }
    )

    scraper = WebScraper(override_settings=override)
    stats = await scraper.scrape(seed_urls=seed_urls, output_dir=output_dir)

    # Print summary
    summary = {
        "pages_crawled": stats.pages_crawled,
        "pages_saved":   stats.pages_saved,
        "pages_skipped": stats.pages_skipped,
        "pages_failed":  stats.pages_failed,
        "elapsed_s":     stats.elapsed_s,
        "files":         stats.saved_files,
    }
    print("\n" + json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Stats written to {args.output_json}")

    if stats.pages_saved == 0:
        logger.warning("No pages were saved. Check seed URLs and scraper settings.")
        return 1

    # Auto-ingest
    if auto_ingest and stats.saved_files:
        logger.info(f"\nTriggering ingestion for {len(stats.saved_files)} files…")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion/"))
        from ingest import run_ingestion
        ingest_stats = run_ingestion(data_dir=output_dir)
        logger.info(f"Ingestion complete: {ingest_stats}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Web Scraper — crawl websites and feed into RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", "-s",
        action="append",
        metavar="URL",
        help="Seed URL(s) to crawl (repeatable). Overrides SCRAPER_SEED_URLS.",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=None,
        help=f"Max crawl depth (default: SCRAPER_MAX_DEPTH={settings.max_depth})",
    )
    parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=None,
        help=f"Max pages per seed (default: SCRAPER_MAX_PAGES_PER_SEED={settings.max_pages_per_seed})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"Output directory (default: SCRAPER_OUTPUT_DIR={settings.output_dir})",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default=None,
        help=f"Output format (default: SCRAPER_OUTPUT_FORMAT={settings.output_format})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=f"Seconds between requests to same domain (default: {settings.request_delay_s})",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=None,
        help=f"Concurrent requests (default: {settings.concurrency})",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Follow links outside the seed domain",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt (use responsibly)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help=f"Minimum content length in chars to save (default: {settings.min_content_length})",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip auto-ingestion after scraping",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write crawl stats to this JSON file",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
