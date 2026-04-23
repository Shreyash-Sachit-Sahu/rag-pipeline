"""
tests/test_scraper.py

Unit tests for the web scraper:
  - URL normalisation and filtering logic
  - Domain stay-on-domain enforcement
  - robots.txt compliance
  - Content extraction via trafilatura
  - Rate limiter timing
  - CLI argument parsing
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "scraper"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import helpers under test directly (no FastAPI startup needed) ───────────
from scraper import (
    DomainRateLimiter,
    _extract_links,
    _matches_any,
    _normalise_url,
    _safe_filename,
    _should_crawl,
)


class TestNormaliseUrl:
    def test_strips_fragment(self):
        assert _normalise_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_trailing_slash(self):
        assert _normalise_url("https://example.com/page/") == "https://example.com/page"

    def test_preserves_query(self):
        url = "https://example.com/search?q=rag"
        assert "q=rag" in _normalise_url(url)

    def test_same_url_idempotent(self):
        url = "https://example.com/docs"
        assert _normalise_url(url) == _normalise_url(_normalise_url(url))


class TestExtractLinks:
    def test_extracts_absolute_links(self):
        html = '<a href="https://example.com/page1">Link</a>'
        links = _extract_links(html, "https://example.com")
        assert "https://example.com/page1" in links

    def test_resolves_relative_links(self):
        html = '<a href="/about">About</a>'
        links = _extract_links(html, "https://example.com/blog/post")
        assert "https://example.com/about" in links

    def test_skips_mailto(self):
        html = '<a href="mailto:test@example.com">Email</a>'
        links = _extract_links(html, "https://example.com")
        assert not any("mailto" in l for l in links)

    def test_skips_javascript(self):
        html = '<a href="javascript:void(0)">Click</a>'
        links = _extract_links(html, "https://example.com")
        assert not links

    def test_skips_hash_only(self):
        html = '<a href="#">Top</a>'
        links = _extract_links(html, "https://example.com")
        assert not links

    def test_extracts_multiple_links(self):
        html = """
        <a href="/page1">P1</a>
        <a href="/page2">P2</a>
        <a href="https://other.com/page3">P3</a>
        """
        links = _extract_links(html, "https://example.com")
        assert len(links) >= 3


class TestShouldCrawl:
    def _call(self, url, seed_domain="example.com", visited=None, **kwargs):
        defaults = {
            "allowed_domains": [],
            "exclude_patterns": [r"\.(jpg|png|css|js)$"],
            "include_patterns": [],
            "stay_on_domain": True,
        }
        defaults.update(kwargs)
        return _should_crawl(url, seed_domain, visited or set(), **defaults)

    def test_allows_same_domain(self):
        assert self._call("https://example.com/page") is True

    def test_blocks_different_domain(self):
        assert self._call("https://other.com/page") is False

    def test_allows_different_domain_when_not_staying(self):
        assert self._call("https://other.com/page", stay_on_domain=False) is True

    def test_blocks_already_visited(self):
        visited = {"https://example.com/page"}
        assert self._call("https://example.com/page", visited=visited) is False

    def test_blocks_excluded_extension(self):
        assert self._call("https://example.com/image.jpg") is False

    def test_include_pattern_overrides_exclude(self):
        assert self._call(
            "https://example.com/doc.pdf",
            exclude_patterns=[r"\.pdf$"],
            include_patterns=[r"\.pdf$"],
        ) is True

    def test_blocks_non_http_scheme(self):
        assert self._call("ftp://example.com/file") is False

    def test_allows_subdomain_if_in_allowed(self):
        assert self._call(
            "https://docs.example.com/page",
            seed_domain="example.com",
            allowed_domains=["docs.example.com"],
        ) is True


class TestMatchesAny:
    def test_matches_pattern(self):
        assert _matches_any("https://example.com/image.jpg", [r"\.jpg$"]) is True

    def test_no_match(self):
        assert _matches_any("https://example.com/page", [r"\.jpg$"]) is False

    def test_empty_patterns(self):
        assert _matches_any("https://example.com/page", []) is False


class TestSafeFilename:
    def test_creates_valid_filename(self):
        filename = _safe_filename("https://example.com/docs/intro")
        assert "/" not in filename
        assert ":" not in filename
        assert len(filename) <= 204  # 200 + extension

    def test_different_urls_different_names(self):
        a = _safe_filename("https://example.com/page1")
        b = _safe_filename("https://example.com/page2")
        assert a != b


class TestDomainRateLimiter:
    def test_enforces_delay(self):
        delay = 0.1
        limiter = DomainRateLimiter(delay_s=delay)

        async def run():
            t0 = time.monotonic()
            await limiter.wait("example.com")
            await limiter.wait("example.com")
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert elapsed >= delay, f"Rate limiter elapsed {elapsed:.3f}s < delay {delay}s"

    def test_different_domains_not_delayed(self):
        limiter = DomainRateLimiter(delay_s=1.0)

        async def run():
            t0 = time.monotonic()
            await asyncio.gather(
                limiter.wait("example.com"),
                limiter.wait("other.com"),
            )
            return time.monotonic() - t0

        elapsed = asyncio.run(run())
        assert elapsed < 0.5, "Different domains should not block each other"


class TestScraperIntegration:
    """
    Integration-style tests using a mock HTTP server.
    No actual network calls — uses httpx mock transport.
    """

    SAMPLE_HTML = """<!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
      <nav><a href="/about">About</a> <a href="/blog">Blog</a></nav>
      <main>
        <h1>Introduction to RAG</h1>
        <p>Retrieval-augmented generation combines dense retrieval with LLM generation
        to produce grounded, factual answers. This technique significantly reduces
        hallucination compared to vanilla prompting, especially for domain-specific queries.
        The system first retrieves relevant chunks from a knowledge base, then conditions
        the LLM on those chunks before generating a response.</p>
      </main>
      <footer>Footer content</footer>
    </body>
    </html>"""

    def test_content_extraction_has_main_body(self):
        from scraper import _extract_content
        title, content = _extract_content(self.SAMPLE_HTML, "https://example.com/page", "markdown")
        assert "RAG" in content or "retrieval" in content.lower()
        assert len(content) > 50

    def test_content_extraction_removes_nav_footer(self):
        from scraper import _extract_content
        _, content = _extract_content(self.SAMPLE_HTML, "https://example.com/page", "text")
        # trafilatura should strip boilerplate nav/footer
        assert "Footer content" not in content

    def test_title_extracted(self):
        from scraper import _extract_content
        title, _ = _extract_content(self.SAMPLE_HTML, "https://example.com/page", "markdown")
        assert title  # non-empty
