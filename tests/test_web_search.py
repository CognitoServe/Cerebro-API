"""
tests/test_web_search.py — Phase 4
Tests for the async web_search tool.

Mocked tests (default, no key required):
  - Structural assertions on the sync shim (web_search)
  - Async interface (web_search_async) via asyncio.run / pytest-asyncio
  - Error handling: empty query raises ValueError

Live tests (gated by WEB_SEARCH_LIVE=1):
  - Real DuckDuckGo fetch verifying non-empty results with correct structure
"""
from __future__ import annotations

import asyncio
import os

import pytest

from tools.web_search import web_search, web_search_async


# ---------------------------------------------------------------------------
# Mocked / structural tests
# ---------------------------------------------------------------------------

class TestWebSearchMocked:

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            web_search("")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            web_search("   ")

    @pytest.mark.asyncio
    async def test_async_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            await web_search_async("")


# ---------------------------------------------------------------------------
# Live tests — only run if WEB_SEARCH_LIVE=1
# ---------------------------------------------------------------------------

_live = pytest.mark.skipif(
    os.environ.get("WEB_SEARCH_LIVE") != "1",
    reason="Set WEB_SEARCH_LIVE=1 to run live web-search tests",
)


@_live
def test_live_web_search_returns_list():
    results = web_search("Python programming language")
    assert isinstance(results, list)
    assert len(results) > 0


@_live
def test_live_web_search_result_structure():
    results = web_search("Python 3.13 release", max_results=3)
    assert len(results) > 0
    for r in results:
        assert "title"   in r
        assert "url"     in r
        assert "snippet" in r
        assert isinstance(r["url"], str) and r["url"]


@_live
def test_live_max_results_respected():
    results = web_search("machine learning tutorial", max_results=2)
    assert len(results) <= 2


@_live
@pytest.mark.asyncio
async def test_live_async_web_search():
    results = await web_search_async("FastAPI Python web framework", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "url" in r and r["url"]
