"""
tools/web_search.py — Phase 4
Async web search with tenacity-based exponential backoff.

Changes from Phase 3:
  - Scrapingdog backend uses async httpx instead of blocking requests.
  - DuckDuckGo backend (ddgs) is sync-only; wrapped in asyncio.to_thread.
  - All network paths decorated with @retry (tenacity exponential backoff).
  - Public interface is now async: await web_search_async(query)
  - Sync shim web_search() kept for backward-compat with CLI / tests.

Backend selection via WEB_SEARCH_BACKEND env var:
  "duckduckgo"   — default, no API key required
  "tavily"       — requires TAVILY_API_KEY
  "scrapingdog"  — requires SCRAPINGDOG_API_KEY

Returns:
    list[{"title": str, "url": str, "snippet": str}]
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# ---------------------------------------------------------------------------
# Retry decorator — shared by all network-bound helpers
# ---------------------------------------------------------------------------

_retry = retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)

# ---------------------------------------------------------------------------
# DuckDuckGo backend (sync library → thread-pool)
# ---------------------------------------------------------------------------

def _search_duckduckgo_sync(query: str, max_results: int) -> list[dict[str, str]]:
    """Synchronous DuckDuckGo search — run this inside asyncio.to_thread."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError("ddgs is not installed. Run: pip install ddgs") from exc

    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   hit.get("title", ""),
                "url":     hit.get("href", ""),
                "snippet": hit.get("body", ""),
            })
    return results


async def _search_duckduckgo_async(query: str, max_results: int) -> list[dict[str, str]]:
    """Run the synchronous DuckDuckGo client in a thread so it doesn't block the event loop."""
    return await asyncio.to_thread(_search_duckduckgo_sync, query, max_results)


# ---------------------------------------------------------------------------
# Tavily backend (sync SDK → thread-pool)
# ---------------------------------------------------------------------------

def _search_tavily_sync(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ImportError("tavily-python is not installed. Run: pip install tavily-python") from exc

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY environment variable is not set.")

    client   = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)

    return [
        {
            "title":   hit.get("title", ""),
            "url":     hit.get("url", ""),
            "snippet": hit.get("content", ""),
        }
        for hit in response.get("results", [])
    ]


async def _search_tavily_async(query: str, max_results: int) -> list[dict[str, str]]:
    return await asyncio.to_thread(_search_tavily_sync, query, max_results)


# ---------------------------------------------------------------------------
# Scrapingdog backend — native async httpx + tenacity
# ---------------------------------------------------------------------------

@_retry
async def _search_scrapingdog_async(query: str, max_results: int) -> list[dict[str, str]]:
    """Async httpx call to Scrapingdog DuckDuckGo API with exponential backoff."""
    api_key = os.environ.get("SCRAPINGDOG_API_KEY")
    if not api_key:
        raise EnvironmentError("SCRAPINGDOG_API_KEY environment variable is not set.")

    url    = "https://api.scrapingdog.com/duckduckgo/search"
    params = {"api_key": api_key, "query": query}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    organic = data.get("organic_results", [])
    return [
        {
            "title":   hit.get("title", ""),
            "url":     hit.get("link", hit.get("displayed_link", "")),
            "snippet": hit.get("snippet", ""),
        }
        for hit in organic[:max_results]
    ]


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def web_search_async(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search the web asynchronously and return structured results.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 5, clamped [1, 20]).

    Returns:
        A list of result dicts with keys: title, url, snippet.
        Returns an empty list if the backend returns nothing.

    Raises:
        ValueError: If query is empty.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    max_results = max(1, min(max_results, 20))
    backend     = os.environ.get("WEB_SEARCH_BACKEND", "duckduckgo").lower()

    if backend == "tavily":
        return await _search_tavily_async(query, max_results)
    if backend == "scrapingdog":
        return await _search_scrapingdog_async(query, max_results)
    # default
    return await _search_duckduckgo_async(query, max_results)


# ---------------------------------------------------------------------------
# Sync shim for backward-compat (CLI runner / unit tests)
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Synchronous wrapper around web_search_async.
    Useful for CLI usage and tests that don't run inside an event loop.
    """
    return asyncio.run(web_search_async(query, max_results))
