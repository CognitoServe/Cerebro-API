"""
agent/tool_registry.py — Phase 4
Async tool dispatch table.

All tools are exposed as async coroutines so the runner can await them
uniformly. Sync tools (calculator, memory, rag_query) are wrapped via
asyncio.to_thread so they don't block the event loop.

The memory tools accept an injected `job_id` kwarg — see runner.py for
how job_id is threaded through from the API layer.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from tools.calculator  import calculate
from tools.memory      import save_memory, recall_memory
from tools.rag_query   import rag_query
from tools.web_search  import web_search_async

# ---------------------------------------------------------------------------
# 1. Tool schemas (OpenAI function-call format) — unchanged
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, facts, recent events, or data "
                "not available in memory or the knowledge base. "
                "Prefer this for anything that requires up-to-date information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to send to the web.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1–20). Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": (
                "Search the local knowledge base for relevant document chunks. "
                "Use this before web_search when the topic may already be covered "
                "by internal documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to search the knowledge base for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of document chunks to return (1–10). Default 3.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely evaluate a numeric math expression. "
                "Supports: + − * / // % ** and unary −. "
                "Do NOT use for anything other than arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A numeric expression string, e.g. '2**10 + 100'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Store an important fact in memory for later retrieval during this session. "
                "Use sparingly — only for facts you will definitely need to reference again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact or piece of information to remember.",
                    },
                    "label": {
                        "type": "string",
                        "description": "A short identifier tag for the fact (optional).",
                        "default": "",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search previously saved memory for facts matching the query. "
                "Use this before web_search to avoid redundant lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search string to match against stored facts.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Async tool wrappers
# ---------------------------------------------------------------------------

async def _async_web_search(query: str, max_results: int = 5, **_: Any) -> list[dict]:
    return await web_search_async(query, max_results)


async def _async_rag_query(query: str, top_k: int = 3, **_: Any) -> list[dict]:
    return await asyncio.to_thread(rag_query, query, top_k)


async def _async_calculate(expression: str, **_: Any) -> dict[str, Any]:
    result = await asyncio.to_thread(calculate, expression)
    return {"expression": expression, "result": result}


async def _async_save_memory(fact: str, label: str = "", *, job_id: str = "_global", **_: Any) -> dict:
    return await asyncio.to_thread(save_memory, fact, label, job_id=job_id)


async def _async_recall_memory(query: str, *, job_id: str = "_global", **_: Any) -> list:
    return await asyncio.to_thread(recall_memory, query, job_id=job_id)


# ---------------------------------------------------------------------------
# 3. Async dispatch table
# ---------------------------------------------------------------------------

ASYNC_TOOL_DISPATCH: dict[str, Callable[..., Awaitable[Any]]] = {
    "web_search":    _async_web_search,
    "rag_query":     _async_rag_query,
    "calculate":     _async_calculate,
    "save_memory":   _async_save_memory,
    "recall_memory": _async_recall_memory,
}

# Which tools receive the injected job_id kwarg
_MEMORY_TOOLS = {"save_memory", "recall_memory"}
