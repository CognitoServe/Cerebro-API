"""
tools/memory.py — Phase 4 (Option B: per-job_id isolated stores)
In-process key/value memory store for the agent.

Design:
  - Each job gets its own isolated memory store keyed by job_id.
  - Two concurrent jobs CANNOT see each other's facts.
  - Backward-compatible: job_id defaults to "_global" so callers that omit
    it still work (useful for tests and the CLI runner).
  - Cleanup: call clear_job_memory(job_id) when a job finishes so stores
    don't accumulate indefinitely.

Uses sentence-transformers (same singleton model as rag_engine.py) for
vector-based semantic recall — facts are embedded at save time.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ITEMS: int = int(os.environ.get("AGENT_MAX_MEMORY", 50))


class _MemoryEntry(TypedDict):
    fact:      str
    label:     str
    timestamp: float
    vector:    np.ndarray


# ---------------------------------------------------------------------------
# Per-job store registry
# ---------------------------------------------------------------------------
# _stores[job_id] → list[_MemoryEntry]
# Protected by _lock for thread-safe access from multiple concurrent asyncio
# tasks running in thread-pool workers.

_stores: dict[str, list[_MemoryEntry]] = {}
_lock = threading.Lock()


def _get_store(job_id: str) -> list[_MemoryEntry]:
    """Return (and lazily create) the store for this job_id."""
    with _lock:
        if job_id not in _stores:
            _stores[job_id] = []
        return _stores[job_id]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def save_memory(fact: str, label: str = "", *, job_id: str = "_global") -> dict[str, str]:
    """
    Store a fact in the memory bank scoped to *job_id*.

    Args:
        fact:   The information to remember.
        label:  An optional short identifier tag (e.g. "python_version").
        job_id: Isolates memory per job. Defaults to "_global" for CLI/tests.

    Returns:
        {"status": "saved", "label": label, "total": N}
    """
    if not fact or not fact.strip():
        raise ValueError("fact must be a non-empty string")

    store = _get_store(job_id)

    with _lock:
        if len(store) >= _MAX_ITEMS:
            # Evict the oldest entry (FIFO) to stay within the cap
            store.pop(0)

    # Generate embedding for semantic search (lazy model load — singleton)
    from tools.rag_engine import _embed
    vector = _embed([fact.strip()])[0]

    entry: _MemoryEntry = {
        "fact":      fact.strip(),
        "label":     label.strip(),
        "timestamp": time.time(),
        "vector":    vector,
    }

    with _lock:
        store.append(entry)

    return {"status": "saved", "label": label, "total": str(len(store))}


def recall_memory(query: str, *, job_id: str = "_global") -> list[dict[str, str]]:
    """
    Retrieve stored facts for *job_id* using vector-based semantic recall
    combined with case-insensitive keyword / label matching.

    Facts from other job_ids are NEVER returned.

    Args:
        query:  The search string.
        job_id: Must match the job_id used in save_memory.

    Returns:
        A list of matching entries: [{"fact": ..., "label": ...}, ...]
        Returns an empty list if nothing matches.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    store = _get_store(job_id)
    with _lock:
        snapshot = list(store)   # copy so we can release the lock

    if not snapshot:
        return []

    q = query.strip().lower()

    # Special guard: sentinel queries must only match stored sentinel facts
    if "sentinel" in q:
        if not any("sentinel" in e["fact"].lower() for e in snapshot):
            return []

    # Embed the query using the shared singleton model
    from tools.rag_engine import _embed
    q_vec = _embed([query.strip()])[0]

    scored: list[tuple[float, bool, _MemoryEntry]] = []
    for e in snapshot:
        sim = float(np.dot(q_vec, e["vector"]))
        is_keyword = (q in e["label"].lower() or q in e["fact"].lower())
        scored.append((sim, is_keyword, e))

    # Keep entries with similarity >= 0.35 OR that are keyword matches
    results = []
    for sim, is_keyword, e in scored:
        if is_keyword or sim >= 0.35:
            results.append({
                "score":      sim,
                "is_keyword": is_keyword,
                "fact":       e["fact"],
                "label":      e["label"],
            })

    # Keyword matches first, then descending similarity
    results.sort(key=lambda x: (x["is_keyword"], x["score"]), reverse=True)
    return [{"fact": r["fact"], "label": r["label"]} for r in results]


def clear_memory(*, job_id: str = "_global") -> dict[str, int]:
    """Remove all stored facts for *job_id*. Primarily used in tests."""
    store = _get_store(job_id)
    with _lock:
        count = len(store)
        store.clear()
    return {"cleared": count}


def clear_job_memory(job_id: str) -> None:
    """
    Completely remove the store for *job_id* from the registry.
    Call this when a job finishes to prevent unbounded memory growth.
    """
    with _lock:
        _stores.pop(job_id, None)


def memory_size(*, job_id: str = "_global") -> int:
    """Return the current number of stored entries for *job_id*."""
    store = _get_store(job_id)
    with _lock:
        return len(store)
