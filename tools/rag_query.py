"""
tools/rag_query.py
Public interface for querying the knowledge base.
Delegates to the real RAGEngine (sentence-transformers + FAISS).
"""
from __future__ import annotations

from tools.rag_engine import get_engine


def rag_query(query: str, top_k: int = 3) -> list[dict[str, str | float]]:
    """
    Query the local knowledge base for relevant document chunks.

    Args:
        query: The natural-language question to search for.
        top_k: Maximum number of chunks to return (1–10).

    Returns:
        A list of result dicts: [{"content": ..., "source": ..., "score": float}, ...]
        Returns an empty list if the knowledge base is empty.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    top_k = max(1, min(top_k, 10))
    engine = get_engine()
    return engine.search(query, top_k=top_k)
