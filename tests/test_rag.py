"""
tests/test_rag.py
Tests for the real RAG engine and rag_query public interface.
Uses a temporary knowledge base directory to stay fully isolated from production data.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT_A = """
Python 3.13 Performance Improvements

Python 3.13 introduces an experimental free-threaded mode without the GIL.
PEP 703 describes the design. The JIT compiler (PEP 744) is also included.
Benchmarks show 5 to 30 percent speedups on CPU-bound workloads.
"""

SAMPLE_TEXT_B = """
Large Language Models and RAG

Retrieval-Augmented Generation combines vector search with language models.
Documents are embedded into a vector space and retrieved by semantic similarity.
This allows models to answer questions about private or recent documents.
"""


@pytest.fixture(scope="module")
def engine_with_docs(tmp_path_factory):
    """
    Build a fresh RAGEngine using a temporary knowledge base directory.
    Returned engine has two documents indexed.
    """
    from tools.rag_engine import RAGEngine, _chunk_text

    tmp_kb = tmp_path_factory.mktemp("kb")
    (tmp_kb / "doc_a.txt").write_text(SAMPLE_TEXT_A, encoding="utf-8")
    (tmp_kb / "doc_b.txt").write_text(SAMPLE_TEXT_B, encoding="utf-8")

    from tools.rag_engine import _load_documents
    chunks = _load_documents(tmp_kb)

    engine = RAGEngine()
    engine.build(chunks)
    return engine


# ---------------------------------------------------------------------------
# RAGEngine tests
# ---------------------------------------------------------------------------

class TestRAGEngine:

    def test_builds_successfully(self, engine_with_docs):
        assert engine_with_docs.is_ready()
        assert len(engine_with_docs._chunks) > 0

    def test_search_returns_list(self, engine_with_docs):
        results = engine_with_docs.search("Python performance")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_structure(self, engine_with_docs):
        results = engine_with_docs.search("free-threaded mode", top_k=2)
        for r in results:
            assert "content" in r
            assert "source"  in r
            assert "score"   in r

    def test_score_between_0_and_1(self, engine_with_docs):
        """Cosine similarity scores should be in [0, 1] for normalised vectors."""
        results = engine_with_docs.search("GIL removal", top_k=3)
        for r in results:
            assert -0.01 <= r["score"] <= 1.01, f"Unexpected score: {r['score']}"

    def test_relevant_result_ranks_first(self, engine_with_docs):
        """A query about Python should rank a Python chunk above an LLM chunk."""
        results = engine_with_docs.search("Python JIT compiler PEP 744", top_k=3)
        assert results, "Expected at least one result"
        top_source = results[0]["source"]
        assert "doc_a" in top_source, (
            f"Expected Python doc to rank first, got: {top_source}"
        )

    def test_top_k_respected(self, engine_with_docs):
        results = engine_with_docs.search("language model", top_k=1)
        assert len(results) <= 1

    def test_search_on_empty_engine_returns_empty(self):
        from tools.rag_engine import RAGEngine
        empty = RAGEngine()
        assert empty.search("anything") == []

    def test_add_text_increases_chunk_count(self, engine_with_docs):
        before = len(engine_with_docs._chunks)
        added  = engine_with_docs.add_text(
            "This is a new document about reinforcement learning.",
            source="rag:extra.txt"
        )
        assert added > 0
        assert len(engine_with_docs._chunks) == before + added

    def test_persistence_roundtrip(self, tmp_path):
        """Index saved by one engine instance is loadable by another."""
        from tools.rag_engine import RAGEngine, _chunk_text

        chunks = _chunk_text(SAMPLE_TEXT_A, "rag:test.txt")

        eng1 = RAGEngine()
        # Override the index directory to tmp_path
        import tools.rag_engine as re_mod
        orig_index = re_mod._INDEX_FILE
        orig_meta  = re_mod._META_FILE
        orig_dir   = re_mod._INDEX_DIR

        re_mod._INDEX_DIR  = tmp_path / ".idx"
        re_mod._INDEX_FILE = re_mod._INDEX_DIR / "faiss.index"
        re_mod._META_FILE  = re_mod._INDEX_DIR / "metadata.pkl"

        try:
            eng1.build(chunks)

            eng2 = RAGEngine()
            loaded = eng2.load()
            assert loaded, "Expected load() to succeed"
            assert eng2.is_ready()
            assert len(eng2._chunks) == len(eng1._chunks)
        finally:
            re_mod._INDEX_DIR  = orig_dir
            re_mod._INDEX_FILE = orig_index
            re_mod._META_FILE  = orig_meta


# ---------------------------------------------------------------------------
# rag_query() public interface tests
# ---------------------------------------------------------------------------

class TestRagQueryInterface:

    def test_empty_query_raises(self):
        from tools.rag_query import rag_query
        with pytest.raises(ValueError, match="non-empty"):
            rag_query("")

    def test_whitespace_query_raises(self):
        from tools.rag_query import rag_query
        with pytest.raises(ValueError, match="non-empty"):
            rag_query("   ")

    def test_returns_list(self, engine_with_docs):
        """Patch get_engine so rag_query uses our fixture engine."""
        with patch("tools.rag_query.get_engine", return_value=engine_with_docs):
            from tools.rag_query import rag_query
            result = rag_query("Python threading")
            assert isinstance(result, list)

    def test_top_k_clamping(self, engine_with_docs):
        with patch("tools.rag_query.get_engine", return_value=engine_with_docs):
            from tools.rag_query import rag_query
            result = rag_query("Python", top_k=0)   # clamped to 1
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Chunking unit tests (no embeddings needed)
# ---------------------------------------------------------------------------

class TestChunking:

    def test_short_text_single_chunk(self):
        from tools.rag_engine import _chunk_text
        chunks = _chunk_text("Short text.", "rag:test.txt")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short text."
        assert chunks[0]["source"]  == "rag:test.txt"

    def test_long_text_multiple_chunks(self):
        from tools.rag_engine import CHUNK_SIZE, _chunk_text
        long_text = "a" * (CHUNK_SIZE * 3)
        chunks = _chunk_text(long_text, "rag:long.txt")
        assert len(chunks) > 1

    def test_all_chunks_have_source(self):
        from tools.rag_engine import _chunk_text
        chunks = _chunk_text(SAMPLE_TEXT_A, "rag:python.txt")
        for c in chunks:
            assert c["source"] == "rag:python.txt"

    def test_empty_text_no_chunks(self):
        from tools.rag_engine import _chunk_text
        chunks = _chunk_text("   ", "rag:empty.txt")
        assert chunks == []
