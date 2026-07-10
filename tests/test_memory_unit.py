"""
tests/test_memory.py — Milestone 4c
Tests the in-process memory store.
"""
import pytest
from tools.memory import clear_memory, memory_size, recall_memory, save_memory


@pytest.fixture(autouse=True)
def reset_memory():
    """Clear memory before and after every test to keep them isolated."""
    clear_memory()
    yield
    clear_memory()


class TestSaveMemory:

    def test_save_returns_status(self):
        result = save_memory("Python was created by Guido van Rossum")
        assert result["status"] == "saved"

    def test_save_increments_count(self):
        save_memory("fact one")
        save_memory("fact two")
        assert memory_size() == 2

    def test_save_with_label(self):
        result = save_memory("Python 3.13 released in 2024", label="python_version")
        assert result["label"] == "python_version"

    def test_empty_fact_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            save_memory("")

    def test_whitespace_fact_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            save_memory("   ")


class TestRecallMemory:

    def test_recall_matches_content(self):
        save_memory("Python 3.13 released in October 2024")
        results = recall_memory("Python 3.13")
        assert len(results) == 1
        assert "Python 3.13" in results[0]["fact"]

    def test_recall_case_insensitive(self):
        save_memory("The GIL was removed experimentally in Python 3.13")
        results = recall_memory("gil")
        assert len(results) == 1

    def test_recall_by_label(self):
        save_memory("Python 3.13 release notes", label="python_release")
        results = recall_memory("python_release")
        assert len(results) >= 1

    def test_recall_no_match(self):
        save_memory("Python is great")
        results = recall_memory("JavaScript")
        assert results == []

    def test_recall_empty_query_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            recall_memory("")

    def test_recall_multiple_matches(self):
        save_memory("Python 3.13 free-threaded mode")
        save_memory("Python 3.13 JIT compiler added")
        results = recall_memory("Python 3.13")
        assert len(results) == 2


class TestMemoryCap:

    def test_cap_enforced(self):
        """Memory should never grow beyond MAX_MEMORY_ITEMS (default 50)."""
        import os
        max_items = int(os.environ.get("AGENT_MAX_MEMORY", 50))
        # Add max_items + 5 entries
        for i in range(max_items + 5):
            save_memory(f"fact number {i}")
        assert memory_size() <= max_items

    def test_oldest_evicted_first(self):
        """When cap is reached, the oldest fact is evicted (FIFO)."""
        import os
        max_items = int(os.environ.get("AGENT_MAX_MEMORY", 50))
        # Fill to the cap with a distinctive first fact
        save_memory("FIRST_FACT_SENTINEL")
        for i in range(max_items - 1):
            save_memory(f"filler fact {i}")
        # One more to trigger eviction
        save_memory("trigger eviction")
        results = recall_memory("FIRST_FACT_SENTINEL")
        assert results == [], "First fact should have been evicted"
