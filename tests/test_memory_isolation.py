"""
tests/test_memory_isolation.py — Phase 4, Option B
Verifies that two concurrent jobs cannot see each other's saved facts.

This test directly exercises tools/memory.py's per-job_id store,
not the full agent — isolation is a property of the store, not the LLM.

Two strategies tested:
  1. Sequential saves + recalls with different job_ids (unit-style).
  2. asyncio.gather: two coroutines each save a fact and then recall —
     proves the lock and store partitioning work under concurrency.
"""
from __future__ import annotations

import asyncio
import pytest

from tools.memory import (
    save_memory,
    recall_memory,
    clear_job_memory,
    memory_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOB_A = "test-job-alpha"
JOB_B = "test-job-beta"


def _cleanup() -> None:
    clear_job_memory(JOB_A)
    clear_job_memory(JOB_B)


# ---------------------------------------------------------------------------
# 1. Unit-style: sequential saves, isolated recall
# ---------------------------------------------------------------------------

class TestMemoryIsolationSequential:

    def setup_method(self) -> None:
        _cleanup()

    def teardown_method(self) -> None:
        _cleanup()

    def test_job_a_cannot_see_job_b_facts(self) -> None:
        """Facts saved under job_b must not appear in job_a recall."""
        save_memory("JavaScript is a scripting language", label="js", job_id=JOB_B)
        results_a = recall_memory("scripting language", job_id=JOB_A)
        assert results_a == [], (
            f"job_a recalled job_b's fact — isolation failed: {results_a}"
        )

    def test_job_b_cannot_see_job_a_facts(self) -> None:
        """Facts saved under job_a must not appear in job_b recall."""
        save_memory("Python is used for data science", label="py", job_id=JOB_A)
        results_b = recall_memory("data science", job_id=JOB_B)
        assert results_b == [], (
            f"job_b recalled job_a's fact — isolation failed: {results_b}"
        )

    def test_each_job_recalls_only_its_own_fact(self) -> None:
        """
        Core isolation proof:
          job_a saves Fact-A, job_b saves Fact-B.
          job_a recall returns only Fact-A.
          job_b recall returns only Fact-B.
        """
        save_memory("Python 3.13 removed the GIL", label="py313", job_id=JOB_A)
        save_memory("Rust has zero-cost abstractions", label="rust", job_id=JOB_B)

        results_a = recall_memory("GIL", job_id=JOB_A)
        results_b = recall_memory("zero-cost", job_id=JOB_B)

        assert len(results_a) == 1, f"Expected 1 result for job_a, got: {results_a}"
        assert "GIL" in results_a[0]["fact"], f"Wrong fact returned for job_a: {results_a}"

        assert len(results_b) == 1, f"Expected 1 result for job_b, got: {results_b}"
        assert "Rust" in results_b[0]["fact"], f"Wrong fact returned for job_b: {results_b}"

        # Confirm no cross-contamination
        cross_a = recall_memory("Rust", job_id=JOB_A)
        cross_b = recall_memory("GIL",  job_id=JOB_B)

        assert cross_a == [], f"job_a incorrectly recalled job_b's Rust fact: {cross_a}"
        assert cross_b == [], f"job_b incorrectly recalled job_a's GIL fact: {cross_b}"

    def test_memory_size_is_independent_per_job(self) -> None:
        """memory_size() must count only entries for that specific job."""
        save_memory("Fact 1", job_id=JOB_A)
        save_memory("Fact 2", job_id=JOB_A)
        save_memory("Fact 3", job_id=JOB_B)

        assert memory_size(job_id=JOB_A) == 2
        assert memory_size(job_id=JOB_B) == 1

    def test_clear_job_memory_does_not_affect_other_jobs(self) -> None:
        """Clearing job_b's store must leave job_a's facts intact."""
        save_memory("Important Python fact", job_id=JOB_A)
        save_memory("Disposable fact",       job_id=JOB_B)

        clear_job_memory(JOB_B)

        # job_a still has its fact
        assert memory_size(job_id=JOB_A) == 1
        # job_b is gone
        assert memory_size(job_id=JOB_B) == 0


# ---------------------------------------------------------------------------
# 2. Concurrent: asyncio.gather — proves isolation under true concurrency
# ---------------------------------------------------------------------------

class TestMemoryIsolationConcurrent:
    """
    Run two coroutines simultaneously under asyncio.gather.
    Each saves a distinct, semantically unique fact then recalls it back.
    Neither may see the other's fact.
    """

    def setup_method(self) -> None:
        _cleanup()

    def teardown_method(self) -> None:
        _cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_jobs_do_not_cross_pollinate(self) -> None:
        """
        The definitive concurrent isolation test.
        Both coroutines run concurrently (via asyncio.gather) and each must
        recall only its own fact — no cross-contamination allowed.
        """

        async def job_alpha() -> list[dict]:
            # save_memory and recall_memory are sync; run in thread pool
            await asyncio.to_thread(
                save_memory,
                "Neural networks use gradient descent for training",
                "ml_fact",
                job_id=JOB_A,
            )
            # Small yield to allow job_beta to run concurrently
            await asyncio.sleep(0)
            return await asyncio.to_thread(
                recall_memory, "gradient descent", job_id=JOB_A
            )

        async def job_beta() -> list[dict]:
            await asyncio.to_thread(
                save_memory,
                "Kubernetes orchestrates containerised workloads",
                "k8s_fact",
                job_id=JOB_B,
            )
            await asyncio.sleep(0)
            return await asyncio.to_thread(
                recall_memory, "containerised", job_id=JOB_B
            )

        results_a, results_b = await asyncio.gather(job_alpha(), job_beta())

        # Each job should get exactly its own fact back
        assert len(results_a) == 1, (
            f"job_alpha expected 1 result, got {len(results_a)}: {results_a}"
        )
        assert "gradient descent" in results_a[0]["fact"].lower(), (
            f"job_alpha got wrong fact: {results_a[0]['fact']}"
        )

        assert len(results_b) == 1, (
            f"job_beta expected 1 result, got {len(results_b)}: {results_b}"
        )
        assert "kubernetes" in results_b[0]["fact"].lower(), (
            f"job_beta got wrong fact: {results_b[0]['fact']}"
        )

        # Neither job should see the other's fact
        leaked_to_a = await asyncio.to_thread(recall_memory, "kubernetes",   job_id=JOB_A)
        leaked_to_b = await asyncio.to_thread(recall_memory, "gradient",     job_id=JOB_B)

        assert leaked_to_a == [], (
            f"ISOLATION FAILURE: job_alpha saw job_beta's k8s fact: {leaked_to_a}"
        )
        assert leaked_to_b == [], (
            f"ISOLATION FAILURE: job_beta saw job_alpha's ML fact: {leaked_to_b}"
        )

    @pytest.mark.asyncio
    async def test_many_concurrent_jobs_no_cross_contamination(self) -> None:
        """
        Stress test: 5 concurrent jobs each save a unique topic, then each
        recalls its own topic and must not see any other job's topic.
        """
        topics = {
            "job-stress-0": ("quantum computing uses qubits",      "qubit"),
            "job-stress-1": ("blockchain stores data in blocks",   "blockchain"),
            "job-stress-2": ("DNA encodes genetic information",    "genetic"),
            "job-stress-3": ("HTTP is a stateless protocol",       "stateless"),
            "job-stress-4": ("SQL databases use relational tables","relational"),
        }

        async def worker(job_id: str, fact: str, recall_key: str) -> tuple[str, list]:
            await asyncio.to_thread(save_memory, fact, job_id=job_id)
            await asyncio.sleep(0)
            results = await asyncio.to_thread(recall_memory, recall_key, job_id=job_id)
            return job_id, results

        outputs = await asyncio.gather(
            *[worker(jid, fact, key) for jid, (fact, key) in topics.items()]
        )

        for job_id, results in outputs:
            assert len(results) == 1, (
                f"[{job_id}] Expected exactly 1 result, got {len(results)}: {results}"
            )
            expected_fact, _ = topics[job_id]
            assert results[0]["fact"] == expected_fact, (
                f"[{job_id}] Wrong fact returned: {results[0]['fact']!r} "
                f"(expected: {expected_fact!r})"
            )

        # Cleanup stress jobs
        for jid in topics:
            clear_job_memory(jid)
