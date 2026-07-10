"""
tests/test_api.py — Phase 4
Tests for the FastAPI endpoints using httpx's async test client.

Mocked tests (run by default, no API key required):
  - POST /research → 202 with job_id
  - POST /research with empty question → 422 validation error
  - GET /status/<unknown> → 404
  - GET /status/<pending job> → pending
  - GET /status after mock-completed job → done with ResearchReport
  - GET /health → 200

All agent runs are patched so no real LLM call is made.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from agent.report_schema import ResearchReport
from api.app import app, _jobs

# ---------------------------------------------------------------------------
# Helpers — build minimal mock LLM responses
# ---------------------------------------------------------------------------

def _make_final_json_response() -> MagicMock:
    """Mock response that returns a valid final-answer JSON block."""
    final_json = {
        "topic": "Test topic",
        "findings": [
            {"claim": "Test claim", "source": "https://example.com", "confidence": "high"}
        ],
        "summary": "Test summary.",
    }
    content = f"```json\n{json.dumps(final_json)}\n```"

    msg = MagicMock()
    msg.content    = content
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": content}

    choice           = MagicMock()
    choice.message   = msg
    response         = MagicMock()
    response.choices = [choice]
    response.usage   = None
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_jobs():
    """Ensure _jobs dict is clean before and after each test."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def async_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# POST /research
# ---------------------------------------------------------------------------

class TestPostResearch:

    @pytest.mark.asyncio
    async def test_returns_202_and_job_id(self, async_client):
        """POST /research must return 202 with a non-empty job_id."""
        async with async_client as client:
            resp = await client.post("/research", json={"question": "What is Python?"})
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["job_id"]           # non-empty
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_empty_question_returns_422(self, async_client):
        """Empty question must be rejected by Pydantic before reaching the agent."""
        async with async_client as client:
            resp = await client.post("/research", json={"question": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_whitespace_question_returns_422(self, async_client):
        async with async_client as client:
            resp = await client.post("/research", json={"question": "   "})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_question_field_returns_422(self, async_client):
        async with async_client as client:
            resp = await client.post("/research", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_job_is_registered_as_pending(self, async_client):
        """After POST, the job_id must immediately appear in the store as pending."""
        async with async_client as client:
            resp = await client.post("/research", json={"question": "Any question"})
        job_id = resp.json()["job_id"]
        assert job_id in _jobs
        assert _jobs[job_id]["status"] == "pending"


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

class TestGetStatus:

    @pytest.mark.asyncio
    async def test_unknown_job_returns_404(self, async_client):
        async with async_client as client:
            resp = await client.get("/status/nonexistent-job-id-xyz")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_pending_job_returns_pending(self, async_client):
        """A job that hasn't finished yet must return status='pending'."""
        _jobs["test-pending"] = {"status": "pending"}
        async with async_client as client:
            resp = await client.get("/status/test-pending")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_done_job_returns_report(self, async_client):
        """A completed job must return status='done' with the result dict."""
        report = ResearchReport(
            topic="Test",
            findings=[{"claim": "Claim", "source": "https://x.com", "confidence": "high"}],
            summary="Summary.",
        )
        _jobs["test-done"] = {"status": "done", "result": report.model_dump()}
        async with async_client as client:
            resp = await client.get("/status/test-done")
        body = resp.json()
        assert body["status"] == "done"
        assert body["result"]["topic"] == "Test"
        assert len(body["result"]["findings"]) == 1

    @pytest.mark.asyncio
    async def test_failed_job_returns_detail(self, async_client):
        """A failed job must return status='failed' with a detail string."""
        _jobs["test-failed"] = {"status": "failed", "detail": "Max iterations exceeded"}
        async with async_client as client:
            resp = await client.get("/status/test-failed")
        body = resp.json()
        assert body["status"] == "failed"
        assert "Max iterations" in body["detail"]

    @pytest.mark.asyncio
    async def test_full_flow_post_then_poll(self, async_client):
        """
        Integration: POST a question (with mocked agent), wait for the
        background task to finish, then poll GET /status and assert done.
        """
        with patch("api.app.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ResearchReport(
                topic="Test question",
                findings=[
                    {"claim": "A fact", "source": "https://example.com", "confidence": "high"}
                ],
                summary="A summary.",
            )

            async with async_client as client:
                post_resp = await client.post(
                    "/research", json={"question": "Test question"}
                )
                assert post_resp.status_code == 202
                job_id = post_resp.json()["job_id"]

                # Give the background task time to complete
                await asyncio.sleep(0.2)

                status_resp = await client.get(f"/status/{job_id}")

        body = status_resp.json()
        assert body["status"] == "done"
        assert body["result"]["topic"] == "Test question"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client):
        async with async_client as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
