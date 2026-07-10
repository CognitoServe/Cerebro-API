"""
tests/test_agent_e2e.py — Phase 4
End-to-end test of the full async research agent.

Uses unittest.mock to patch AsyncOpenAI so tests:
  - Run without a real API key
  - Are fast and deterministic
  - Still exercise the full async runner.py logic

Sequence mocked:
  Iteration 1 → tool call: web_search("Python 3.13 performance improvements")
  Iteration 2 → tool call: calculate("2**10 + 100")
  Iteration 3 → final answer JSON

Live e2e test (gated by AGENT_E2E_LIVE=1):
  Runs the real AsyncOpenAI client and asserts a valid ResearchReport is returned.
"""
from __future__ import annotations

import json
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.report_schema import AgentFailure, ResearchReport
from agent.runner        import run_agent, MAX_ITERATIONS

# ---------------------------------------------------------------------------
# Helpers to build mock AsyncOpenAI response objects
# ---------------------------------------------------------------------------

def _make_tool_call(call_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name      = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_tool_call_response(tool_calls: list) -> MagicMock:
    msg            = MagicMock()
    msg.content    = None
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }
    choice           = MagicMock()
    choice.message   = msg
    response         = MagicMock()
    response.choices = [choice]
    response.usage   = None
    return response


def _make_text_response(content: str) -> MagicMock:
    msg            = MagicMock()
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
# Final answer payload
# ---------------------------------------------------------------------------

FINAL_JSON = {
    "topic": "Python 3.13 performance improvements and 2**10 + 100",
    "findings": [
        {
            "claim":      "Python 3.13 introduced an experimental free-threaded (no-GIL) mode.",
            "source":     "https://docs.python.org/3.13/whatsnew/3.13.html",
            "confidence": "high",
        },
        {
            "claim":      "Python 3.13 includes an experimental JIT compiler (PEP 744).",
            "source":     "https://peps.python.org/pep-0744/",
            "confidence": "high",
        },
        {
            "claim":      "2**10 + 100 equals 1124.",
            "source":     "calculate",
            "confidence": "high",
        },
    ],
    "summary": (
        "Python 3.13 is a major release focused on concurrency and speed: "
        "it experimentally removes the GIL and adds a JIT compiler. "
        "Additionally, 2**10 + 100 = 1124."
    ),
}

FINAL_ANSWER_TEXT = f"```json\n{json.dumps(FINAL_JSON, indent=2)}\n```"


# ---------------------------------------------------------------------------
# Mocked e2e tests
# ---------------------------------------------------------------------------

class TestAgentE2EMocked:

    @patch("agent.runner.AsyncOpenAI")
    def test_full_run_produces_valid_report(self, mock_openai_cls):
        """Full pipeline with mocked LLM: 3-iteration run → valid ResearchReport."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # AsyncOpenAI uses async context; mock the coroutine chain
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_tool_call_response([
                _make_tool_call("call_001", "web_search",
                                {"query": "Python 3.13 performance improvements"})
            ]),
            _make_tool_call_response([
                _make_tool_call("call_002", "calculate", {"expression": "2**10 + 100"})
            ]),
            _make_text_response(FINAL_ANSWER_TEXT),
        ])

        result = asyncio.run(
            run_agent(
                "What are the main performance improvements in Python 3.13 "
                "and what is 2**10 + 100?",
                job_id="test-e2e-001",
            )
        )

        assert isinstance(result, ResearchReport)
        assert len(result.findings) == 3
        assert "1124" in result.summary

        for finding in result.findings:
            assert finding.source
            assert finding.confidence in ("high", "low")

        call_count = mock_client.chat.completions.create.call_count
        assert 2 <= call_count < MAX_ITERATIONS

    @patch("agent.runner.AsyncOpenAI")
    def test_iteration_cap_returns_failure(self, mock_openai_cls):
        """If the LLM loops indefinitely, return AgentFailure after MAX_ITERATIONS."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_tool_call_response([
                _make_tool_call("call_x", "recall_memory", {"query": "anything"})
            ])
        )

        result = asyncio.run(run_agent("A question that never resolves", job_id="test-cap"))
        assert isinstance(result, AgentFailure)
        assert result.reason == "max_iterations_exceeded"

    @patch("agent.runner.AsyncOpenAI")
    def test_malformed_json_returns_failure(self, mock_openai_cls):
        """If LLM returns a JSON block that fails schema validation → AgentFailure."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # Missing 'findings'
        bad_json = '```json\n{"topic": "test", "summary": "ok"}\n```'
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_text_response(bad_json)
        )

        result = asyncio.run(run_agent("Any question", job_id="test-malformed"))
        assert isinstance(result, AgentFailure)
        assert result.reason == "output_validation_failed"

    @patch("agent.runner.AsyncOpenAI")
    def test_empty_question_returns_failure(self, mock_openai_cls):
        """Empty question must be rejected before any LLM call is made."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        result = asyncio.run(run_agent(""))
        assert isinstance(result, AgentFailure)
        assert result.reason == "llm_error"
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Live e2e test (AGENT_E2E_LIVE=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("AGENT_E2E_LIVE") != "1",
    reason="Set AGENT_E2E_LIVE=1 to run live tests against the real OpenAI API",
)
def test_live_e2e():
    """Real end-to-end run against the live OpenAI / OpenRouter API."""
    from dotenv import load_dotenv
    load_dotenv()

    result = asyncio.run(
        run_agent(
            "What are the main performance improvements in Python 3.13 "
            "and what is 2**10 + 100?",
            job_id="live-e2e",
        )
    )

    assert isinstance(result, ResearchReport), f"Expected ResearchReport, got: {result}"
    assert len(result.findings) >= 2
    assert any(f.source for f in result.findings)
