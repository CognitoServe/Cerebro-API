"""
agent/runner.py — Phase 4
Fully async research agent loop.

Changes from Phase 3:
  - OpenAI → AsyncOpenAI; every LLM call is awaited.
  - Multiple tool calls in one LLM turn are dispatched concurrently via
    asyncio.gather, not sequentially.
  - LLM call wrapped with tenacity exponential backoff (handles 429 / 5xx).
  - job_id threaded from the API layer through to memory tools so each job
    has its own isolated memory store (Option B).
  - Named 'agent' logger: timestamped, propagate=False, orchestration-only.
  - Per-iteration and final total accumulation of token / cost from
    response.usage.

Architecture:
  run_agent(question, job_id) → ResearchReport | AgentFailure

Every iteration:
  1. Await LLM call (with tenacity retry)
  2. Accumulate usage (tokens + cost)
  3. If tool calls → asyncio.gather all tool executions concurrently
  4. If FINAL_ANSWER → parse JSON → validate ResearchReport
  5. Trace every step via the 'agent' logger
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Union

from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from agent.report_schema import AgentFailure, ResearchReport
from agent.tool_registry  import ASYNC_TOOL_DISPATCH, TOOL_SCHEMAS, _MEMORY_TOOLS
from agent.tracer         import print_separator, print_trace

# ---------------------------------------------------------------------------
# Named logger — timestamped, isolated, propagate=False
# ---------------------------------------------------------------------------

log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)
log.propagate = False

if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(_handler)

# ---------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ---------------------------------------------------------------------------

MAX_ITERATIONS: int = int(os.environ.get("AGENT_MAX_ITER", 10))
MODEL: str          = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous research agent. Your job is to answer research questions
thoroughly by using the tools available to you, then produce a structured report.

## Workflow
1. PLAN: Think step-by-step about what information you need.
2. RESEARCH: Use tools (web_search, rag_query, recall_memory, calculate) to
   gather facts. Prefer rag_query and recall_memory before web_search.
3. SAVE: Use save_memory to store key facts you will reference again.
4. SYNTHESIZE: Once you have enough information, produce your final answer.

## Final Answer Format
When you are ready to give your final answer, respond with ONLY a JSON code block
in this exact format — nothing before or after:

```json
{
  "topic": "<the original research question>",
  "findings": [
    {
      "claim": "<a specific factual claim>",
      "source": "<URL, rag:<doc>, memory, or agent_knowledge>",
      "confidence": "high"
    }
  ],
  "summary": "<one paragraph synthesising all findings>"
}
```

## Rules
- Every finding MUST have a non-empty source and a confidence of "high" or "low".
- Use confidence "high" only when the claim came from a tool result.
- Use confidence "low" only when using your general training knowledge.
- "high" confidence claims MUST NOT have source "agent_knowledge".
- Do not make up URLs. Use the exact URLs from web_search results.
- You will be stopped after a fixed number of iterations — be efficient.
"""

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the first ```json ... ``` block out of an LLM response."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# LLM call with tenacity retry
# ---------------------------------------------------------------------------

def _make_llm_caller(client: AsyncOpenAI, model_name: str):
    """
    Return an async callable that wraps the LLM call with tenacity retry.
    The decorator is applied here (not at module level) so it captures the
    per-request client/model without needing global state.
    """
    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call(messages: list[dict], tools: list[dict]) -> Any:
        return await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )

    return _call


# ---------------------------------------------------------------------------
# Concurrent tool execution
# ---------------------------------------------------------------------------

async def _execute_tool_async(
    tc: Any,
    job_id: str,
) -> tuple[str, str, str]:
    """
    Execute a single tool call asynchronously.

    Returns:
        (tool_call_id, tool_name, observation_json)
    """
    tool_name = tc.function.name
    try:
        tool_args: dict = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        tool_args = {}

    # Inject job_id transparently for memory tools — the LLM schema does not
    # expose job_id, so this injection is invisible to the model.
    if tool_name in _MEMORY_TOOLS:
        tool_args["job_id"] = job_id

    fn = ASYNC_TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return tc.id, tool_name, json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        result      = await fn(**tool_args)
        observation = json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        observation = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return tc.id, tool_name, observation


# ---------------------------------------------------------------------------
# Main async agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    question: str,
    job_id:   str = "_global",
) -> Union[ResearchReport, AgentFailure]:
    """
    Run the research agent on the given question.

    Args:
        question: The natural-language research question.
        job_id:   Unique identifier for this job, used to scope memory isolation.
                  Defaults to "_global" for CLI / test usage.

    Returns:
        A validated ResearchReport on success,
        or an AgentFailure on iteration exhaustion / validation failure.
    """
    if not question or not question.strip():
        return AgentFailure(reason="llm_error", detail="Question must not be empty.")

    # ── Client setup ──────────────────────────────────────────────────────────
    api_key    = os.environ.get("OPENAI_API_KEY", "")
    model_name = MODEL

    if api_key.startswith("sk-or-"):
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/CognitoServe/autonomous-research-agent",
                "X-Title":      "Autonomous Research Agent API",
            },
        )
        if model_name == "gpt-4o-mini":
            model_name = "openai/gpt-4o-mini"
        elif model_name == "gpt-4o":
            model_name = "openai/gpt-4o"
    else:
        client = AsyncOpenAI()

    llm_call = _make_llm_caller(client, model_name)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]

    # ── Usage accumulators ────────────────────────────────────────────────────
    total_tokens: int   = 0
    total_cost:   float = 0.0

    print_separator("RESEARCH AGENT START")
    print_trace(-1, "PLAN", f"Question: {question.strip()}")
    print_separator()
    log.info("[job=%s] Starting run — question: %s", job_id, question.strip())

    for iteration in range(MAX_ITERATIONS):

        # ── LLM call (with tenacity retry) ────────────────────────────────────
        try:
            response = await llm_call(messages, TOOL_SCHEMAS)
        except Exception as exc:  # noqa: BLE001
            log.error("[job=%s] LLM call failed on iter %02d: %s", job_id, iteration + 1, exc)
            return AgentFailure(reason="llm_error", detail=str(exc))

        # ── Accumulate token / cost usage ─────────────────────────────────────
        usage = getattr(response, "usage", None)
        if usage is not None:
            iter_tokens  = getattr(usage, "total_tokens", 0) or 0
            iter_cost    = getattr(usage, "cost", None) or 0.0
            total_tokens += iter_tokens
            total_cost   += iter_cost
            log.debug(
                "[job=%s] Iter %02d | tokens: %d | running: %d tokens / $%.6f",
                job_id, iteration + 1, iter_tokens, total_tokens, total_cost,
            )

        message = response.choices[0].message

        # ── Concurrent tool call branch ───────────────────────────────────────
        if message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))

            # Log each action before dispatching
            for tc in message.tool_calls:
                try:
                    args_str = tc.function.arguments
                except Exception:
                    args_str = "{}"
                print_trace(
                    iteration, "ACTION",
                    f"{tc.function.name}({args_str})"
                )
                log.info(
                    "[job=%s] Iter %02d | ACTION: %s(%s)",
                    job_id, iteration + 1, tc.function.name, args_str,
                )

            # Fire all tool calls concurrently — this is the key Phase 4 upgrade
            results = await asyncio.gather(
                *[_execute_tool_async(tc, job_id) for tc in message.tool_calls],
                return_exceptions=True,
            )

            for item in results:
                if isinstance(item, Exception):
                    tc_id, tool_name, observation = "unknown", "unknown", json.dumps({"error": str(item)})
                else:
                    tc_id, tool_name, observation = item

                print_trace(
                    iteration, "OBSERVATION",
                    observation[:400] + ("…" if len(observation) > 400 else "")
                )
                log.debug(
                    "[job=%s] Iter %02d | OBSERVATION(%s): %s",
                    job_id, iteration + 1, tool_name,
                    observation[:300] + ("…" if len(observation) > 300 else ""),
                )

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc_id,
                    "content":      observation,
                })

            continue  # Back to LLM with all tool results in context

        # ── Final answer branch ───────────────────────────────────────────────
        content = message.content or ""
        if "```json" in content or content.strip().startswith("{"):
            print_trace(iteration, "FINAL", "LLM returned final answer — parsing…")
            log.info("[job=%s] Iter %02d | FINAL: parsing JSON report…", job_id, iteration + 1)

            raw = _extract_json(content)
            if raw is None:
                log.error("[job=%s] Could not extract JSON from LLM response", job_id)
                return AgentFailure(
                    reason="output_validation_failed",
                    detail="Could not extract JSON from LLM response.",
                )
            try:
                report = ResearchReport(**raw)
                print_separator("VALIDATED REPORT")
                log.info(
                    "[job=%s] Run complete — total tokens: %d | total cost: $%.6f",
                    job_id, total_tokens, total_cost,
                )
                return report
            except ValidationError as exc:
                log.error("[job=%s] Pydantic validation failed: %s", job_id, exc)
                return AgentFailure(reason="output_validation_failed", detail=str(exc))

        # ── Planning / thinking text ──────────────────────────────────────────
        print_trace(iteration, "PLAN", content[:300])
        log.debug("[job=%s] Iter %02d | PLAN: %s", job_id, iteration + 1, content[:300])
        messages.append({"role": "assistant", "content": content})

    else:
        # for-loop exhausted
        print_trace(MAX_ITERATIONS - 1, "GUARDRAIL",
                    f"Max iterations ({MAX_ITERATIONS}) reached — stopping.")
        log.warning(
            "[job=%s] GUARDRAIL — max iterations (%d). "
            "Total tokens: %d | total cost: $%.6f",
            job_id, MAX_ITERATIONS, total_tokens, total_cost,
        )
        return AgentFailure(
            reason="max_iterations_exceeded",
            detail=f"Agent did not finish within {MAX_ITERATIONS} iterations.",
        )
