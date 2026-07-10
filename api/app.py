"""
api/app.py — Phase 4 FastAPI service
Exposes two endpoints:

  POST /research
    Body:    {"question": "..."}
    Returns: 202 {"job_id": "<uuid>", "status": "pending"}
    Effect:  Launches run_agent() as an asyncio background task.
             Returns immediately — the connection closes in < 50 ms.

  GET /status/{job_id}
    Returns: {"status": "pending"}
               {"status": "done",   "result": <ResearchReport as dict>}
               {"status": "failed", "detail": "..."}
             404 if job_id is unknown.

Job store:
  In-memory dict keyed by UUID string.
  Correct for single-process deployment; document the limitation explicitly.
  For multi-process or persistent storage: replace with Redis / a database.

Memory isolation:
  Each job passes its job_id into run_agent(), which threads it through to
  save_memory / recall_memory.  Two concurrent jobs cannot share facts.
  clear_job_memory(job_id) is called when the job finishes to prevent
  unbounded store growth.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agent.report_schema import AgentFailure, ResearchReport
from agent.runner        import run_agent
from api.models          import JobAccepted, JobStatus, ResearchRequest
from tools.memory        import clear_job_memory

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Research Agent API",
    description=(
        "Async research agent service. POST a question, get a job_id, "
        "then poll GET /status/{job_id} until done."
    ),
    version="4.0.0",
)

# ---------------------------------------------------------------------------
# In-memory job store
# NOTE: Does not survive process restart.  Replace with Redis for production.
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

async def _run_job(job_id: str, question: str) -> None:
    """
    Execute the research agent and write the result into _jobs[job_id].
    Called as an asyncio background task — the HTTP response has already
    been sent by the time this starts executing.
    """
    log = logging.getLogger("agent")
    log.info("[job=%s] Background task started", job_id)

    try:
        result = await run_agent(question, job_id=job_id)
    except Exception as exc:  # noqa: BLE001
        log.error("[job=%s] Uncaught exception in run_agent: %s", job_id, exc)
        _jobs[job_id] = {"status": "failed", "detail": f"Internal error: {exc}"}
        return
    finally:
        # Always release the per-job memory store when the job finishes
        clear_job_memory(job_id)

    if isinstance(result, ResearchReport):
        _jobs[job_id] = {"status": "done", "result": result.model_dump()}
        log.info("[job=%s] Done — report has %d findings", job_id, len(result.findings))
    elif isinstance(result, AgentFailure):
        _jobs[job_id] = {"status": "failed", "detail": result.detail}
        log.warning("[job=%s] Failed — reason: %s | %s", job_id, result.reason, result.detail)
    else:
        _jobs[job_id] = {"status": "failed", "detail": "Unexpected result type from run_agent"}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/research",
    status_code=202,
    response_model=JobAccepted,
    summary="Submit a research question",
    response_description="Job accepted; poll /status/{job_id} for results.",
)
async def post_research(body: ResearchRequest) -> JobAccepted:
    """
    Accept a research question and immediately return a job_id.
    The agent runs as a background asyncio task.
    """
    job_id = str(uuid4())
    _jobs[job_id] = {"status": "pending"}

    # asyncio.create_task schedules the coroutine without blocking the response
    asyncio.create_task(_run_job(job_id, body.question))

    return JobAccepted(job_id=job_id, status="pending")


@app.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Check the status of a research job",
    responses={404: {"description": "Job not found"}},
)
async def get_status(job_id: str) -> JobStatus:
    """
    Return the current status of a previously submitted research job.

    - **pending** — still running, check back later
    - **done** — completed; `result` contains the full ResearchReport
    - **failed** — agent could not complete; `detail` explains why
    """
    entry = _jobs.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatus(**entry)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok", "jobs": len(_jobs)}
