"""
api/models.py
Pydantic request / response models for the FastAPI service.
Kept separate from agent/report_schema.py to allow the API contract to evolve
independently of the internal agent models.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator


class ResearchRequest(BaseModel):
    """Body for POST /research."""

    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must be a non-empty string")
        return v.strip()


class JobAccepted(BaseModel):
    """202 response body for POST /research."""

    job_id: str
    status: Literal["pending"] = "pending"


class JobStatus(BaseModel):
    """
    Response body for GET /status/{job_id}.

    status values:
      "pending" — job is still running
      "done"    — job completed successfully; result contains the ResearchReport
      "failed"  — job failed; detail explains why
    """

    status: Literal["pending", "done", "failed"]
    result: Optional[dict[str, Any]] = None   # ResearchReport serialised as dict
    detail: Optional[str]            = None   # AgentFailure.detail on "failed"
