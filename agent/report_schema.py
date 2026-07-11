"""
agent/report_schema.py
Pydantic models that define the validated final output of the research agent.
Every claim must carry a source and a confidence level — enforced at runtime.
"""
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, field_validator, model_validator


class ResearchFinding(BaseModel):
    """A single cited claim from the research process."""

    claim: str
    source: str
    confidence: Literal["high", "low"]

    @field_validator("claim")
    @classmethod
    def claim_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("claim must be a non-empty string")
        return v.strip()

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source must be a non-empty string")
        return v.strip()


class ResearchReport(BaseModel):
    """The complete, validated output of a research agent run."""

    topic:    str
    findings: List[ResearchFinding]
    summary:  str

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("topic must be a non-empty string")
        return v.strip()

    @field_validator("findings")
    @classmethod
    def at_least_one_finding(cls, v: List[ResearchFinding]) -> List[ResearchFinding]:
        if not v:
            raise ValueError("findings must contain at least one ResearchFinding")
        return v

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("summary must be a non-empty string")
        return v.strip()


class AgentFailure(BaseModel):
    """Returned when the agent cannot produce a valid ResearchReport."""

    reason: str
    detail: str = ""
