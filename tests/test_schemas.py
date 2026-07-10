"""
tests/test_schemas.py
Pydantic schema validation tests for ResearchReport and AgentFailure.
"""
import pytest
from pydantic import ValidationError
from agent.report_schema import AgentFailure, ResearchFinding, ResearchReport


VALID_FINDING = {
    "claim":      "Python 3.13 removed the GIL experimentally.",
    "source":     "https://docs.python.org/3.13/",
    "confidence": "high",
}

VALID_REPORT = {
    "topic":    "Python 3.13",
    "findings": [VALID_FINDING],
    "summary":  "Python 3.13 is a major release.",
}


class TestResearchFinding:

    def test_valid_finding_parses(self):
        f = ResearchFinding(**VALID_FINDING)
        assert f.claim == VALID_FINDING["claim"]

    def test_empty_claim_raises(self):
        with pytest.raises(ValidationError, match="claim"):
            ResearchFinding(claim="", source="https://x.com", confidence="high")

    def test_empty_source_raises(self):
        with pytest.raises(ValidationError, match="source"):
            ResearchFinding(claim="A claim", source="", confidence="high")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValidationError):
            ResearchFinding(claim="A claim", source="https://x.com", confidence="medium")


class TestResearchReport:

    def test_valid_report_parses(self):
        r = ResearchReport(**VALID_REPORT)
        assert r.topic    == "Python 3.13"
        assert len(r.findings) == 1

    def test_empty_findings_raises(self):
        with pytest.raises(ValidationError, match="findings"):
            ResearchReport(topic="Test", findings=[], summary="ok")

    def test_empty_topic_raises(self):
        with pytest.raises(ValidationError, match="topic"):
            ResearchReport(topic="", findings=[VALID_FINDING], summary="ok")

    def test_empty_summary_raises(self):
        with pytest.raises(ValidationError, match="summary"):
            ResearchReport(topic="Test", findings=[VALID_FINDING], summary="")

    def test_high_confidence_agent_knowledge_raises(self):
        bad_finding = {**VALID_FINDING, "source": "agent_knowledge"}
        with pytest.raises(ValidationError):
            ResearchReport(topic="Test", findings=[bad_finding], summary="Summary.")

    def test_low_confidence_agent_knowledge_is_valid(self):
        low_finding = {**VALID_FINDING, "source": "agent_knowledge", "confidence": "low"}
        r = ResearchReport(topic="Test", findings=[low_finding], summary="Summary.")
        assert r.findings[0].confidence == "low"


class TestAgentFailure:

    def test_basic_failure(self):
        f = AgentFailure(reason="max_iterations_exceeded", detail="ran out")
        assert f.reason == "max_iterations_exceeded"
        assert f.detail == "ran out"

    def test_detail_defaults_to_empty_string(self):
        f = AgentFailure(reason="llm_error")
        assert f.detail == ""
