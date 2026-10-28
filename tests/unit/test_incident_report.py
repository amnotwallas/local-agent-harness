import pytest
from pydantic import ValidationError

from agent_harness.agents.incident_report import IncidentReport


def test_incident_report_requires_all_categories_and_rejects_extra_fields() -> None:
    raw = {
        "confirmed_observations": [
            {"statement": "The API returned 503.", "evidence_sources": ["get_service_health"]}
        ],
        "hypotheses": [],
        "missing_evidence": ["Recent logs"],
        "tool_failures": [],
    }

    parsed = IncidentReport.model_validate(raw, strict=True)

    assert parsed.confirmed_observations[0].statement == "The API returned 503."
    with pytest.raises(ValidationError):
        IncidentReport.model_validate({**raw, "unexpected": "field"}, strict=True)
    with pytest.raises(ValidationError):
        IncidentReport.model_validate({"confirmed_observations": []}, strict=True)
    with pytest.raises(ValidationError):
        IncidentReport.model_validate(
            {
                **raw,
                "confirmed_observations": [
                    {
                        "statement": "The API returned 503.",
                        "evidence_sources": ["get_service_health"],
                        "confidence": "certain",
                    }
                ],
            },
            strict=True,
        )


def test_incident_report_schema_is_a_strict_object_contract() -> None:
    schema = IncidentReport.model_json_schema()

    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "confirmed_observations",
        "hypotheses",
        "missing_evidence",
        "tool_failures",
    }
    assert schema["additionalProperties"] is False
