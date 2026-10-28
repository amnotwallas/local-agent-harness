import pytest

from agent_harness.agents.workflows import WORKFLOW_NAMES, build_workflow_plan


def test_reusable_workflows_select_tools_from_task_text() -> None:
    plan = build_workflow_plan("service_diagnosis", "Investigate payments-api latency")

    assert [(call.name, call.arguments) for call in plan] == [
        ("get_service_health", {"service_name": "payments-api"}),
        (
            "get_recent_logs",
            {"service_name": "payments-api", "severity": "ERROR", "limit": 20},
        ),
    ]


def test_workflows_are_named_application_capabilities() -> None:
    assert "service_health" in WORKFLOW_NAMES
    assert "compare_services" in WORKFLOW_NAMES
    assert "incident_cause_review" in WORKFLOW_NAMES


def test_workflow_rejects_missing_identifiers() -> None:
    with pytest.raises(ValueError, match="named service"):
        build_workflow_plan("service_health", "Check system health")
