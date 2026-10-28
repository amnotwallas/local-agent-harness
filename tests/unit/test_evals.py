import pytest

from agent_harness.agents.runtime import AgentResult, ToolExecution
from agent_harness.evals.runner import evaluate, score_result
from agent_harness.evals.scenarios import load_scenarios


def test_eval_dataset_has_ten_scenarios_with_explicit_assertions() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) >= 10
    assert all(
        scenario.test_id and scenario.task and scenario.expected_tool_calls is not None
        for scenario in scenarios
    )
    assert {
        "healthy",
        "latency",
        "errors",
        "database",
        "deployment",
        "conflicting",
        "missing",
        "incomplete",
        "tool_failure",
        "ambiguous",
    } <= {scenario.category for scenario in scenarios}


@pytest.mark.asyncio
async def test_mock_evaluation_completes_every_scenario() -> None:
    report = await evaluate(mode="mock")
    assert report["summary"]["task_successes"] == 10
    assert report["metadata"]["evaluation_scope"] == "harness_behavior"


def test_required_fact_presence_affects_success_without_claiming_truth() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "healthy-service")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="orders-api is stable",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "orders-api"},
                result={},
                arguments_valid=True,
            )
        ],
        total_execution_time_ms=1,
        model_calls=1,
        termination_reason="final_response",
    )

    scored = score_result(scenario, result)

    assert scored["required_facts"]["healthy"] == "uncovered"
    assert scored["unsupported_claims"]["database caused the issue"] == "unverified"
    assert scored["task_success"] is False


def test_invalid_response_schema_fails_task_success() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "healthy-service")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="orders-api is healthy",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "orders-api"},
                result={},
                arguments_valid=True,
            )
        ],
        total_execution_time_ms=1,
        model_calls=1,
        termination_reason="final_response",
    ).model_copy(update={"model_calls": "invalid"})

    assert score_result(scenario, result)["output_schema"] == "failed"
    assert score_result(scenario, result)["task_success"] is False
