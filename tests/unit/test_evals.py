import pytest

from agent_harness.agents.runtime import AgentResult, ToolExecution
from agent_harness.evals.runner import evaluate, score_result
from agent_harness.evals.scenarios import ExpectedCall, load_scenarios


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


def test_allowed_investigative_tool_call_does_not_fail_selection() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "elevated-http-errors")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="payments-api has an elevated error rate",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "payments-api"},
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_recent_logs",
                arguments={"service_name": "payments-api", "severity": "ERROR", "limit": 20},
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=3,
    )

    scored = score_result(scenario, result)

    assert scored["tool_selection_accuracy"] is True
    assert scored["task_success"] is True


def test_semantic_fact_equivalents_count_without_claiming_truth() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "incomplete-logs")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="No recent logs were found; that does not prove no errors occurred.",
        tool_history=[
            ToolExecution(
                tool_name="get_recent_logs",
                arguments={"service_name": "orders-api", "severity": "ERROR", "limit": 20},
                arguments_valid=True,
            )
        ],
        total_execution_time_ms=1,
        model_calls=2,
    )

    scored = score_result(scenario, result)

    assert all(value == "covered" for value in scored["required_facts"].values())
    assert scored["unsupported_claims"]["no errors occurred"] == "unverified"
    assert scored["task_success"] is True


def test_semantic_equivalents_cover_retrieval_failure_and_uncertain_cause() -> None:
    scenarios = {s.test_id: s for s in load_scenarios()}
    missing = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="I cannot retrieve the service health information for catalog-api.",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "catalog-api"},
                error="Tool execution failed",
                arguments_valid=True,
            )
        ],
        total_execution_time_ms=1,
        model_calls=2,
    )
    ambiguous = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer=(
            "The evidence indicates a possible cause, but it is possible that the database "
            "caused the incident."
        ),
        tool_history=[
            ToolExecution(
                tool_name="get_incident_details",
                arguments={"incident_id": "INC-1042"},
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "database"},
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=3,
    )

    assert (
        score_result(scenarios["missing-evidence"], missing)["required_facts"]["unknown"]
        == "covered"
    )
    assert (
        score_result(scenarios["ambiguous-cause"], ambiguous)["required_facts"]["cannot confirm"]
        == "covered"
    )


def test_unlisted_extra_tool_call_fails_selection() -> None:
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
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_incident_details",
                arguments={"incident_id": "INC-1042"},
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=2,
    )

    scored = score_result(scenario, result)

    assert scored["tool_selection_accuracy"] is False
    assert scored["task_success"] is False


def test_required_tool_calls_must_remain_in_order() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "ambiguous-cause")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="Evidence is incomplete; cannot confirm the cause.",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "database"},
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_incident_details",
                arguments={"incident_id": "INC-1042"},
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=3,
    )

    assert score_result(scenario, result)["tool_selection_accuracy"] is False


def test_forbidden_call_fails_even_if_it_is_also_allowed() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "healthy-service").model_copy(
        update={
            "allowed_tool_calls": [
                ExpectedCall(name="get_incident_details", arguments={"incident_id": "INC-1042"})
            ],
            "forbidden_tool_calls": [
                ExpectedCall(name="get_incident_details", arguments={"incident_id": "INC-1042"})
            ],
        }
    )
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="orders-api is healthy",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "orders-api"},
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_incident_details",
                arguments={"incident_id": "INC-1042"},
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=2,
    )

    assert score_result(scenario, result)["tool_selection_accuracy"] is False


def test_allowed_log_lookup_after_health_tool_error_is_not_a_selection_failure() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "tool-failure")
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="The health lookup failed; no additional logs were found.",
        tool_history=[
            ToolExecution(
                tool_name="get_service_health",
                arguments={"service_name": "unknown-api"},
                error="Tool execution failed",
                arguments_valid=True,
            ),
            ToolExecution(
                tool_name="get_recent_logs",
                arguments={"service_name": "unknown-api", "limit": 20},
                result=[],
                arguments_valid=True,
            ),
        ],
        total_execution_time_ms=1,
        model_calls=3,
    )

    assert score_result(scenario, result)["tool_selection_accuracy"] is True
