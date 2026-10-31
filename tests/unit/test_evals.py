import json
from pathlib import Path

import httpx
import pytest

from agent_harness.agents.incident_report import EvidenceStatement, IncidentReport
from agent_harness.agents.runtime import AgentResult, ExecutionStrategy, OutputMode, ToolExecution
from agent_harness.evals.reporting import render_summary
from agent_harness.evals.runner import EvaluationAPIError, evaluate, score_result
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


@pytest.mark.asyncio
async def test_live_evaluation_sends_each_scenario_to_running_api() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json=AgentResult(
                request_id="api-request",
                execution_id="api-execution",
                status="completed",
                final_answer="No incident found",
                tool_history=[],
                total_execution_time_ms=12,
                model_calls=1,
            ).model_dump(mode="json"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert requests[0].url.path == "/health"
    assert requests[0].extensions["timeout"]["read"] == 5
    posts = requests[1:]
    assert len(posts) == len(load_scenarios())
    assert all(request.url.path == "/agents/run" for request in posts)
    assert posts[0].extensions["timeout"]["read"] == 3610
    assert posts[0].extensions["timeout"]["connect"] == 5
    assert posts[0].extensions["timeout"]["write"] == 5
    assert json.loads(posts[0].content)["task"] == load_scenarios()[0].task
    assert json.loads(posts[0].content)["strategy"] == "autonomous"
    assert report["metadata"]["tool_data_scope"] == "api_tools"
    assert report["metadata"]["inference"] == {
        "source": "api_server",
        "api_base_url": "http://api.test",
    }
    assert report["metadata"]["configuration"] == {"trials": 1}
    assert report["metadata"]["response_format_config"] == {
        "mode": "free_text",
        "source": "api_server",
        "schema": "not_introspected",
    }
    assert len(report["results"]) == len(load_scenarios())
    assert report["results"][0]["runtime_latency_ms"] == 12
    assert report["results"][0]["latency_ms"] > 0
    assert report["summary"]["mean_runtime_latency_ms"] == 12
    report["metadata"]["model"] = "/private/models/local-model.gguf"
    assert "Modelo: local-model.gguf" in render_summary(report, Path("reports/latest.json"))
    assert "Ejecución media del servidor" in render_summary(report, Path("reports/latest.json"))
    assert "causas: selección de herramientas" in render_summary(
        report, Path("reports/latest.json")
    )


@pytest.mark.asyncio
async def test_terminal_summary_is_readable_and_keeps_per_scenario_results() -> None:
    report = await evaluate(mode="mock")

    summary = render_summary(report, Path("reports/latest.json"))

    assert "10/10" in summary
    assert "Escenario" in summary
    assert "healthy-service" in summary
    assert "Cobertura" in summary
    assert "Latencia" in summary
    assert "No solicitado: 10" in summary
    assert "reports/latest.json" in summary
    assert '"by_scenario"' not in summary


@pytest.mark.asyncio
async def test_live_evaluation_records_model_http_failures_in_report() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(502, headers={"x-request-id": "api-123"}, json={"error": "failed"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["summary"]["infrastructure_errors"] == 10
    assert report["results"][0]["termination_reason"] == "api_http_502"
    assert report["results"][0]["api_status_code"] == 502
    assert report["results"][0]["latency_ms"] > 0
    assert "causas: API HTTP 502" in render_summary(report, Path("reports/latest.json"))


@pytest.mark.asyncio
async def test_live_evaluation_records_non_json_server_errors() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, text="upstream error page")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["summary"]["infrastructure_errors"] == 10
    assert report["results"][0]["termination_reason"] == "api_http_500"


@pytest.mark.asyncio
async def test_live_evaluation_checks_fields_present_in_raw_api_response() -> None:
    result = AgentResult(
        request_id="r",
        execution_id="e",
        status="completed",
        final_answer="Done",
        tool_history=[],
        total_execution_time_ms=1,
        model_calls=1,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=result.model_dump(exclude={"model_duration_ms"}))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["results"][0]["output_schema"] == "failed"


@pytest.mark.asyncio
async def test_live_report_keeps_each_trial_model_and_marks_mixed_runs() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        calls += 1
        if calls > 2:
            return httpx.Response(502, json={"error": "failed"})
        return httpx.Response(
            200,
            json=AgentResult(
                request_id="r",
                execution_id="e",
                model_id="model-A" if calls == 1 else "model-B",
                status="completed",
                final_answer="Done",
                tool_history=[],
                total_execution_time_ms=1,
                model_calls=1,
            ).model_dump(mode="json"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["results"][0]["model_identity"] == "model-A"
    assert report["results"][1]["model_identity"] == "model-B"
    assert report["results"][2]["model_identity"] is None
    assert report["metadata"]["model"] == "mixed"


@pytest.mark.asyncio
async def test_live_evaluation_keeps_report_when_api_drops_after_health() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["summary"]["infrastructure_errors"] == 10
    assert report["results"][0]["termination_reason"] == "api_transport_ReadTimeout"
    assert "causas: conexión API ReadTimeout" in render_summary(report, Path("reports/latest.json"))


@pytest.mark.asyncio
async def test_live_evaluation_ignores_invalid_local_model_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", "invalid")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json=AgentResult(
                request_id="r",
                execution_id="e",
                status="completed",
                final_answer="Done",
                tool_history=[],
                total_execution_time_ms=1,
                model_calls=1,
            ).model_dump(mode="json"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://api.test"
    ) as client:
        report = await evaluate(mode="live", api_client=client)

    assert report["summary"]["scenarios"] == 10


@pytest.mark.asyncio
async def test_live_evaluation_requires_available_api() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unavailable), base_url="http://api.test"
    ) as client:
        with pytest.raises(EvaluationAPIError, match="make dev"):
            await evaluate(mode="live", api_client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["autonomous", "guided", "deterministic"])
async def test_mock_report_records_strategy_and_resource_metrics(
    strategy: ExecutionStrategy,
) -> None:
    report = await evaluate(mode="mock", strategy=strategy)

    assert report["metadata"]["strategy"] == strategy
    assert report["summary"]["task_successes"] == 10
    assert "mean_tool_calls" in report["summary"]
    assert "required_evidence_coverage" in report["summary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["autonomous", "guided", "deterministic"])
@pytest.mark.parametrize("output_mode", ["free_text", "prompt_json", "schema_json"])
async def test_mock_evaluation_keeps_scoring_dimensions_separate(
    strategy: ExecutionStrategy, output_mode: OutputMode
) -> None:
    report = await evaluate(mode="mock", strategy=strategy, output_mode=output_mode)

    assert report["metadata"]["output_mode"] == output_mode
    format_config = report["metadata"]["response_format_config"]
    assert format_config["prompt_schema"] == (
        IncidentReport.model_json_schema() if output_mode != "free_text" else None
    )
    assert format_config["response_format"] == (
        {"type": "json_object", "schema": IncidentReport.model_json_schema()}
        if output_mode == "schema_json"
        else None
    )
    assert report["summary"]["task_successes"] == 10
    assert report["summary"]["factual_coverage"]["total"] > 0
    assert "format_compliance" in report["summary"]
    assert "evidence_attribution" in report["summary"]
    assert report["summary"]["approximate_unsupported_claim_screen"]["label"].startswith(
        "approximate"
    )
    assert all(
        row["format_compliance"] == ("not_applicable" if output_mode == "free_text" else "valid")
        for row in report["results"]
    )


def test_output_format_and_attribution_do_not_change_task_success() -> None:
    scenario = next(s for s in load_scenarios() if s.test_id == "healthy-service")
    report = IncidentReport(
        confirmed_observations=[
            EvidenceStatement(statement="orders-api is healthy", evidence_sources=["not-a-tool"])
        ],
        hypotheses=[],
        missing_evidence=[],
        tool_failures=[],
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
                result={},
                arguments_valid=True,
            )
        ],
        total_execution_time_ms=1,
        model_calls=1,
        termination_reason="final_response",
        output_mode="prompt_json",
        incident_report=report,
        format_compliance="invalid",
    )

    scored = score_result(scenario, result)

    assert scored["task_success"] is True
    assert scored["format_compliance"] == "invalid"
    assert scored["evidence_attribution"]["matched"] == 0


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
