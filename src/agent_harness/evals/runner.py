import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import httpx
from dotenv import load_dotenv

from agent_harness.agents.incident_report import EvidenceStatement, IncidentReport
from agent_harness.agents.runtime import AgentResult, AgentRuntime, ExecutionStrategy, OutputMode
from agent_harness.config import Settings
from agent_harness.evals.reporting import render_summary
from agent_harness.evals.scenarios import Scenario, load_scenarios
from agent_harness.logging_config import configure_logging
from agent_harness.models.client import ChatResult, ToolCall
from agent_harness.tools.registry import build_registry


class ScenarioModel:
    def __init__(
        self, scenario: Scenario, strategy: ExecutionStrategy, output_mode: OutputMode
    ) -> None:
        self.scenario = scenario
        self.strategy = strategy
        self.output_mode = output_mode
        self.sent_tools = False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if (
            self.strategy in {"autonomous", "guided"}
            and not self.sent_tools
            and self.scenario.expected_tool_calls
        ):
            self.sent_tools = True
            calls = [
                ToolCall(
                    id=f"{self.scenario.test_id}-{i}", name=call.name, arguments=call.arguments
                )
                for i, call in enumerate(self.scenario.expected_tool_calls)
            ]
            return ChatResult(tool_calls=calls)
        answer = " ".join(phrases[0] for phrases in self.scenario.required_facts.values())
        if self.output_mode != "free_text" and tools is None:
            sources = list(dict.fromkeys(call.name for call in self.scenario.expected_tool_calls))
            return ChatResult(
                content=IncidentReport(
                    confirmed_observations=[
                        EvidenceStatement(statement=phrase[0], evidence_sources=sources)
                        for phrase in self.scenario.required_facts.values()
                    ],
                    hypotheses=[],
                    missing_evidence=[],
                    tool_failures=[],
                ).model_dump_json()
            )
        return ChatResult(content=json.dumps({"final": answer}) if tools is None else answer)


class EvaluationAPIError(RuntimeError):
    pass


# Settings caps server execution at 3600 seconds; leave room for HTTP overhead.
API_RUN_TIMEOUT_SECONDS = 3610


def _api_failure_result(
    reason: str,
    strategy: ExecutionStrategy,
    workflow: str | None,
    output_mode: OutputMode,
    request_id: str = "api-error",
) -> AgentResult:
    return AgentResult(
        request_id=request_id,
        execution_id="api-error",
        status="failed",
        final_answer=None,
        tool_history=[],
        total_execution_time_ms=0,
        model_calls=0,
        termination_reason=reason,
        strategy=strategy,
        workflow=workflow,
        output_mode=output_mode,
    )


def score_result(
    scenario: Scenario, result: AgentResult, response_fields: set[str] | None = None
) -> dict[str, Any]:
    answer = result.final_answer or ""
    actual_calls = [
        {"name": item.tool_name, "arguments": item.arguments} for item in result.tool_history
    ]
    expected_calls = [call.model_dump() for call in scenario.expected_tool_calls]
    errors = sum(item.error is not None for item in result.tool_history)
    evidence_results = [
        any(
            item.tool_name == call.name
            and item.arguments == call.arguments
            and item.error is None
            and item.result is not None
            for item in result.tool_history
        )
        for call in scenario.expected_tool_calls
    ]
    evidence_attempts = [
        any(
            item.tool_name == call.name and item.arguments == call.arguments
            for item in result.tool_history
        )
        for call in scenario.expected_tool_calls
    ]
    required_facts = {
        fact: "covered"
        if any(phrase.casefold() in answer.casefold() for phrase in equivalents)
        else "uncovered"
        for fact, equivalents in scenario.required_facts.items()
    }
    try:
        typed_result = AgentResult.model_validate(dict(result.__dict__))
        fields = response_fields if response_fields is not None else set(typed_result.model_dump())
        schema_ok = all(key in fields for key in scenario.expected_output_structure)
    except Exception:
        schema_ok = False
    allowed_calls = [call.model_dump() for call in scenario.allowed_tool_calls]
    forbidden_calls = [call.model_dump() for call in scenario.forbidden_tool_calls]
    ordered_index = 0
    required_in_order = True
    for actual in actual_calls:
        if ordered_index < len(expected_calls) and actual == expected_calls[ordered_index]:
            ordered_index += 1
        elif actual in expected_calls and actual not in allowed_calls:
            required_in_order = False
    selection_ok = (
        required_in_order
        and ordered_index == len(expected_calls)
        and all(actual in expected_calls or actual in allowed_calls for actual in actual_calls)
        and not any(actual in forbidden_calls for actual in actual_calls)
    )
    arguments_valid = all(item.arguments_valid is True for item in result.tool_history)
    evaluation_ok = (
        arguments_valid
        and all(value == "covered" for value in required_facts.values())
        and schema_ok
        and errors == scenario.expected_tool_errors
    )
    task_success = result.status == "completed" and selection_ok and evaluation_ok
    report = result.incident_report
    available_sources = {
        item.tool_name
        for item in result.tool_history
        if item.error is None and item.result is not None
    }
    report_observations = report.confirmed_observations if report is not None else []
    attribution_matches = sum(
        bool(observation.evidence_sources)
        and all(source in available_sources for source in observation.evidence_sources)
        for observation in report_observations
    )
    unsupported_matches = {
        claim: "matched" if claim.casefold() in answer.casefold() else "not_detected"
        for claim in scenario.unsupported_claims
    }
    return {
        "status": result.status,
        "tool_calls": actual_calls,
        "expected_tool_calls": expected_calls,
        "tool_selection_accuracy": selection_ok,
        "evaluation_assertions_passed": evaluation_ok,
        "infrastructure_error": result.status in {"failed", "deadline_exceeded"},
        "tool_argument_validity": [
            "valid" if item.arguments_valid else "invalid" for item in result.tool_history
        ],
        "required_facts": required_facts,
        "required_facts_present": all(value == "covered" for value in required_facts.values()),
        "required_evidence_coverage": {
            "covered": sum(evidence_results),
            "attempted": sum(evidence_attempts),
            "total": len(expected_calls),
            "rate": sum(evidence_results) / len(expected_calls) if expected_calls else 1.0,
        },
        "unsupported_claims": {claim: "unverified" for claim in scenario.unsupported_claims},
        "output_schema": "compliant" if schema_ok else "failed",
        "output_mode": result.output_mode,
        "format_compliance": result.format_compliance,
        "response_format_status": result.response_format_status,
        "response_format_status_code": result.response_format_status_code,
        "evidence_attribution": {
            "status": "measured" if report is not None else "not_applicable",
            "method": "successful_tool_name_match; does_not_verify_statement_support",
            "matched": attribution_matches,
            "total": len(report_observations),
            "rate": attribution_matches / len(report_observations) if report_observations else None,
        },
        "approximate_unsupported_claim_screen": {
            "method": "case_insensitive_exact_phrase_match",
            "label": "approximate; not a verified hallucination rate",
            "matches": unsupported_matches,
            "count": sum(value == "matched" for value in unsupported_matches.values()),
        },
        "execution_errors": errors,
        "expected_tool_errors": scenario.expected_tool_errors,
        "task_success": task_success,
    }


async def evaluate(
    mode: str = "mock",
    trials: int = 1,
    baseline: Path | None = None,
    strategy: ExecutionStrategy = "autonomous",
    output_mode: OutputMode = "free_text",
    api_client: httpx.AsyncClient | None = None,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    load_dotenv(dotenv_path=".env", override=False)
    settings = Settings.from_env() if mode == "mock" else Settings()
    scenarios = load_scenarios()
    results: list[dict[str, Any]] = []
    observed_models: set[str] = set()
    api_url = api_base_url or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000"
    async with AsyncExitStack() as stack:
        if mode == "live":
            if api_client is None:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        base_url=api_url,
                        timeout=5,
                    )
                )
            else:
                client = api_client
                api_url = str(client.base_url).rstrip("/")
            try:
                health = await client.get("/health", timeout=5)
                health.raise_for_status()
                if health.json().get("status") != "ok":
                    raise ValueError("unexpected health response")
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                raise EvaluationAPIError(
                    "La API no está disponible; inicia `make dev` y revisa API_BASE_URL."
                ) from exc
        for scenario in scenarios:
            for trial in range(trials):
                workflow = scenario.deterministic_workflow if strategy == "deterministic" else None
                if mode == "mock":
                    runtime_settings = Settings(
                        **{
                            **settings.__dict__,
                            "max_iterations": min(
                                settings.max_iterations, scenario.max_tool_iterations
                            ),
                        }
                    )
                    runtime = AgentRuntime(
                        ScenarioModel(scenario, strategy, output_mode),
                        build_registry(mock_responses=scenario.mock_tool_responses),
                        runtime_settings,
                    )
                    result = await runtime.run(
                        scenario.task,
                        strategy=strategy,
                        workflow=workflow,
                        output_mode=output_mode,
                    )
                    api_status_code = None
                    api_latency_ms = 0.0
                    runtime_latency_ms: float | None = result.total_execution_time_ms
                else:
                    api_started = perf_counter()
                    try:
                        response = await client.post(
                            "/agents/run",
                            json={
                                "task": scenario.task,
                                "strategy": strategy,
                                "workflow": workflow,
                                "output_mode": output_mode,
                            },
                            timeout=httpx.Timeout(
                                API_RUN_TIMEOUT_SECONDS, connect=5, write=5, pool=5
                            ),
                        )
                    except httpx.HTTPError as exc:
                        api_latency_ms = (perf_counter() - api_started) * 1000
                        api_status_code = None
                        runtime_latency_ms = None
                        result = _api_failure_result(
                            f"api_transport_{type(exc).__name__}", strategy, workflow, output_mode
                        )
                        response_fields = set()
                    else:
                        api_latency_ms = (perf_counter() - api_started) * 1000
                        api_status_code = response.status_code
                        if api_status_code in {404, 422}:
                            raise EvaluationAPIError(
                                f"La API rechazó el escenario {scenario.test_id} "
                                f"(HTTP {api_status_code})."
                            )
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = None
                        if isinstance(payload, dict) and "tool_history" in payload:
                            try:
                                result = AgentResult.model_validate(payload)
                            except ValueError as exc:
                                raise EvaluationAPIError(
                                    "La API devolvió una respuesta inválida "
                                    f"para {scenario.test_id}."
                                ) from exc
                            if result.model_id:
                                observed_models.add(result.model_id)
                            runtime_latency_ms = result.total_execution_time_ms
                            response_fields = set(payload)
                        elif api_status_code >= 500:
                            result = _api_failure_result(
                                f"api_http_{api_status_code}",
                                strategy,
                                workflow,
                                output_mode,
                                response.headers.get("x-request-id", "api-error"),
                            )
                            runtime_latency_ms = None
                            response_fields = set()
                        else:
                            raise EvaluationAPIError(
                                f"La API devolvió una respuesta inválida para {scenario.test_id}."
                            )
                scored = score_result(scenario, result, response_fields if mode == "live" else None)
                results.append(
                    {
                        "test_id": scenario.test_id,
                        "trial": trial + 1,
                        **scored,
                        "api_status_code": api_status_code,
                        "latency_ms": api_latency_ms
                        if mode == "live"
                        else result.total_execution_time_ms,
                        "runtime_latency_ms": runtime_latency_ms,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "termination_reason": result.termination_reason,
                        "model_identity": result.model_id if mode == "live" else settings.model,
                    }
                )
    report: dict[str, Any] = {
        "metadata": {
            "mode": mode,
            "evaluation_scope": "harness_behavior" if mode == "mock" else "live_model_behavior",
            "tool_data_scope": "api_tools" if mode == "live" else "scenario_fixtures",
            "api_base_url": api_url if mode == "live" else None,
            "strategy": strategy,
            "output_mode": output_mode,
            "model": (
                settings.model
                if mode == "mock"
                else next(iter(observed_models))
                if len(observed_models) == 1
                else "mixed"
                if observed_models
                else None
            ),
            "dataset": "aiops-v1",
            "harness_version": "0.1.0",
            "inference": (
                {"source": "api_server", "api_base_url": api_url}
                if mode == "live"
                else {
                    "base_url": settings.llm_base_url,
                    "temperature": settings.temperature,
                    "tool_protocol": settings.tool_protocol,
                }
            ),
            "response_format_config": (
                {"mode": output_mode, "source": "api_server", "schema": "not_introspected"}
                if mode == "live"
                else {
                    "mode": output_mode,
                    "prompt_schema": IncidentReport.model_json_schema()
                    if output_mode in {"prompt_json", "schema_json"}
                    else None,
                    "response_format": {
                        "type": "json_object",
                        "schema": IncidentReport.model_json_schema(),
                    }
                    if output_mode == "schema_json"
                    else None,
                    "prompt_schema_included": output_mode in {"prompt_json", "schema_json"},
                    "server_acceptance_semantics": (
                        "HTTP 2xx means request accepted; it does not prove schema enforcement."
                    ),
                }
            ),
            "configuration": (
                {"trials": trials}
                if mode == "live"
                else {
                    "max_iterations": settings.max_iterations,
                    "max_tool_calls": settings.max_tool_calls,
                    "execution_timeout": settings.execution_timeout,
                    "tool_timeout": settings.tool_timeout,
                    "llm_timeout": settings.llm_timeout,
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                    "trials": trials,
                }
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "trials": trials,
        },
        "results": results,
        "summary": {
            "scenarios": len(scenarios),
            "trials": trials,
            "completed": sum(row["status"] == "completed" for row in results),
            "task_successes": sum(row["task_success"] for row in results),
            "tool_selection_passed": sum(row["tool_selection_accuracy"] for row in results),
            "tool_selection_failures": sum(not row["tool_selection_accuracy"] for row in results),
            "evaluation_failures": sum(not row["evaluation_assertions_passed"] for row in results),
            "infrastructure_errors": sum(row["infrastructure_error"] for row in results),
            "format_compliance": {
                state: sum(row["format_compliance"] == state for row in results)
                for state in ("valid", "invalid", "not_produced", "not_applicable")
            },
            "response_format_status": {
                state: sum(row["response_format_status"] == state for row in results)
                for state in ("accepted", "rejected", "unknown", "not_requested")
            },
            "factual_coverage": {
                "covered": sum(
                    sum(value == "covered" for value in row["required_facts"].values())
                    for row in results
                ),
                "total": sum(len(row["required_facts"]) for row in results),
            },
            "evidence_attribution": {
                "method": "successful_tool_name_match; does_not_verify_statement_support",
                "matched": sum(row["evidence_attribution"]["matched"] for row in results),
                "total": sum(row["evidence_attribution"]["total"] for row in results),
            },
            "approximate_unsupported_claim_screen": {
                "label": "approximate; not a verified hallucination rate",
                "matches": sum(
                    row["approximate_unsupported_claim_screen"]["count"] for row in results
                ),
                "assertions": sum(
                    len(row["approximate_unsupported_claim_screen"]["matches"]) for row in results
                ),
            },
            "required_evidence_coverage": {
                "covered": sum(row["required_evidence_coverage"]["covered"] for row in results),
                "attempted": sum(row["required_evidence_coverage"]["attempted"] for row in results),
                "total": sum(row["required_evidence_coverage"]["total"] for row in results),
            },
            "mean_tool_calls": mean(len(row["tool_calls"]) for row in results),
            "mean_tool_errors": mean(row["execution_errors"] for row in results),
            "mean_latency_ms": mean(row["latency_ms"] for row in results),
            "median_latency_ms": median(row["latency_ms"] for row in results),
            "mean_runtime_latency_ms": (
                mean(
                    row["runtime_latency_ms"]
                    for row in results
                    if row["runtime_latency_ms"] is not None
                )
                if any(row["runtime_latency_ms"] is not None for row in results)
                else None
            ),
            "mean_prompt_tokens": mean(row["prompt_tokens"] for row in results),
            "mean_completion_tokens": mean(row["completion_tokens"] for row in results),
            "by_scenario": {},
        },
    }
    evidence = report["summary"]["required_evidence_coverage"]
    evidence["rate"] = evidence["covered"] / evidence["total"] if evidence["total"] else 1.0
    report_facts = report["summary"]["factual_coverage"]
    report_facts["rate"] = (
        report_facts["covered"] / report_facts["total"] if report_facts["total"] else 1.0
    )
    report_attribution = report["summary"]["evidence_attribution"]
    report_attribution["rate"] = (
        report_attribution["matched"] / report_attribution["total"]
        if report_attribution["total"]
        else None
    )
    for scenario in scenarios:
        rows = [row for row in results if row["test_id"] == scenario.test_id]
        report["summary"]["by_scenario"][scenario.test_id] = {
            "trials": len(rows),
            "task_successes": sum(row["task_success"] for row in rows),
            "tool_selection_passed": sum(row["tool_selection_accuracy"] for row in rows),
            "evaluation_failures": sum(not row["evaluation_assertions_passed"] for row in rows),
            "infrastructure_errors": sum(row["infrastructure_error"] for row in rows),
            "format_compliance": {
                state: sum(row["format_compliance"] == state for row in rows)
                for state in ("valid", "invalid", "not_produced", "not_applicable")
            },
            "response_format_status": {
                state: sum(row["response_format_status"] == state for row in rows)
                for state in ("accepted", "rejected", "unknown", "not_requested")
            },
            "factual_coverage": {
                "covered": sum(
                    sum(value == "covered" for value in row["required_facts"].values())
                    for row in rows
                ),
                "total": sum(len(row["required_facts"]) for row in rows),
            },
            "evidence_attribution": {
                "method": "successful_tool_name_match; does_not_verify_statement_support",
                "matched": sum(row["evidence_attribution"]["matched"] for row in rows),
                "total": sum(row["evidence_attribution"]["total"] for row in rows),
            },
            "approximate_unsupported_claim_matches": sum(
                row["approximate_unsupported_claim_screen"]["count"] for row in rows
            ),
            "required_evidence_coverage": {
                "covered": sum(row["required_evidence_coverage"]["covered"] for row in rows),
                "total": sum(row["required_evidence_coverage"]["total"] for row in rows),
            },
            "mean_tool_calls": mean(len(row["tool_calls"]) for row in rows),
            "mean_latency_ms": mean(row["latency_ms"] for row in rows),
            "mean_prompt_tokens": mean(row["prompt_tokens"] for row in rows),
            "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        }
    if baseline:
        previous = json.loads(baseline.read_text())
        report["baseline_comparison"] = {
            "completed_delta": report["summary"]["completed"]
            - previous.get("summary", {}).get("completed", 0),
            "task_success_delta": report["summary"]["task_successes"]
            - previous.get("summary", {}).get("task_successes", 0),
            "tool_selection_delta": report["summary"]["tool_selection_passed"]
            - previous.get("summary", {}).get("tool_selection_passed", 0),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live"], default="live")
    parser.add_argument(
        "--strategy", choices=["autonomous", "guided", "deterministic"], default="autonomous"
    )
    parser.add_argument(
        "--output-mode", choices=["free_text", "prompt_json", "schema_json"], default="free_text"
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    load_dotenv(dotenv_path=".env", override=False)
    configure_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/server.jsonl") or None,
        console=False,
    )
    try:
        report = asyncio.run(
            evaluate(
                args.mode,
                args.trials,
                args.baseline,
                strategy=args.strategy,
                output_mode=args.output_mode,
                api_base_url=args.api_url,
            )
        )
    except EvaluationAPIError as exc:
        parser.exit(2, f"Error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(render_summary(report, args.output))


if __name__ == "__main__":
    main()
