import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_harness.agents.runtime import AgentResult, AgentRuntime
from agent_harness.config import Settings
from agent_harness.evals.scenarios import Scenario, load_scenarios
from agent_harness.models.client import ChatResult, OpenAIClient, ToolCall
from agent_harness.tools.registry import build_registry


class ScenarioModel:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.sent_tools = False

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        if not self.sent_tools and self.scenario.expected_tool_calls:
            self.sent_tools = True
            calls = [
                ToolCall(
                    id=f"{self.scenario.test_id}-{i}", name=call.name, arguments=call.arguments
                )
                for i, call in enumerate(self.scenario.expected_tool_calls)
            ]
            return ChatResult(tool_calls=calls)
        answer = " ".join(phrases[0] for phrases in self.scenario.required_facts.values())
        return ChatResult(content=json.dumps({"final": answer}) if tools is None else answer)


def score_result(scenario: Scenario, result: AgentResult) -> dict[str, Any]:
    answer = result.final_answer or ""
    actual_calls = [
        {"name": item.tool_name, "arguments": item.arguments} for item in result.tool_history
    ]
    expected_calls = [call.model_dump() for call in scenario.expected_tool_calls]
    errors = sum(item.error is not None for item in result.tool_history)
    required_facts = {
        fact: "covered"
        if any(phrase.casefold() in answer.casefold() for phrase in equivalents)
        else "uncovered"
        for fact, equivalents in scenario.required_facts.items()
    }
    try:
        typed_result = AgentResult.model_validate(dict(result.__dict__))
        response_fields = set(typed_result.model_dump())
        schema_ok = all(key in response_fields for key in scenario.expected_output_structure)
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
        "unsupported_claims": {claim: "unverified" for claim in scenario.unsupported_claims},
        "output_schema": "compliant" if schema_ok else "failed",
        "execution_errors": errors,
        "task_success": task_success,
    }


async def evaluate(
    mode: str = "mock", trials: int = 1, baseline: Path | None = None
) -> dict[str, Any]:
    settings = Settings.from_env()
    scenarios = load_scenarios()
    results: list[dict[str, Any]] = []
    resolved_model_id = settings.model
    for scenario in scenarios:
        for trial in range(trials):
            if mode == "mock":
                model: Any = ScenarioModel(scenario)
            else:
                model = OpenAIClient(settings)
            runtime_settings = Settings(
                **{
                    **settings.__dict__,
                    "max_iterations": min(settings.max_iterations, scenario.max_tool_iterations),
                }
            )
            runtime = AgentRuntime(
                model, build_registry(mock_responses=scenario.mock_tool_responses), runtime_settings
            )
            result = await runtime.run(scenario.task)
            if isinstance(model, OpenAIClient):
                resolved_model_id = model.model_id or resolved_model_id
                await model.close()
            scored = score_result(scenario, result)
            results.append(
                {
                    "test_id": scenario.test_id,
                    "trial": trial + 1,
                    **scored,
                    "latency_ms": result.total_execution_time_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "termination_reason": result.termination_reason,
                }
            )
    report: dict[str, Any] = {
        "metadata": {
            "mode": mode,
            "evaluation_scope": "harness_behavior" if mode == "mock" else "live_model_behavior",
            "tool_data_scope": "scenario_fixtures",
            "model": resolved_model_id,
            "dataset": "aiops-v1",
            "harness_version": "0.1.0",
            "inference": {
                "base_url": settings.llm_base_url,
                "temperature": settings.temperature,
                "tool_protocol": settings.tool_protocol,
            },
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
            "by_scenario": {},
        },
    }
    for scenario in scenarios:
        rows = [row for row in results if row["test_id"] == scenario.test_id]
        report["summary"]["by_scenario"][scenario.test_id] = {
            "trials": len(rows),
            "task_successes": sum(row["task_success"] for row in rows),
            "tool_selection_passed": sum(row["tool_selection_accuracy"] for row in rows),
            "evaluation_failures": sum(not row["evaluation_assertions_passed"] for row in rows),
            "infrastructure_errors": sum(row["infrastructure_error"] for row in rows),
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
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    report = asyncio.run(evaluate(args.mode, args.trials, args.baseline))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
