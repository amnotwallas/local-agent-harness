import asyncio
import logging
from typing import Any

import pytest

from agent_harness.agents.runtime import AgentRuntime, ExecutionStrategy
from agent_harness.config import Settings
from agent_harness.logging_config import request_id_var
from agent_harness.models.client import ChatResult, ToolCall
from agent_harness.tools.registry import build_registry


class ScriptedModel:
    def __init__(self) -> None:
        self.responses = [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="get_service_health",
                        arguments={"service_name": "payments-api"},
                    )
                ]
            ),
            ChatResult(content="Payments API is degraded with elevated latency."),
        ]
        self.calls = 0

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        self.calls += 1
        return self.responses.pop(0)


async def test_runtime_executes_authorized_tool_and_returns_trace() -> None:
    model = ScriptedModel()
    runtime = AgentRuntime(model, build_registry(), Settings())

    result = await runtime.run("Summarize payments-api health")

    assert result.status == "completed"
    assert result.final_answer == "Payments API is degraded with elevated latency."
    assert [item.tool_name for item in result.tool_history] == ["get_service_health"]
    assert result.model_calls == 2
    assert result.termination_reason == "final_response"


async def test_runtime_logs_tool_and_final_metrics_without_arguments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_harness")
    result = await AgentRuntime(ScriptedModel(), build_registry(), Settings()).run(
        "private task text", request_id="runtime-123"
    )

    tool = next(r for r in caplog.records if getattr(r, "event", None) == "tool.completed")
    finished = next(r for r in caplog.records if getattr(r, "event", None) == "agent.finished")
    assert tool.__dict__["request_id"] == finished.__dict__["request_id"] == result.request_id
    assert tool.__dict__["execution_id"] == finished.__dict__["execution_id"] == result.execution_id
    assert tool.__dict__["tool_name"] == "get_service_health"
    assert tool.__dict__["tool_status"] == "ok"
    assert tool.__dict__["tool_latency_ms"] >= 0
    assert finished.__dict__["status"] == "completed"
    assert finished.__dict__["model_calls"] == 2
    assert finished.__dict__["tool_calls"] == 1
    assert "private task text" not in caplog.text
    assert "payments-api" not in caplog.text


async def test_direct_runtime_sets_and_resets_request_context() -> None:
    class ContextModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            assert request_id_var.get() == "direct-123"
            return ChatResult(content="Done")

    assert request_id_var.get() is None
    result = await AgentRuntime(ContextModel(), build_registry(), Settings()).run(
        "private task", request_id="direct-123"
    )

    assert result.status == "completed"
    assert request_id_var.get() is None


@pytest.mark.asyncio
async def test_guided_strategy_validates_and_executes_a_complete_plan() -> None:
    class GuidedModel:
        calls = 0
        final_prompt: str | None = None

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            self.calls += 1
            if self.calls == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        ),
                        ToolCall(
                            id="logs",
                            name="get_recent_logs",
                            arguments={"service_name": "orders-api", "severity": "ERROR"},
                        ),
                    ]
                )
            assert tools is None
            self.final_prompt = messages[-1]["content"]
            return ChatResult(content="Evidence gathered.")

    model = GuidedModel()
    result = await AgentRuntime(model, build_registry(), Settings()).run(
        "Investigate orders-api", strategy="guided"
    )

    assert result.status == "completed"
    assert [item.tool_name for item in result.tool_history] == [
        "get_service_health",
        "get_recent_logs",
    ]
    assert model.calls == 2
    assert model.final_prompt == (
        "Summarize the results of the completed investigation. Use only the evidence above, "
        "describe missing or failed evidence, and state uncertainty."
    )


@pytest.mark.asyncio
async def test_guided_strategy_rejects_the_entire_invalid_plan_before_execution() -> None:
    class InvalidPlanModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="valid-first",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    ),
                    ToolCall(id="invalid-second", name="shell", arguments={}),
                ]
            )

    result = await AgentRuntime(InvalidPlanModel(), build_registry(), Settings()).run(
        "Check orders-api", strategy="guided"
    )

    assert result.status == "failed"
    assert result.termination_reason == "invalid_tool_plan"
    assert result.tool_history == []


@pytest.mark.asyncio
async def test_guided_strategy_rejects_invalid_arguments_before_any_execution() -> None:
    class InvalidPlanModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="valid-first",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    ),
                    ToolCall(
                        id="invalid-second",
                        name="get_recent_logs",
                        arguments={"severity": "TRACE"},
                    ),
                ]
            )

    result = await AgentRuntime(InvalidPlanModel(), build_registry(), Settings()).run(
        "Check orders-api", strategy="guided"
    )

    assert result.status == "failed"
    assert result.termination_reason == "invalid_tool_plan"
    assert result.tool_history == []


@pytest.mark.asyncio
async def test_guided_strategy_checks_full_plan_against_tool_call_cap() -> None:
    class TwoCallPlanModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="health",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    ),
                    ToolCall(
                        id="logs",
                        name="get_recent_logs",
                        arguments={"service_name": "orders-api"},
                    ),
                ]
            )

    result = await AgentRuntime(
        TwoCallPlanModel(), build_registry(), Settings(max_tool_calls=1)
    ).run("Check orders-api", strategy="guided")

    assert result.status == "tool_call_limit_reached"
    assert result.tool_history == []


@pytest.mark.asyncio
async def test_guided_strategy_rejects_plan_that_cannot_fit_remaining_time_budget() -> None:
    class TwoCallPlanModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="health",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    ),
                    ToolCall(
                        id="logs",
                        name="get_recent_logs",
                        arguments={"service_name": "orders-api"},
                    ),
                ]
            )

    result = await AgentRuntime(
        TwoCallPlanModel(),
        build_registry(),
        Settings(execution_timeout=1, tool_timeout=0.7),
    ).run("Check orders-api", strategy="guided")

    assert result.status == "deadline_exceeded"
    assert result.termination_reason == "plan_exceeds_time_budget"
    assert result.tool_history == []


@pytest.mark.asyncio
async def test_deterministic_strategy_requires_a_known_workflow() -> None:
    result = await AgentRuntime(ScriptedModel(), build_registry(), Settings()).run(
        "Check orders-api", strategy="deterministic"
    )

    assert result.status == "failed"
    assert result.termination_reason == "missing_workflow"


@pytest.mark.asyncio
async def test_deterministic_strategy_rejects_unknown_workflow() -> None:
    result = await AgentRuntime(ScriptedModel(), build_registry(), Settings()).run(
        "Check orders-api", strategy="deterministic", workflow="scenario-id"
    )

    assert result.status == "failed"
    assert result.termination_reason == "unknown_workflow"
    assert result.tool_history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["autonomous", "guided", "deterministic"])
async def test_each_strategy_enforces_end_to_end_deadline(strategy: ExecutionStrategy) -> None:
    class SlowModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            await asyncio.sleep(1)
            return ChatResult(content="too late")

    result = await AgentRuntime(
        SlowModel(),
        build_registry(),
        Settings(execution_timeout=0.05, tool_timeout=0.01),
    ).run(
        "Check unknown-api",
        strategy=strategy,
        workflow="service_health" if strategy == "deterministic" else None,
    )

    assert result.status == "deadline_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["autonomous", "guided", "deterministic"])
async def test_tool_failures_are_traced_for_each_strategy(strategy: ExecutionStrategy) -> None:
    class ToolFailureModel:
        calls = 0

        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            self.calls += 1
            if strategy != "deterministic" and self.calls == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "unknown-api"},
                        )
                    ]
                )
            return ChatResult(content="The health lookup failed; status is unavailable.")

    result = await AgentRuntime(ToolFailureModel(), build_registry(), Settings()).run(
        "Check unknown-api",
        strategy=strategy,
        workflow="service_health" if strategy == "deterministic" else None,
    )

    assert result.status == "completed"
    assert len(result.tool_history) == 1
    assert result.tool_history[0].error == "Tool execution failed"
    assert result.tool_history[0].arguments_valid is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["autonomous", "guided", "deterministic"])
async def test_caller_cancellation_propagates_for_each_strategy(
    strategy: ExecutionStrategy,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return ChatResult(content="unreachable")

    task = asyncio.create_task(
        AgentRuntime(
            WaitingModel(),
            build_registry(),
            Settings(execution_timeout=10, tool_timeout=1),
        ).run(
            "Check orders-api",
            strategy=strategy,
            workflow="service_health" if strategy == "deterministic" else None,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_json_fallback_is_used_only_when_explicitly_configured() -> None:
    class JsonModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            assert tools is None
            return ChatResult(content='{"final":"structured fallback result"}')

    result = await AgentRuntime(JsonModel(), build_registry(), Settings(tool_protocol="json")).run(
        "Summarize"
    )
    assert result.status == "completed"
    assert result.final_answer == "structured fallback result"


@pytest.mark.asyncio
async def test_malformed_json_fallback_fails_explicitly() -> None:
    class JsonModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(content="not json")

    result = await AgentRuntime(JsonModel(), build_registry(), Settings(tool_protocol="json")).run(
        "Summarize"
    )
    assert result.status == "failed"
    assert result.termination_reason == "model_or_protocol_error"


@pytest.mark.asyncio
async def test_json_fallback_rejects_non_object_response() -> None:
    class JsonModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(content='["not", "an", "object"]')

    result = await AgentRuntime(JsonModel(), build_registry(), Settings(tool_protocol="json")).run(
        "Summarize"
    )
    assert result.status == "failed"
    assert result.termination_reason == "model_or_protocol_error"


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_replayed_as_one_assistant_message() -> None:
    class MultiCallModel:
        calls = 0
        messages_after_tools: list[dict[str, Any]] | None = None

        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            self.calls += 1
            if self.calls == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        ),
                        ToolCall(
                            id="logs",
                            name="get_recent_logs",
                            arguments={"service_name": "orders-api"},
                        ),
                    ]
                )
            self.messages_after_tools = messages
            return ChatResult(content="Both checks completed.")

    model = MultiCallModel()
    result = await AgentRuntime(model, build_registry(), Settings()).run("Check orders-api")

    assert result.status == "completed"
    assert model.messages_after_tools is not None
    assistant_call = model.messages_after_tools[2]
    assert assistant_call["role"] == "assistant"
    assert [call["id"] for call in assistant_call["tool_calls"]] == ["health", "logs"]
    assert [message["tool_call_id"] for message in model.messages_after_tools[3:]] == [
        "health",
        "logs",
    ]


@pytest.mark.asyncio
async def test_execution_deadline_returns_a_termination_result() -> None:
    class SlowModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            await asyncio.sleep(1)
            return ChatResult(content="too late")

    result = await AgentRuntime(
        SlowModel(), build_registry(), Settings(execution_timeout=0.01)
    ).run("Wait")

    assert result.status == "deadline_exceeded"
    assert result.termination_reason == "execution_deadline"
    assert result.model_calls == 1


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_to_the_model_call() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return ChatResult(content="unexpected")

    task = asyncio.create_task(
        AgentRuntime(WaitingModel(), build_registry(), Settings(execution_timeout=10)).run("Wait")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_tool_call_cap_stops_before_executing_an_oversized_batch() -> None:
    class MultiCallModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="health",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    ),
                    ToolCall(
                        id="logs",
                        name="get_recent_logs",
                        arguments={"service_name": "orders-api"},
                    ),
                ]
            )

    result = await AgentRuntime(MultiCallModel(), build_registry(), Settings(max_tool_calls=1)).run(
        "Check orders-api"
    )

    assert result.status == "tool_call_limit_reached"
    assert result.tool_history == []


@pytest.mark.asyncio
async def test_tool_call_cap_applies_across_model_iterations() -> None:
    class RepeatedCallModel:
        calls = 0

        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            self.calls += 1
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id=str(self.calls),
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    )
                ]
            )

    result = await AgentRuntime(
        RepeatedCallModel(), build_registry(), Settings(max_tool_calls=1)
    ).run("Check orders-api")

    assert result.status == "tool_call_limit_reached"
    assert len(result.tool_history) == 1


@pytest.mark.asyncio
async def test_tool_timeout_cancels_async_tool_and_runtime_can_finish() -> None:
    class ToolThenFinalModel:
        calls = 0

        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            self.calls += 1
            if self.calls == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        )
                    ]
                )
            return ChatResult(content="Timed out safely.")

    registry = build_registry()
    cancelled = asyncio.Event()

    async def slow_call(name: str, arguments: dict[str, Any]) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    registry.call = slow_call  # type: ignore[method-assign]
    result = await AgentRuntime(
        ToolThenFinalModel(),
        registry,
        Settings(tool_timeout=0.01, execution_timeout=1),
    ).run("Check orders-api")

    assert result.status == "completed"
    assert result.tool_history[0].error == "Tool execution failed"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_execution_deadline_cancels_a_tool_call() -> None:
    class ToolModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="health",
                        name="get_service_health",
                        arguments={"service_name": "orders-api"},
                    )
                ]
            )

    registry = build_registry()
    cancelled = asyncio.Event()

    async def slow_call(name: str, arguments: dict[str, Any]) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    registry.call = slow_call  # type: ignore[method-assign]
    result = await AgentRuntime(
        ToolModel(),
        registry,
        Settings(tool_timeout=1, execution_timeout=0.01),
    ).run("Check orders-api")

    assert result.status == "deadline_exceeded"
    assert result.termination_reason == "execution_deadline"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_schema_json_is_requested_only_for_final_synthesis_after_tool_calls() -> None:
    from agent_harness.agents.incident_report import IncidentReport

    valid_report = IncidentReport(
        confirmed_observations=[],
        hypotheses=[],
        missing_evidence=[],
        tool_failures=[],
    ).model_dump_json()

    class SchemaModel:
        calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None, Any]] = []

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            self.calls.append((messages, tools, response_format))
            if len(self.calls) == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        )
                    ]
                )
            if len(self.calls) == 2:
                return ChatResult(content="Investigation complete.")
            return ChatResult(content=valid_report, response_format_status="accepted")

    model = SchemaModel()
    result = await AgentRuntime(model, build_registry(), Settings()).run(
        "Check orders-api", output_mode="schema_json"
    )

    assert result.status == "completed"
    assert result.format_compliance == "valid"
    assert result.incident_report is not None
    assert len(model.calls) == 3
    assert model.calls[0][1] is not None
    assert model.calls[1][1] is not None
    assert model.calls[2][1] is None
    assert model.calls[0][2] is None
    assert model.calls[1][2] is None
    assert model.calls[2][2] == {
        "type": "json_object",
        "schema": IncidentReport.model_json_schema(),
    }


@pytest.mark.asyncio
async def test_invalid_structured_output_is_retained_without_repair_or_fallback() -> None:
    class InvalidJsonModel:
        calls = 0

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            self.calls += 1
            if self.calls < 3:
                if self.calls == 1:
                    return ChatResult(
                        tool_calls=[
                            ToolCall(
                                id="health",
                                name="get_service_health",
                                arguments={"service_name": "orders-api"},
                            )
                        ]
                    )
                return ChatResult(content="Investigation complete.")
            return ChatResult(content='{"confirmed_observations": []}')

    model = InvalidJsonModel()
    result = await AgentRuntime(model, build_registry(), Settings()).run(
        "Check orders-api", output_mode="prompt_json"
    )

    assert result.status == "completed"
    assert result.format_compliance == "invalid"
    assert result.incident_report is None
    assert result.final_answer == '{"confirmed_observations": []}'
    assert model.calls == 3


@pytest.mark.asyncio
async def test_rejected_schema_format_is_reported_without_fallback() -> None:
    from agent_harness.models.client import ModelError

    class RejectingSchemaModel:
        calls = 0

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            self.calls += 1
            if self.calls == 1:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        )
                    ]
                )
            if self.calls == 2:
                return ChatResult(content="Investigation complete.")
            assert tools is None
            assert response_format is not None
            raise ModelError(
                "The inference server returned an invalid or failed response",
                response_format_status="rejected",
                response_format_status_code=400,
            )

    model = RejectingSchemaModel()
    result = await AgentRuntime(model, build_registry(), Settings()).run(
        "Check orders-api", output_mode="schema_json"
    )

    assert result.status == "failed"
    assert result.response_format_status == "rejected"
    assert result.response_format_status_code == 400
    assert result.format_compliance == "not_produced"
    assert model.calls == 3


@pytest.mark.asyncio
async def test_schema_timeout_records_unproduced_format_and_unknown_acceptance() -> None:
    class SlowSummaryModel:
        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            await asyncio.sleep(1)
            return ChatResult(content="too late")

    result = await AgentRuntime(
        SlowSummaryModel(),
        build_registry(),
        Settings(execution_timeout=0.05, tool_timeout=0.01),
    ).run(
        "Check orders-api",
        strategy="deterministic",
        workflow="service_health",
        output_mode="schema_json",
    )

    assert result.status == "deadline_exceeded"
    assert result.output_mode == "schema_json"
    assert result.format_compliance == "not_produced"
    assert result.response_format_status == "unknown"
