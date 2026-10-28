import asyncio
from typing import Any

import pytest

from agent_harness.agents.runtime import AgentRuntime
from agent_harness.config import Settings
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
