import asyncio
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agent_harness.api.app import create_app
from agent_harness.config import Settings
from agent_harness.models.client import ChatResult, ToolCall


class FinalModel:
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        return ChatResult(content="No active incident is indicated.")


@pytest.mark.asyncio
async def test_health_tools_and_agent_run_contracts() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health")).json()["status"] == "ok"
        assert len((await client.get("/tools")).json()["tools"]) == 3
        response = await client.post("/agents/run", json={"task": "Check system status"})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["request_id"]
    assert response.json()["execution_id"]
    assert response.json()["final_answer"] == "No active incident is indicated."


@pytest.mark.asyncio
async def test_agent_run_rejects_tool_outside_allowlist() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agents/run", json={"task": "Check", "allowed_tools": ["shell"]}
        )
    assert response.status_code == 422
    assert response.json()["error"] == "Requested tool is not available"


@pytest.mark.asyncio
async def test_deterministic_strategy_requires_and_uses_named_workflow() -> None:
    app = create_app(settings=Settings(), model=FinalModel())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing = await client.post(
            "/agents/run", json={"task": "Check orders-api", "strategy": "deterministic"}
        )
        response = await client.post(
            "/agents/run",
            json={
                "task": "Check orders-api health",
                "strategy": "deterministic",
                "workflow": "service_health",
            },
        )

    assert missing.status_code == 422
    assert response.status_code == 200
    assert response.json()["tool_history"][0]["tool_name"] == "get_service_health"
    assert response.json()["strategy"] == "deterministic"


@pytest.mark.asyncio
async def test_guided_strategy_is_selectable_through_api() -> None:
    class GuidedModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            if tools is not None:
                return ChatResult(
                    tool_calls=[
                        ToolCall(
                            id="health",
                            name="get_service_health",
                            arguments={"service_name": "orders-api"},
                        )
                    ]
                )
            return ChatResult(content="Orders-api health checked.")

    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=GuidedModel())),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agents/run", json={"task": "Check orders-api", "strategy": "guided"}
        )

    assert response.status_code == 200
    assert response.json()["strategy"] == "guided"
    assert response.json()["tool_history"][0]["tool_name"] == "get_service_health"


@pytest.mark.asyncio
async def test_agent_response_schema_is_published_in_openapi() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        schema = (await client.get("/openapi.json")).json()

    response = schema["paths"]["/agents/run"]["post"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/AgentResult")
    assert "422" in schema["paths"]["/agents/run"]["post"]["responses"]
    assert "502" in schema["paths"]["/agents/run"]["post"]["responses"]
    assert "504" in schema["paths"]["/agents/run"]["post"]["responses"]


@pytest.mark.asyncio
async def test_execution_deadline_returns_gateway_timeout_with_typed_result() -> None:
    class SlowModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            await asyncio.sleep(1)
            return ChatResult(content="too late")

    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings=Settings(execution_timeout=0.01), model=SlowModel())
        ),
        base_url="http://test",
    ) as client:
        response = await client.post("/agents/run", json={"task": "Wait"})

    assert response.status_code == 504
    assert response.json()["status"] == "deadline_exceeded"
