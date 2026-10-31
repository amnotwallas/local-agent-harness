import asyncio
import logging
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
async def test_api_exposes_resolved_model_id_for_evaluation() -> None:
    class NamedModel(FinalModel):
        model_id = "local-model"

    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=NamedModel())),
        base_url="http://test",
    ) as client:
        response = await client.post("/agents/run", json={"task": "Check status"})

    assert response.status_code == 200
    assert response.json()["model_id"] == "local-model"


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
async def test_request_log_correlates_agent_run_without_logging_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_harness")
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agents/run",
            headers={"x-request-id": "test-request-123"},
            json={"task": "private task text"},
        )

    events = [r for r in caplog.records if getattr(r, "event", None) == "http.request"]
    assert len(events) == 1
    assert events[0].__dict__["request_id"] == response.json()["request_id"] == "test-request-123"
    assert events[0].__dict__["method"] == "POST"
    assert events[0].__dict__["path"] == "/agents/run"
    assert events[0].__dict__["status_code"] == 200
    assert events[0].__dict__["latency_ms"] >= 0
    assert response.headers["x-request-id"] == "test-request-123"
    assert "private task text" not in caplog.text


@pytest.mark.asyncio
async def test_unexpected_error_keeps_request_id_and_redacts_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenModel:
        async def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> ChatResult:
            raise RuntimeError("private server detail")

    caplog.set_level(logging.INFO, logger="agent_harness")
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(settings=Settings(), model=BrokenModel()), raise_app_exceptions=False
        ),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agents/run", headers={"x-request-id": "error-123"}, json={"task": "check"}
        )

    assert response.status_code == 500
    assert response.json()["request_id"] == "error-123"
    assert response.headers["x-request-id"] == "error-123"
    assert any(getattr(r, "event", None) == "http.error" for r in caplog.records)
    assert "private server detail" not in caplog.text


@pytest.mark.asyncio
async def test_untrusted_request_id_is_replaced_for_validation_errors() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        response = await client.post("/agents/run", headers={"x-request-id": "../bad"}, json={})

    assert response.status_code == 422
    assert response.json()["request_id"] != "../bad"
    assert response.headers["x-request-id"] == response.json()["request_id"]


@pytest.mark.asyncio
async def test_unmatched_url_is_not_persisted_in_request_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_harness")
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=FinalModel())),
        base_url="http://test",
    ) as client:
        response = await client.get("/private-token-123")

    event = next(r for r in caplog.records if getattr(r, "event", None) == "http.request")
    assert response.status_code == 404
    assert event.__dict__["path"] == "<unmatched>"
    assert "private-token-123" not in caplog.text


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
    assert "incident_report" in schema["components"]["schemas"]["AgentResult"]["properties"]
    assert (
        schema["components"]["schemas"]["RunRequest"]["properties"]["output_mode"]["default"]
        == "free_text"
    )


@pytest.mark.asyncio
async def test_api_returns_typed_incident_report_when_requested() -> None:
    class ReportModel:
        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> ChatResult:
            if tools is not None:
                return ChatResult(content="Ready to summarize.")
            return ChatResult(
                content=(
                    '{"confirmed_observations":[],"hypotheses":[],"missing_evidence":[],'
                    '"tool_failures":[]}'
                ),
                response_format_status="accepted" if response_format else "not_requested",
            )

    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=Settings(), model=ReportModel())),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/agents/run",
            json={"task": "Review system status", "output_mode": "schema_json"},
        )

    assert response.status_code == 200
    assert response.json()["incident_report"]["hypotheses"] == []
    assert response.json()["format_compliance"] == "valid"


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
