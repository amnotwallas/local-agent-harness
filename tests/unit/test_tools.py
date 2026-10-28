import pytest

from agent_harness.tools.registry import build_registry


@pytest.mark.asyncio
async def test_registry_exposes_only_read_only_aiops_tools() -> None:
    registry = build_registry()
    assert set(registry.names) == {
        "get_service_health",
        "get_recent_logs",
        "get_incident_details",
    }
    assert (
        await registry.call("get_service_health", {"service_name": "payments-api"})
    ).status == "degraded"


@pytest.mark.asyncio
async def test_tool_arguments_are_validated() -> None:
    registry = build_registry()
    try:
        await registry.call("get_service_health", {"service_name": "payments-api", "extra": True})
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected tool arguments must be rejected")


@pytest.mark.asyncio
async def test_mock_health_fixture_only_matches_its_validated_service_arguments() -> None:
    database_fixture = {
        "service_name": "database",
        "status": "degraded",
        "latency_ms": {"p50": 1, "p95": 2, "p99": 3},
        "error_rate": 0.5,
        "resource_utilization": {"cpu": 1, "memory": 1},
    }
    registry = build_registry(
        mock_responses={
            "get_service_health": [
                {
                    "arguments": {"service_name": "database"},
                    "response": database_fixture,
                }
            ]
        }
    )

    database = await registry.call("get_service_health", {"service_name": "database"})
    payments = await registry.call("get_service_health", {"service_name": "payments-api"})

    assert database.error_rate == 0.5
    assert payments.error_rate == 0.087
