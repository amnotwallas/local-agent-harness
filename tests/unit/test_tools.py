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
