import re
from collections.abc import Callable

from agent_harness.models.client import ToolCall

_SERVICE = re.compile(r"\b(?:[a-z][a-z0-9]*(?:-[a-z0-9]+)*-api|database)\b", re.IGNORECASE)
_INCIDENT = re.compile(r"\bINC-\d+\b", re.IGNORECASE)


def _services(task: str) -> list[str]:
    return list(dict.fromkeys(match.group().lower() for match in _SERVICE.finditer(task)))


def _incident(task: str) -> str:
    match = _INCIDENT.search(task)
    if match is None:
        raise ValueError("Workflow needs an incident ID")
    return match.group().upper()


def _one_service(task: str) -> str:
    services = _services(task)
    if len(services) != 1:
        raise ValueError("Workflow needs exactly one named service")
    return services[0]


def _call(index: int, name: str, arguments: dict[str, str | int | None]) -> ToolCall:
    return ToolCall(id=f"workflow-{index}", name=name, arguments=arguments)


def _service_health(task: str) -> list[ToolCall]:
    return [_call(1, "get_service_health", {"service_name": _one_service(task)})]


def _service_diagnosis(task: str) -> list[ToolCall]:
    service = _one_service(task)
    lowered = task.casefold()
    severity = (
        "WARNING"
        if any(word in lowered for word in ("warning", "capacity", "pool"))
        else "ERROR"
        if any(word in lowered for word in ("error", "timeout", "latency"))
        else None
    )
    return [
        _call(1, "get_service_health", {"service_name": service}),
        _call(
            2,
            "get_recent_logs",
            {"service_name": service, "severity": severity, "limit": 20},
        ),
    ]


def _recent_error_logs(task: str) -> list[ToolCall]:
    return [
        _call(
            1,
            "get_recent_logs",
            {"service_name": _one_service(task), "severity": "ERROR", "limit": 20},
        )
    ]


def _incident_summary(task: str) -> list[ToolCall]:
    return [_call(1, "get_incident_details", {"incident_id": _incident(task)})]


def _compare_services(task: str) -> list[ToolCall]:
    services = _services(task)
    if len(services) < 2:
        raise ValueError("Workflow needs at least two named services")
    return [
        _call(index, "get_service_health", {"service_name": service})
        for index, service in enumerate(services, start=1)
    ]


def _incident_cause_review(task: str) -> list[ToolCall]:
    incident = _incident(task)
    services = _services(task)
    if not services:
        raise ValueError("Workflow needs a named service")
    calls = [_call(1, "get_incident_details", {"incident_id": incident})]
    calls.extend(
        _call(index, "get_service_health", {"service_name": service})
        for index, service in enumerate(services, start=2)
    )
    return calls


WORKFLOWS: dict[str, Callable[[str], list[ToolCall]]] = {
    "service_health": _service_health,
    "service_diagnosis": _service_diagnosis,
    "recent_error_logs": _recent_error_logs,
    "incident_summary": _incident_summary,
    "compare_services": _compare_services,
    "incident_cause_review": _incident_cause_review,
}
WORKFLOW_NAMES = frozenset(WORKFLOWS)


def build_workflow_plan(name: str, task: str) -> list[ToolCall]:
    try:
        workflow = WORKFLOWS[name]
    except KeyError as exc:
        raise ValueError("Unknown deterministic workflow") from exc
    return workflow(task)
