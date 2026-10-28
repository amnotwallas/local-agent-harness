from collections.abc import Awaitable, Callable
from typing import Any, get_origin

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceHealthInput(StrictModel):
    service_name: str = Field(min_length=1, max_length=80)


class LogsInput(StrictModel):
    service_name: str | None = Field(default=None, max_length=80)
    severity: str | None = Field(default=None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    limit: int = Field(default=20, ge=1, le=100)


class IncidentInput(StrictModel):
    incident_id: str = Field(min_length=1, max_length=80)


class ServiceHealth(StrictModel):
    service_name: str
    status: str
    latency_ms: dict[str, float]
    error_rate: float
    resource_utilization: dict[str, float]


class LogEntry(StrictModel):
    timestamp: str
    service_name: str
    severity: str
    message: str


class IncidentDetails(StrictModel):
    incident_id: str
    timeline: list[dict[str, str]]
    symptoms: list[str]
    related_services: list[str]
    evidence: list[str]


HEALTH = {
    "payments-api": {
        "service_name": "payments-api",
        "status": "degraded",
        "latency_ms": {"p50": 82, "p95": 940, "p99": 1830},
        "error_rate": 0.087,
        "resource_utilization": {"cpu": 0.74, "memory": 0.68},
    },
    "orders-api": {
        "service_name": "orders-api",
        "status": "healthy",
        "latency_ms": {"p50": 35, "p95": 110, "p99": 190},
        "error_rate": 0.001,
        "resource_utilization": {"cpu": 0.32, "memory": 0.57},
    },
    "database": {
        "service_name": "database",
        "status": "degraded",
        "latency_ms": {"p50": 12, "p95": 480, "p99": 900},
        "error_rate": 0.021,
        "resource_utilization": {"cpu": 0.91, "memory": 0.83},
    },
}
LOGS = [
    {
        "timestamp": "2026-10-28T09:12:00Z",
        "service_name": "payments-api",
        "severity": "ERROR",
        "message": "upstream request timed out after 2s",
    },
    {
        "timestamp": "2026-10-28T09:11:30Z",
        "service_name": "database",
        "severity": "WARNING",
        "message": "connection pool at capacity",
    },
    {
        "timestamp": "2026-10-28T09:10:00Z",
        "service_name": "orders-api",
        "severity": "INFO",
        "message": "health check passed",
    },
]
INCIDENTS = {
    "INC-1042": {
        "incident_id": "INC-1042",
        "timeline": [
            {"timestamp": "2026-10-28T09:00:00Z", "event": "deployment started"},
            {"timestamp": "2026-10-28T09:08:00Z", "event": "payment errors increased"},
        ],
        "symptoms": ["elevated payment latency", "increased 5xx rate"],
        "related_services": ["payments-api", "database"],
        "evidence": ["deployment preceded symptoms", "connection pool warnings"],
    }
}


class ToolRegistry:
    def __init__(
        self, allowed: set[str] | None = None, mock_responses: dict[str, Any] | None = None
    ) -> None:
        self._allowed = allowed
        self._mock_responses = mock_responses or {}
        self._mock_indices: dict[str, int] = {}
        self._tools: dict[str, tuple[type[BaseModel], Any, Callable[[Any], Awaitable[Any]]]] = {}
        self.register("get_service_health", ServiceHealthInput, ServiceHealth, self._health)
        self.register("get_recent_logs", LogsInput, list[LogEntry], self._logs)
        self.register("get_incident_details", IncidentInput, IncidentDetails, self._incident)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in self._tools if self._allowed is None or name in self._allowed)

    def subset(self, allowed: set[str]) -> "ToolRegistry":
        return ToolRegistry(allowed, self._mock_responses)

    def register(
        self,
        name: str,
        input_model: type[BaseModel],
        output_model: Any,
        fn: Callable[[Any], Awaitable[Any]],
    ) -> None:
        self._tools[name] = (input_model, output_model, fn)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name.replace("_", " "),
                    "parameters": self._tools[name][0].model_json_schema(),
                },
            }
            for name in self.names
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        parsed = self.validate(name, arguments)
        _, output_model, fn = self._tools[name]
        result = await fn(parsed)
        if name in self._mock_responses:
            index = self._mock_indices.get(name, 0)
            fixture = self._mock_responses[name]
            if isinstance(fixture, list) and get_origin(output_model) is not list:
                fixture = fixture[min(index, len(fixture) - 1)]
            self._mock_indices[name] = index + 1
            result = fixture
        return TypeAdapter(output_model).validate_python(result)

    def validate(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        if name not in self.names:
            raise ValueError(f"Tool is not allowed: {name}")
        input_model, _, _ = self._tools[name]
        try:
            return input_model.model_validate(arguments)
        except Exception as exc:
            raise ValueError(f"Invalid arguments for tool: {name}") from exc

    @staticmethod
    async def _health(args: ServiceHealthInput) -> ServiceHealth:
        if args.service_name not in HEALTH:
            raise ValueError("Unknown service")
        return ServiceHealth.model_validate(HEALTH[args.service_name])

    @staticmethod
    async def _logs(args: LogsInput) -> list[LogEntry]:
        logs = [
            row
            for row in LOGS
            if (not args.service_name or row["service_name"] == args.service_name)
            and (not args.severity or row["severity"] == args.severity)
        ]
        return [LogEntry.model_validate(row) for row in logs[: args.limit]]

    @staticmethod
    async def _incident(args: IncidentInput) -> IncidentDetails:
        if args.incident_id not in INCIDENTS:
            raise ValueError("Unknown incident")
        return IncidentDetails.model_validate(INCIDENTS[args.incident_id])


def build_registry(
    allowed: set[str] | None = None, mock_responses: dict[str, Any] | None = None
) -> ToolRegistry:
    return ToolRegistry(allowed, mock_responses)
