import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from importlib import import_module
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_otel_trace: ModuleType | None
try:
    _otel_trace = import_module("opentelemetry.trace")
except ImportError:
    _otel_trace = None

_FIELDS = (
    "event",
    "request_id",
    "execution_id",
    "method",
    "path",
    "status_code",
    "latency_ms",
    "client_ip",
    "strategy",
    "workflow",
    "output_mode",
    "status",
    "termination_reason",
    "model_calls",
    "tool_calls",
    "model_duration_ms",
    "provider",
    "model",
    "llm_latency_ms",
    "tool_name",
    "tool_status",
    "tool_latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "format_compliance",
    "response_format_status",
    "error_type",
)
_CONSOLE_FIELDS = (
    "method",
    "path",
    "status_code",
    "status",
    "latency_ms",
    "tool_name",
    "tool_status",
    "tool_latency_ms",
    "model",
    "llm_latency_ms",
    "input_tokens",
    "output_tokens",
    "error_type",
)


def _trace_id(request_id: str | None) -> str:
    if _otel_trace is not None:
        try:
            span = _otel_trace.get_current_span()
            context = span.get_span_context()
            if context.is_valid:
                return format(context.trace_id, "032x")
        except Exception:
            pass
    return request_id or "N/A"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = record.__dict__.get("request_id") or request_id_var.get()
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "message": record.getMessage(),
            "trace_id": _trace_id(request_id),
        }
        if request_id:
            data["request_id"] = request_id
        for field in _FIELDS:
            value = record.__dict__.get(field)
            if value is not None and isinstance(value, str | int | float | bool):
                data[field] = value
        if record.exc_info and record.exc_info[0]:
            data["error_type"] = record.exc_info[0].__name__
        return json.dumps(data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\x1b[38;20m",
        logging.INFO: "\x1b[34;20m",
        logging.WARNING: "\x1b[33;20m",
        logging.ERROR: "\x1b[31;20m",
        logging.CRITICAL: "\x1b[31;1m",
    }

    def __init__(self, use_color: bool = False) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        request_id = record.__dict__.get("request_id") or request_id_var.get()
        trace_id = _trace_id(request_id)
        prefix = f"[{trace_id}] " if trace_id != "N/A" else ""
        event = record.__dict__.get("event")
        label = f"{event}: " if event else ""
        line = f"{record.levelname} {prefix}{label}{record.getMessage()}"
        details = " ".join(
            f"{field}={json.dumps(value, ensure_ascii=False) if isinstance(value, str) else value}"
            for field in _CONSOLE_FIELDS
            if (value := record.__dict__.get(field)) is not None
            and isinstance(value, str | int | float | bool)
        )
        if details:
            line += f" {details}"
        if self.use_color:
            return f"{self._COLORS.get(record.levelno, '')}{line}\x1b[0m"
        return line


def configure_logging(
    log_level: str | int = "INFO",
    log_file: Path | str | None = "logs/server.jsonl",
    console: bool = True,
) -> None:
    logger = logging.getLogger("agent_harness")
    if any(
        handler.name and handler.name.startswith("agent_harness_") for handler in logger.handlers
    ):
        return
    logger.setLevel(log_level)
    logger.propagate = False

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.name = "agent_harness_console"
        console_handler.setFormatter(ColoredFormatter(use_color=sys.stdout.isatty()))
        logger.addHandler(console_handler)
    else:
        null_handler = logging.NullHandler()
        null_handler.name = "agent_harness_null"
        logger.addHandler(null_handler)

    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.name = "agent_harness_file"
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning(
                "Could not initialize file logging",
                extra={"event": "logging.file_unavailable", "error_type": type(exc).__name__},
            )

    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
