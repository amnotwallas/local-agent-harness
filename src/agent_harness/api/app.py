import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent_harness.agents.runtime import AgentResult, AgentRuntime
from agent_harness.agents.workflows import WORKFLOW_NAMES
from agent_harness.config import Settings
from agent_harness.logging_config import configure_logging, request_id_var
from agent_harness.models.client import OpenAIClient
from agent_harness.tools.registry import build_registry

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    allowed_tools: list[str] | None = None
    strategy: Literal["autonomous", "guided", "deterministic"] = "autonomous"
    workflow: str | None = None
    output_mode: Literal["free_text", "prompt_json", "schema_json"] = "free_text"


class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None


def create_app(settings: Settings | None = None, model: Any | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    registry = build_registry()
    client = model or OpenAIClient(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/server.jsonl") or None,
        )
        yield
        if isinstance(client, OpenAIClient):
            await client.close()

    app = FastAPI(title="Agent Harness", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_request(request: Request, call_next: Any) -> Any:
        incoming_id = request.headers.get("x-request-id", "")
        request_id = (
            incoming_id
            if len(incoming_id) <= 128 and re.fullmatch(r"[A-Za-z0-9._-]+", incoming_id)
            else str(uuid4())
        )
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            logger.info(
                "HTTP request completed",
                extra={
                    "event": "http.request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": getattr(request.scope.get("route"), "path", "<unmatched>"),
                    "status_code": status_code,
                    "latency_ms": (perf_counter() - started) * 1000,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            request_id_var.reset(token)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = request.state.request_id
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request", "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.state.request_id
        logger.error(
            "API request failed",
            extra={
                "event": "http.error",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    async def tools() -> dict[str, list[dict[str, Any]]]:
        return {"tools": registry.schemas()}

    @app.post(
        "/agents/run",
        response_model=AgentResult,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            504: {"model": AgentResult},
        },
    )
    async def run_agent(body: RunRequest, request: Request) -> Any:
        if body.strategy == "deterministic" and body.workflow not in WORKFLOW_NAMES:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Deterministic strategy requires a valid workflow",
                    "request_id": request.state.request_id,
                },
            )
        if body.strategy != "deterministic" and body.workflow is not None:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Workflow is only valid for the deterministic strategy",
                    "request_id": request.state.request_id,
                },
            )
        if body.allowed_tools is not None and not set(body.allowed_tools) <= set(registry.names):
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Requested tool is not available",
                    "request_id": request.state.request_id,
                },
            )
        run_registry = (
            registry.subset(set(body.allowed_tools)) if body.allowed_tools is not None else registry
        )
        result = await AgentRuntime(client, run_registry, config).run(
            body.task,
            request.state.request_id,
            strategy=body.strategy,
            workflow=body.workflow,
            output_mode=body.output_mode,
        )
        model_id = getattr(client, "model_id", None)
        result.model_id = model_id if isinstance(model_id, str) else None
        if result.status == "deadline_exceeded":
            return JSONResponse(status_code=504, content=result.model_dump(mode="json"))
        if result.status == "failed" and result.termination_reason == "model_or_protocol_error":
            return JSONResponse(
                status_code=502,
                content=ErrorResponse(
                    error="Model request failed", request_id=result.request_id
                ).model_dump(),
            )
        return result

    return app


app = create_app()
