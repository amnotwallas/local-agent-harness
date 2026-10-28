import json
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_harness.config import Settings

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0


class OpenAIClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.llm_timeout)
        self._owns_client = client is None
        self._model = settings.model

    @property
    def model_id(self) -> str | None:
        return self._model

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def discover_model(self) -> str:
        if self._model:
            return self._model
        try:
            response = await self._client.get(f"{self.settings.llm_base_url.rstrip('/')}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            candidate = models[0].get("id") if models and isinstance(models[0], dict) else None
            self._model = candidate if isinstance(candidate, str) and candidate else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ModelError("Could not discover a model from the inference server") from exc
        if not self._model:
            raise ModelError("The inference server returned no model identifiers")
        return self._model

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        model = await self.discover_model()
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body.update(tools=tools, tool_choice="auto")
        start = perf_counter()
        try:
            response = await self._client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions", json=body
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]["message"]
            raw_calls = choice.get("tool_calls", [])
            tool_calls = [
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                )
                for call in raw_calls
            ]
            usage = payload.get("usage", {})
            return ChatResult(
                content=choice.get("content"),
                tool_calls=tool_calls,
                raw_tool_calls=raw_calls,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                duration_ms=(perf_counter() - start) * 1000,
            )
        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            ValidationError,
        ) as exc:
            logger.warning("Model request failed (%s)", type(exc).__name__)
            raise ModelError("The inference server returned an invalid or failed response") from exc
