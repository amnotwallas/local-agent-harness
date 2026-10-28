import json

import httpx
import pytest

from agent_harness.config import Settings
from agent_harness.models.client import ModelError, OpenAIClient


@pytest.mark.asyncio
async def test_client_discovers_model_and_parses_tool_calls() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_recent_logs",
                                        "arguments": '{"limit": 2}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = OpenAIClient(Settings(llm_base_url="http://model/v1"), http)
    result = await client.chat([{"role": "user", "content": "logs"}], tools=[{"type": "function"}])

    assert requests[0].url.path == "/v1/models"
    assert requests[1].url.path == "/v1/chat/completions"
    assert requests[1].read() and b"local-model" in requests[1].read()
    assert result.tool_calls[0].arguments == {"limit": 2}
    assert result.prompt_tokens == 8
    await http.aclose()


@pytest.mark.asyncio
async def test_client_sends_and_records_schema_response_format() -> None:
    request_body: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        request_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = OpenAIClient(Settings(llm_base_url="http://model/v1"), http)
    schema_format = {"type": "json_object", "schema": {"type": "object"}}

    result = await client.chat(
        [{"role": "user", "content": "respond"}], response_format=schema_format
    )

    assert request_body["response_format"] == schema_format
    assert result.response_format_status == "accepted"
    assert result.response_format_status_code == 200
    await http.aclose()


@pytest.mark.asyncio
async def test_client_records_response_format_rejection() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        return httpx.Response(400, json={"error": {"message": "unsupported format"}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = OpenAIClient(Settings(llm_base_url="http://model/v1"), http)

    with pytest.raises(ModelError) as error:
        await client.chat(
            [{"role": "user", "content": "respond"}],
            response_format={"type": "json_object", "schema": {"type": "object"}},
        )

    assert error.value.response_format_status == "rejected"
    assert error.value.response_format_status_code == 400
    await http.aclose()


@pytest.mark.asyncio
async def test_client_reports_unavailable_model_server_without_leaking_details() -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="private server detail")
        )
    )
    client = OpenAIClient(Settings(llm_base_url="http://model/v1"), http)

    with pytest.raises(ModelError, match="Could not discover") as error:
        await client.discover_model()

    assert "private server detail" not in str(error.value)
    await http.aclose()
