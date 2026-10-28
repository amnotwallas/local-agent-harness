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
