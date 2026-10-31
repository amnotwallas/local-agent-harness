# Usage

## Configuration

Copy `.env.example` to `.env` and set `LLM_BASE_URL` to the `/v1` endpoint of your llama.cpp server. The harness loads `.env` automatically from the working directory; exported environment variables take precedence. If `LLM_MODEL` is unset, the client discovers an available model from `/v1/models`.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible model API |
| `LLM_MODEL` | discovered | Model ID sent to the server |
| `LLM_TIMEOUT` | `120` | Timeout for each model request, in seconds |
| `LLM_TEMPERATURE` | `0` | Model sampling temperature |
| `TOOL_PROTOCOL` | `native` | `native` tool calls or `json` fallback |
| `AGENT_MAX_ITERATIONS` | `5` | Maximum agent iterations (up to 5) |
| `AGENT_MAX_TOOL_CALLS` | `10` | Tool-call limit for a run |
| `AGENT_EXECUTION_TIMEOUT` | `300` | End-to-end run timeout, in seconds |
| `TOOL_TIMEOUT` | `10` | Tool timeout, in seconds |
| `LLM_MAX_TOKENS` | `1024` | Maximum generated tokens |

The API binds to `127.0.0.1:8000` by default. The sample tools are in-memory fixtures and make no external monitoring or incident-system requests.

## Run an agent

`POST /agents/run` accepts a task and optional tool, strategy, and output settings:

```json
{
  "task": "Check payments-api and summarize its recent errors",
  "allowed_tools": ["get_service_health", "get_recent_logs"],
  "strategy": "autonomous",
  "output_mode": "free_text"
}
```

| Field | Values | Default | Notes |
| --- | --- | --- | --- |
| `task` | Text, 1–8000 characters | Required | The question for the agent |
| `allowed_tools` | `get_service_health`, `get_recent_logs`, `get_incident_details` | All tools | Restricts tools available in this run |
| `strategy` | `autonomous`, `guided`, `deterministic` | `autonomous` | Deterministic mode also requires a valid `workflow` |
| `workflow` | A registered workflow name | — | Only valid with `deterministic` |
| `output_mode` | `free_text`, `prompt_json`, `schema_json` | `free_text` | JSON modes request a validated incident report |

The response includes the final answer, tool history, execution status, timing, and format-compliance metadata. In JSON output modes, a valid report is returned in `incident_report`; invalid output remains visible in `final_answer` and is marked as non-compliant.

## Available tools

| Tool | Data returned |
| --- | --- |
| `get_service_health` | Status, latency, error rate, CPU, and memory for a sample service |
| `get_recent_logs` | Sample log entries, optionally filtered by service and severity |
| `get_incident_details` | Timeline, symptoms, related services, and evidence for a sample incident |

These tools use fixed sample records in `src/agent_harness/tools/registry.py`; they are not live integrations.
