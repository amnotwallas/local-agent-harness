# Agent Harness

A small local FastAPI harness for AIOps experiments with llama.cpp's OpenAI-compatible API. It runs deterministic, read-only health, log, and incident tools; it does not run shell commands, make arbitrary HTTP requests, or modify files.

## Setup

Requires Python 3.12+ and uv.

```sh
make setup
cp .env.example .env
# Adjust .env as needed, load it into the shell, then start the API.
set -a; source .env; set +a
make dev
```

The API binds to `127.0.0.1:8000`. Endpoints are `GET /health`, `GET /tools`, and `POST /agents/run` with `{"task":"...", "allowed_tools":[...]}`. The run response and error shapes are published in OpenAPI. If `LLM_MODEL` is unset, the client discovers an ID from `/v1/models`. Default inference uses temperature 0, a 120 second request timeout, and the llama.cpp `chat_template_kwargs.enable_thinking=false` setting. `AGENT_EXECUTION_TIMEOUT` bounds the complete run (default 300 seconds), and `AGENT_MAX_TOOL_CALLS` caps tools across all iterations (default 10). `TOOL_PROTOCOL=native` uses OpenAI tool calls. Set `TOOL_PROTOCOL=json` explicitly to use the constrained JSON protocol; it asks the model for a JSON object and rejects malformed responses rather than switching modes.

## Development

Use Makefile targets for setup, dev, lint, format, format-check, typecheck, test, and check. `make check` is offline and does not contact the inference server. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

## Evaluations

`make eval-mock` runs the shared runtime against the versioned 10-scenario fixture dataset with deterministic model transcripts and mocked tool data. It validates harness behavior; it does not measure model tool selection. Both modes score task success using exact expected calls, valid arguments, required fact presence, response schema, execution status, and expected tool errors. Required fact presence is coverage, not proof of truth. Unsupported semantic claims remain `unverified` and do not get guessed by keyword matching.

`make eval` runs the same scenarios through the configured model. Both write JSON to `reports/latest.json` (ignored by Git). For repeated trials or a baseline comparison, run `uv run --locked python -m agent_harness.evals.runner --mode live --trials 3 --baseline reports/baseline.json`.

The live Qwen3-4B model and server are external dependencies. Native tool calling depends on the model's chat template; inspect live evaluation results before relying on it. The report labels mock and live evaluation scope separately. If native calling fails consistently, explicitly configure the JSON fallback and rerun.
