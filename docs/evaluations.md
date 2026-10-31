# Evaluations

The harness provides two evaluation modes over the versioned scenarios in `evals/datasets/aiops.json`.

| Mode | Model | Tool data | Measures |
| --- | --- | --- | --- |
| `mock` | Deterministic transcripts | Fixed sample fixtures | Runtime and harness behavior; not model tool selection |
| `live` | Configured llama.cpp model through the running API | API read-only sample tools | Model behavior against the same scenarios |

Both modes report task success using expected tool calls, argument validity, required-fact coverage, response shape, execution status, and expected tool errors. Required-fact coverage does not prove the answer is true. Unsupported claims are tracked separately and are not inferred from keyword matches.

## Commands

```sh
# Deterministic offline run
make eval-mock

# In another terminal, start the API before a live run
make dev

# Compare guided strategy and prompted JSON over three live trials
make eval STRATEGY=guided OUTPUT_MODE=prompt_json TRIALS=3 \
  OUTPUT=reports/guided-prompt-json.json

# Compare a run with a prior report
uv run --locked python -m agent_harness.evals.runner \
  --mode live --trials 3 --baseline reports/baseline.json
```

| Option | Values | Default |
| --- | --- | --- |
| `STRATEGY` | `autonomous`, `guided`, `deterministic` | `autonomous` |
| `OUTPUT_MODE` | `free_text`, `prompt_json`, `schema_json` | `free_text` |
| `TRIALS` | Positive integer | `1` |
| `OUTPUT` | Report file path | `reports/latest.json` |
| `API_BASE_URL` | API origin for live runs | `http://127.0.0.1:8000` |

`make eval` requires `make dev` and sends each task to `/agents/run`; `make eval-mock` keeps its deterministic in-process runtime and needs no server. The terminal shows one readable summary after the run. Full per-trial results and the baseline comparison are saved under `reports/`, which is ignored by Git. Schema mode sends a JSON schema response format to llama.cpp and records server acceptance separately from Pydantic validation. Native tool calling depends on the model's chat template; if it fails consistently, select `TOOL_PROTOCOL=json` and rerun.

For live runs, reported latency is the full API round trip; `runtime_latency_ms` keeps the server's agent execution time when available. The report identifies the API as the source of model behavior and does not claim to know its internal settings or schema. A brief health check runs first; model requests allow the server's maximum execution window. API errors during a scenario appear as infrastructure failures in the report.
