# Evaluations

The harness provides two evaluation modes over the versioned scenarios in `evals/datasets/aiops.json`.

| Mode | Model | Tool data | Measures |
| --- | --- | --- | --- |
| `mock` | Deterministic transcripts | Fixed sample fixtures | Runtime and harness behavior; not model tool selection |
| `live` | Configured llama.cpp model | Fixed sample fixtures | Model behavior against the same scenarios |

Both modes report task success using expected tool calls, argument validity, required-fact coverage, response shape, execution status, and expected tool errors. Required-fact coverage does not prove the answer is true. Unsupported claims are tracked separately and are not inferred from keyword matches.

## Commands

```sh
# Deterministic offline run
make eval-mock

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

Reports are written under `reports/`, which is ignored by Git. Schema mode sends a JSON schema response format to llama.cpp and records server acceptance separately from Pydantic validation. Native tool calling depends on the model's chat template; if it fails consistently, select `TOOL_PROTOCOL=json` and rerun.
