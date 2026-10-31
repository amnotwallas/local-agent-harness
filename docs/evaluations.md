# Evaluations

The harness provides two evaluation modes over the versioned scenarios in `evals/datasets/aiops.json`.

| Mode | Model | Tool data | Measures |
| --- | --- | --- | --- |
| `mock` | Deterministic transcripts | Fixed sample fixtures | Runtime and harness behavior; not model tool selection |
| `live` | Configured llama.cpp model through the running API | API read-only sample tools | Model behavior against the same scenarios |

Both modes report task success using expected tool calls, argument validity, required-fact coverage, response shape, execution status, and expected tool errors. Required-fact coverage does not prove the answer is true. Unsupported claims are tracked separately and are not inferred from keyword matches.

## Example output and how to read it

`make eval` prints a compact terminal summary and writes per-trial details to the JSON report. This excerpt is from a saved live run; scores and timings will vary with the model and configuration:

```text
Evaluación LIVE | autonomous | free_text
Escenarios: 10 | Intentos por escenario: 1 | Ejecuciones: 10
API: http://127.0.0.1:8000 | Modelo: Qwen3-4B-Q4_K_M.gguf

Resultados
  Éxito: 5/10 (50%) | Completadas: 10/10 (100%) | Fallos de infraestructura: 0
  Selección de herramientas: 7/10 (70%) | Fallos de aserciones: 4
Cobertura
  Evidencia obtenida: 8/14 (57%) (10 intentos)
  Frases requeridas presentes: 17/21 (81%) | Fuentes atribuidas: 0/0 (n/a)
Formato
  Válido: 0 | Inválido: 0 | No producido: 0 | No aplica: 10
  Esquema aceptado: 0 | Rechazado: 0 | Desconocido: 0 | No solicitado: 10
Rendimiento
  Latencia media: 2071.0 ms | Mediana: 2073.3 ms
  Ejecución media del servidor: 2066.9 ms
  Herramientas por ejecución: 1.1 | Errores de herramienta: 0.2
  Tokens medios: 1432.6 entrada + 155.5 salida

Por escenario (éxito / selección / aserciones / infraestructura | latencia media)
  healthy-service          1/1 | 1/1 | 1/1 | 0/1 | 2163.3 ms
  payment-latency-spike    0/1 | 0/1 | 0/1 | 0/1 | 2234.9 ms

Fallos destacados
  payment-latency-spike intento 1: causas: selección de herramientas; frases faltantes: timeout

Reporte JSON completo: reports/latest.json
```

Interpret the sections this way:

- **Éxito** counts scenarios that met the task criteria. **Completadas** only means execution ended normally; a completed run can still fail evaluation. Here, all 10 requests completed, but only 5 met all criteria.
- **Selección de herramientas** checks whether the agent made the expected tool calls with valid arguments. The per-scenario columns show success, tool selection, assertions, and infrastructure failures, in that order.
- **Evidencia obtenida** measures expected evidence retrieved. **Frases requeridas presentes** checks for configured phrases in the answer; it does not verify truth or whether the evidence supports the claim. `Fuentes atribuidas: 0/0 (n/a)` means this run had no source-attribution checks configured.
- **No aplica** and **No solicitado** mean this run did not request structured output or a response schema; they are not failures.
- **Latencia media** is the full API round trip for live evaluations; **ejecución media del servidor** is the agent runtime reported by the API. Token and tool counts are averages per execution.
- **Fallos de infraestructura** are API/transport failures. A task failure with zero infrastructure failures points to scenario criteria, such as a missing tool call or required phrase. The JSON report contains the full details for each trial.

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
