# Agent Harness Instructions

## Scope and architecture

This directory is the independent local AIOps harness; keep project files here. `api/` owns FastAPI contracts, `agents/` owns execution and tracing, `models/` owns llama.cpp requests, `tools/` owns typed read-only tool contracts, and `evals/` owns scenario loading, validation, and reports.

## Conventions

- Use Python 3.12+, full type hints, Pydantic v2 validation, Ruff, mypy, and pytest.
- Manage dependencies only with uv. Add runtime or development dependencies in `pyproject.toml`, regenerate `uv.lock`, and use `uv sync --locked` / `uv run --locked`.
- Use Makefile targets for setup, dev, lint, format, format-check, typecheck, test, eval, eval-mock, and check.
- Inspect the current architecture and callers before changes. Preserve module boundaries and avoid unnecessary dependencies, files, and abstractions.
- Tools must remain explicitly allowlisted, bounded, read-only, deterministic in mock mode, and strictly validated. Never add arbitrary shell, filesystem mutation, or unrestricted network tools.
- The runtime must cap iterations at five and tool calls at the configured total, enforce an end-to-end deadline, handle cancellation and malformed model outputs, and record traces without logging secrets or prompts.
- Keep mock evaluations deterministic and share the runtime with the API. Mock scores measure harness behavior; live scores measure model behavior. Do not add LLM-as-a-judge or treat required fact presence as proof of truth; label unverifiable claims accordingly.
- Do not modify or reinstall the external llama.cpp server.

## Definition of done

Run relevant tests and `make check`; run `make eval-mock` for evaluation changes. Report exact verification results, and do not claim live behavior without a successful live run. Keep generated reports ignored and avoid unrelated examples, temporary notes, redundant docs, or unnecessary directories.
