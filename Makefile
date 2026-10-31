.PHONY: help setup dev lint format format-check typecheck test eval eval-mock check

STRATEGY ?= autonomous
OUTPUT_MODE ?= free_text
TRIALS ?= 1
OUTPUT ?= reports/latest.json

help:
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## / - /'

setup: ## Install locked dependencies
	uv sync --locked

dev: ## Start the local API on loopback
	uv run --locked --offline uvicorn agent_harness.api.app:app --host 127.0.0.1 --port 8000

lint: ## Run Ruff checks
	uv run --locked --offline ruff check .

format: ## Format Python files
	uv run --locked --offline ruff format .

format-check: ## Verify formatting
	uv run --locked --offline ruff format --check .

typecheck: ## Run mypy
	uv run --locked --offline mypy src tests

test: ## Run deterministic tests
	uv run --locked --offline pytest

eval: ## Run live model evaluations
	@uv run --locked --offline python -m agent_harness.evals.runner --mode live --strategy $(STRATEGY) --output-mode $(OUTPUT_MODE) --trials $(TRIALS) --output $(OUTPUT)

eval-mock: ## Run deterministic mocked evaluations
	@uv run --locked --offline python -m agent_harness.evals.runner --mode mock --strategy $(STRATEGY) --output-mode $(OUTPUT_MODE) --trials $(TRIALS) --output $(OUTPUT)

check: lint format-check typecheck test ## Run all offline quality checks
