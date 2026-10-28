import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExpectedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_id: str
    category: str
    task: str
    expected_tool_calls: list[ExpectedCall]
    allowed_tool_calls: list[ExpectedCall] = Field(default_factory=list)
    forbidden_tool_calls: list[ExpectedCall] = Field(default_factory=list)
    required_facts: dict[str, list[str]]
    unsupported_claims: list[str]
    expected_output_structure: list[str]
    max_tool_iterations: int
    mock_tool_responses: dict[str, list[dict[str, Any]]]
    expected_tool_errors: int = 0


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    dataset = path or Path(__file__).resolve().parents[3] / "evals" / "datasets" / "aiops.json"
    return [Scenario.model_validate(row) for row in json.loads(dataset.read_text())]
