from _pytest.monkeypatch import MonkeyPatch

from agent_harness.config import Settings


def test_settings_read_model_endpoint_and_timeout_from_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("LLM_TIMEOUT", "37")

    settings = Settings.from_env()

    assert settings.llm_base_url == "http://localhost:8080/v1"
    assert settings.llm_timeout == 37
    assert settings.temperature == 0


def test_settings_rejects_iteration_limit_above_five() -> None:
    import pytest

    with pytest.raises(ValueError, match="between 1 and 5"):
        Settings(max_iterations=6)


def test_settings_read_execution_limits_from_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EXECUTION_TIMEOUT", "42")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "7")

    settings = Settings.from_env()

    assert settings.execution_timeout == 42
    assert settings.max_tool_calls == 7
