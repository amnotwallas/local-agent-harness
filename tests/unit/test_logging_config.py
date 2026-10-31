import json
import logging
from pathlib import Path

import pytest

import agent_harness.logging_config as logging_config
from agent_harness.logging_config import (
    ColoredFormatter,
    JsonFormatter,
    configure_logging,
    request_id_var,
)


def test_json_formatter_includes_context_and_allowlisted_metrics_only() -> None:
    token = request_id_var.set("request-123")
    try:
        record = logging.LogRecord(
            "agent_harness.test", logging.INFO, __file__, 1, "Model completed", (), None
        )
        record.event = "llm.completed"
        record.input_tokens = 8
        record.secret = "do not log this"
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["request_id"] == "request-123"
    assert payload["trace_id"] == "request-123"
    assert payload["event"] == "llm.completed"
    assert payload["input_tokens"] == 8
    assert payload["timestamp"].endswith("+00:00")
    assert "secret" not in payload


def test_configure_logging_writes_json_file_once(tmp_path: Path) -> None:
    logger = logging.getLogger("agent_harness")
    old_level, old_propagate = logger.level, logger.propagate
    log_file = tmp_path / "server.jsonl"
    try:
        configure_logging(log_file=log_file)
        configure_logging(log_file=log_file)
        logger.info("Server started", extra={"event": "server.started"})
        lines = log_file.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "server.started"
        assert len(logger.handlers) == 2
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_console_formatter_shows_safe_metrics_without_extra_fields() -> None:
    record = logging.LogRecord("agent_harness.test", logging.INFO, __file__, 1, "Done", (), None)
    record.event = "http.request"
    record.method = "POST"
    record.path = "/agents/run"
    record.status_code = 200
    record.secret = "private body"

    line = ColoredFormatter(use_color=False).format(record)

    assert "http.request" in line
    assert 'method="POST"' in line
    assert 'path="/agents/run"' in line
    assert "status_code=200" in line
    assert "private body" not in line


def test_broken_optional_tracer_falls_back_to_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenTrace:
        def get_current_span(self) -> None:
            raise RuntimeError("tracer unavailable")

    monkeypatch.setattr(logging_config, "_otel_trace", BrokenTrace())
    token = request_id_var.set("request-123")
    try:
        record = logging.LogRecord(
            "agent_harness.test", logging.INFO, __file__, 1, "Done", (), None
        )
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["trace_id"] == "request-123"


def test_eval_logging_keeps_file_events_out_of_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = logging.getLogger("agent_harness")
    old_level, old_propagate = logger.level, logger.propagate
    log_file = tmp_path / "eval.jsonl"
    try:
        configure_logging(log_file=log_file, console=False)
        logger.info("Evaluation event", extra={"event": "agent.finished"})
        assert capsys.readouterr().out == ""
        assert json.loads(log_file.read_text().splitlines()[0])["event"] == "agent.finished"
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_eval_logging_has_no_early_console_warning_when_file_is_unwritable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = logging.getLogger("agent_harness")
    old_level, old_propagate = logger.level, logger.propagate
    try:
        configure_logging(log_file=tmp_path, console=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate
