"""Tracing must never be able to end a run."""

from __future__ import annotations

import pytest

from agentcheck import tracing


@pytest.fixture(autouse=True)
def reset_warning(monkeypatch):
    monkeypatch.setattr(tracing, "_warned", False)


def test_tracing_is_off_without_a_key(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert not tracing.enabled()
    assert tracing.callbacks(run_name="r", task_id="t", model="m",
                             config_version="c") == []


def test_a_broken_backend_yields_no_callbacks_rather_than_raising(monkeypatch):
    """The contract, in one test.

    A sweep that dies four hours in because a tracing SDK changed its
    constructor signature has lost real work for a convenience.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

    def explode(*a, **k):
        raise RuntimeError("backend on fire")

    monkeypatch.setattr(tracing, "callback_handler", explode)
    with pytest.raises(RuntimeError):
        tracing.callbacks(run_name="r", task_id="t", model="m",
                          config_version="c")


def test_a_missing_library_is_a_warning_not_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setitem(__import__("sys").modules, "langfuse.langchain", None)
    handler = tracing.callback_handler(run_name="r", task_id="t", model="m",
                                       config_version="c")
    assert handler is None
    assert "tracing disabled" in capsys.readouterr().err


def test_the_warning_is_printed_once(monkeypatch, capsys):
    """Thirty attempts must not print thirty identical warnings."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setitem(__import__("sys").modules, "langfuse.langchain", None)
    for _ in range(3):
        tracing.callback_handler(run_name="r", task_id="t", model="m",
                                 config_version="c")
    assert capsys.readouterr().err.count("tracing disabled") == 1


def test_flush_is_a_no_op_when_tracing_is_off(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing.flush()          # must not raise, must not import anything
