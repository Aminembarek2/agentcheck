"""The contamination probe's parsing and scoring, without an API call."""

from __future__ import annotations

import pytest

from scripts.contamination_probe import overlap, parse_reply, render


def test_a_fenced_json_reply_parses():
    reply = '```json\n{"familiar": true, "files": ["a.py"]}\n```'
    assert parse_reply(reply)["files"] == ["a.py"]


def test_prose_around_the_object_is_tolerated():
    reply = 'Here is what I recall:\n{"files": []}\nHope that helps.'
    assert parse_reply(reply)["files"] == []


def test_an_unparseable_reply_raises_rather_than_reading_as_no_recall():
    """The failure mode this guard exists for.

    Reading "no files" out of prose that failed to parse would report a
    clean contamination check for a model that may have listed every file
    the PR touched.
    """
    with pytest.raises(ValueError, match="no JSON object"):
        parse_reply("I am not sure I remember that repository.")


def test_recall_is_matched_on_the_basename():
    """A model recalling `backends/sqlite.py` has recalled the file.

    Requiring the full prefix would understate recall, which flatters the
    task — the direction of error this whole probe exists to avoid.
    """
    reference = ("databases/backends/sqlite.py", "databases/core.py")
    assert overlap(["backends/sqlite.py"], reference) == \
        ["databases/backends/sqlite.py"]


def test_a_plausible_but_wrong_path_does_not_count():
    reference = ("databases/core.py",)
    assert overlap(["databases/engine.py", "setup.py"], reference) == []


def test_recall_is_deduplicated():
    reference = ("databases/core.py",)
    assert overlap(["core.py", "databases/core.py"], reference) == \
        ["databases/core.py"]


def test_the_report_says_absence_is_not_evidence():
    """The one-directional reading has to survive into the document."""
    text = render([{"task": "t", "model": "m", "model_id": "m-1",
                    "declared_risk": "high", "reference_files": ["a.py"],
                    "raw": "{}", "parsed": {}, "named_files": [],
                    "recalled": []}])
    assert "not** evidence" in text or "not evidence" in text
    assert "Named no file" in text


def test_an_unparseable_probe_is_reported_as_such_in_the_table():
    text = render([{"task": "t", "model": "m", "model_id": "m-1",
                    "declared_risk": "low", "reference_files": ["a.py"],
                    "raw": "nope", "parsed": None,
                    "parse_error": "no JSON object in the reply"}])
    assert "unparseable" in text
