"""The archive holds records that must stay unloadable.

`runs/archive/` exists because 28 early records cannot be scored: they
have no `before_failed_ids`, and progress computed against an absent
before-state is 100%, silently, for every run. That is bug four in
`docs/findings.md` §1.

The test that matters here is the inverse of the usual one. If a future
schema change made these files load, it would mean a field this project
depends on had become optional again — and the archive is the tripwire
for that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcheck.record import IncompatibleRecord, load
from scripts.archive_runs import (
    _missing_from,
    _rejection,
    _schema_of,
    render_manifest,
)

ARCHIVE = Path(__file__).resolve().parent.parent / "runs" / "archive"
ARCHIVED = sorted(ARCHIVE.glob("*.json"))


@pytest.mark.skipif(not ARCHIVED, reason="no archived records in this tree")
@pytest.mark.parametrize("path", ARCHIVED, ids=lambda p: p.name)
def test_archived_records_still_do_not_load(path: Path) -> None:
    """Every archived record must remain rejected.

    A passing `load` here is not good news. It would mean the loader had
    been relaxed to accept a record with no measured before-state, which
    is the one thing the schema check exists to prevent.
    """
    with pytest.raises(IncompatibleRecord):
        load(path)


@pytest.mark.skipif(not ARCHIVED, reason="no archived records in this tree")
def test_archive_is_not_silently_repairable() -> None:
    """None of them carry the field that would make scoring possible."""
    for path in ARCHIVED:
        raw = json.loads(path.read_text())
        assert "before_failed_ids" not in raw, (
            f"{path.name} has a before-state — it should have been "
            f"re-scored rather than archived")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="no archive in this tree")
def test_manifest_covers_every_archived_record() -> None:
    manifest = (ARCHIVE / "MANIFEST.md").read_text()
    for path in ARCHIVED:
        assert path.name in manifest, f"{path.name} missing from MANIFEST.md"


def test_rejection_returns_none_for_a_loadable_record(tmp_path: Path,
                                                      good_record) -> None:
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(good_record))
    assert _rejection(path) is None


def test_rejection_strips_the_filename_prefix(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"task_id": "x"}))
    reason = _rejection(path)
    assert reason is not None
    assert not reason.startswith("bad.json"), \
        "the manifest has a filename column already"


def test_schema_and_missing_are_parsed_from_the_loader_message() -> None:
    reason = ("schema v2 (current v3); missing "
              "['before_failed_ids', 'verdict_status']. Re-run rather than "
              "re-score — the data is not there.")
    assert _schema_of(reason) == "2"
    assert _missing_from(reason) == "before_failed_ids, verdict_status"


def test_manifest_names_why_migration_is_refused() -> None:
    text = render_manifest([{"name": "a.json", "schema": "2",
                             "missing": "before_failed_ids", "sha": "abc1234",
                             "date": "2026-01-01"}])
    assert "a.json" in text
    # The reason has to survive in the generated text; a manifest that
    # only lists filenames invites someone to try migrating them.
    assert "cannot be reconstructed" in text
    assert "100%" in text
