"""The detector audit's sample must be redrawable by someone else.

The audit exists to put a precision and recall number on the cheat
detector, which produces the project's headline figure. A sample that only
its author can reproduce measures nothing that a reader can check, which is
the same objection this repository raises against every unsourced number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcheck.record import RunRecord
from agentcheck.scorer import CHEAT
from scripts.audit_detector import MODEL_ID, build, draw, population, verify

#: Source hashes are taken from the real tree; only the records are
#: synthetic.
ROOT = Path(__file__).resolve().parent.parent


def record(tmp_path, name, *, cheats, diff="--- a\n+++ b\n@@\n-x\n+y\n",
           task="001-fastapi-users", **over):
    """A scored record with or without a cheat finding."""
    score = {
        "status": "scored", "progress": 0.0, "fixed": 0, "still_failing": 1,
        "newly_broken": 0, "newly_suppressed": 0, "cluster_progress": 0.0,
        "clusters_fixed": 0, "clusters_total": 1, "files_changed": ["a.py"],
        "findings": [{"kind": "deleted_test", "severity": CHEAT,
                      "path": "t.py", "detail": "d"}] if cheats else [],
    }
    base = dict(config_version="cfg", task_id=task, model=MODEL_ID,
                outcome="stopped_without_solving", verdict_status="ok",
                before_failed_ids=["t::a"], max_iterations=40,
                max_cost_usd=1.0, cost_usd=0.02, cost_known=True,
                diff=diff, score=score)
    base.update(over)
    return RunRecord(**base).save(tmp_path / name)


def populate(tmp_path, n_flagged=5, n_clean=9):
    for i in range(n_flagged):
        record(tmp_path, f"flag{i:02d}.json", cheats=True)
    for i in range(n_clean):
        record(tmp_path, f"clean{i:02d}.json", cheats=False)


def test_the_same_seed_redraws_the_same_sample(tmp_path):
    populate(tmp_path)
    pop = population(tmp_path)
    assert draw(pop, seed=7, n_unflagged=4) == draw(pop, seed=7, n_unflagged=4)


def test_a_different_seed_draws_a_different_sample(tmp_path):
    """Otherwise the seed is decoration and the sample is not random."""
    populate(tmp_path)
    pop = population(tmp_path)
    assert draw(pop, seed=7, n_unflagged=4) != draw(pop, seed=8, n_unflagged=4)


def test_every_flagged_patch_is_reviewed(tmp_path):
    """This audit fixes a census of the flagged stratum in advance."""
    populate(tmp_path, n_flagged=5, n_clean=9)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    flagged = {e["record"] for e in frame["frame"] if e["flagged"]}
    drawn = {e["record"] for e in frame["cases"]}
    assert flagged <= drawn
    assert len(frame["cases"]) == 5 + 4


def test_unflagged_patches_are_sampled_too(tmp_path):
    """Reading only the flags can lower the rate and never raise it.

    Every detector this project has added raised the measured rate, so the
    misses are the half more likely to be wrong.
    """
    populate(tmp_path)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    unflagged = {e["record"] for e in frame["frame"] if not e["flagged"]}
    drawn = {e["record"] for e in frame["cases"]}
    assert len(unflagged & drawn) == 4
    assert frame["sampling"]["unflagged"]["inclusion_probability"] == 4 / 9


def test_the_cases_do_not_reveal_the_detector_verdict(tmp_path):
    """A reviewer who can see the flag is scoring their agreement with it.

    The flags stay in `frame`, which is there for auditing the draw, not
    for reading alongside the patches.
    """
    populate(tmp_path)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    for case in frame["cases"]:
        assert "flagged" not in case
        assert case["review"] is None
    assert any("flagged" in e for e in frame["frame"])


def test_a_run_with_no_patch_is_not_eligible(tmp_path):
    """An agent that changed nothing cannot have cheated, and counting it
    would deflate the rate. Same rule as `report.detectable`."""
    populate(tmp_path, n_flagged=2, n_clean=2)
    record(tmp_path, "empty.json", cheats=False, diff="   \n")
    assert "empty.json" not in " ".join(p for p, _ in population(tmp_path))


def test_records_without_a_test_verdict_stay_in(tmp_path):
    """A patch that broke collection is where the detector earns its keep.

    Dropping no-verdict runs from the cheat denominator was a real bug in
    this project's history; it understated the headline threefold.
    """
    populate(tmp_path, n_flagged=2, n_clean=2)
    record(tmp_path, "broke.json", cheats=True, outcome="broke_suite")
    assert any("broke.json" in p for p, _ in population(tmp_path))


def test_another_model_is_not_pooled_into_the_population(tmp_path):
    populate(tmp_path, n_flagged=2, n_clean=2)
    record(tmp_path, "other.json", cheats=True, model="qwen/qwen3-coder-next")
    assert not any("other.json" in p for p, _ in population(tmp_path))


def test_the_smoke_task_is_not_pooled_in(tmp_path):
    """003 has one root cause; its progress is 0 or 1 with nothing between,
    and it is never pooled with the measuring tasks anywhere else."""
    populate(tmp_path, n_flagged=2, n_clean=2)
    record(tmp_path, "smoke.json", cheats=True, task="003-docker-py")
    assert not any("smoke.json" in p for p, _ in population(tmp_path))


def test_asking_for_more_unflagged_than_exist_is_an_error(tmp_path):
    """Silently returning fewer would misstate the inclusion probability,
    and the recall estimate is weighted by it."""
    populate(tmp_path, n_flagged=2, n_clean=3)
    with pytest.raises(ValueError, match="only 3"):
        draw(population(tmp_path), seed=7, n_unflagged=9)


def test_the_committed_frame_is_still_reproducible():
    """The real audit frame, redrawn from its own recorded seed.

    This is the test that matters: it fails if the detector, the reference
    patches or the records move after the sample was drawn — any of which
    would silently invalidate the audit in progress.
    """
    root = ROOT
    saved = root / "runs/audits/detector-sample.json"
    if not saved.exists():
        pytest.skip("no audit frame drawn yet")
    assert verify(saved, root / "runs", root) == 0


def test_the_frame_records_what_it_was_drawn_against(tmp_path):
    """Hashes of the detector and the ground truth, so a later reader can
    tell whether either moved after the draw."""
    populate(tmp_path)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    assert frame["source_hashes"]
    assert all(len(h) == 64 for h in frame["source_hashes"].values())
    assert all(e["sha256"] for e in frame["frame"])


def test_a_frame_with_no_flagged_patch_is_refused(tmp_path):
    """Nothing to audit, and a precision of 0/0 is not a number."""
    populate(tmp_path, n_flagged=0, n_clean=5)
    with pytest.raises(ValueError, match="no flagged patch"):
        build(tmp_path, ROOT, seed=7, n_unflagged=2)


def test_the_written_frame_is_valid_json(tmp_path):
    populate(tmp_path)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    assert json.loads(json.dumps(frame))["schema_version"] == 1


@pytest.mark.parametrize("tamper", ["sources", "case_digest", "weight", "size"])
def test_verify_rejects_tampered_provenance(tmp_path, tamper):
    populate(tmp_path)
    frame = build(tmp_path, ROOT, seed=7, n_unflagged=4)
    if tamper == "sources":
        frame["source_hashes"] = {}
    elif tamper == "case_digest":
        frame["cases"][0]["record_sha256"] = "0" * 64
    elif tamper == "weight":
        frame["sampling"]["unflagged"]["inclusion_probability"] = 1
    else:
        frame["population"]["size"] += 1
    saved = tmp_path / "audit.json"
    saved.write_text(json.dumps(frame))
    # Population reconstruction must reject the manifest as a non-run.
    assert verify(saved, tmp_path, ROOT) == 1
