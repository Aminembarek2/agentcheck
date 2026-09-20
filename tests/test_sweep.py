"""Tests for the sweep matrix.

Pure — no container, no API calls, no subprocesses. The parts that spend
money are a thin shell around run_agent.py; the parts that decide WHAT to
spend it on are here.

    pytest tests/test_sweep.py -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentcheck.record import RunRecord
from scripts.sweep import actual_cost, build_matrix


def test_the_matrix_is_the_full_product():
    cells = build_matrix(["a", "b"], ["m1", "m2"], 3, Path("runs"), [50])
    assert len(cells) == 12
    assert len({(c.task_id, c.model, c.repeat) for c in cells}) == 12


def test_repeats_are_interleaved_across_models():
    """An interrupted sweep must leave an equal number of samples per
    model, not all of one and none of the other. Model-major ordering
    produces four haiku runs and zero deepseek runs when you Ctrl-C
    halfway, and those are not comparable to anything."""
    cells = build_matrix(["t"], ["haiku", "ds-flash"], 4, Path("runs"), [50])
    first_four = cells[:4]
    assert [c.model for c in first_four] == \
        ["haiku", "ds-flash", "haiku", "ds-flash"]
    assert sorted(c.repeat for c in first_four) == [1, 1, 2, 2]


def test_every_cell_has_a_distinct_output_path():
    """Two cells writing the same file would silently overwrite a sample."""
    cells = build_matrix(["a", "b"], ["m1", "m2"], 3, Path("runs"), [50])
    assert len({c.path for c in cells}) == len(cells)


def test_labels_sort_lexicographically_with_repeat_order():
    cells = build_matrix(["t"], ["m"], 12, Path("runs"), [50])
    labels = [c.label for c in cells]
    assert labels == sorted(labels)          # r01 < r02 < ... < r12


# --- budget accounting ------------------------------------------------------

def record(tmp_path, name, **over):
    base = dict(config_version="cfg", task_id="002-databases", model="m",
                outcome="solved", verdict_status="ok",
                before_failed_ids=["t::a"], max_iterations=50,
                max_cost_usd=1.0, cost_usd=0.42, cost_known=True)
    base.update(over)
    return RunRecord(**base).save(tmp_path / name)


def test_a_finished_run_is_charged_what_it_cost(tmp_path):
    assert actual_cost(record(tmp_path, "a.json")) == 0.42


def test_an_unknown_cost_is_charged_at_the_per_run_cap(tmp_path):
    """It is not free. Treating an unreported cost as zero lets a sweep
    with a broken usage report run until the budget is exhausted in the
    other direction — by the bill rather than by the counter."""
    path = record(tmp_path, "b.json", cost_known=False, cost_usd=0.0,
                  max_cost_usd=1.0)
    assert actual_cost(path) == 1.0


def test_an_unreadable_record_is_charged_nothing_and_does_not_raise(tmp_path):
    """A corrupt file must not abort a sweep midway; it is reported by the
    readers, which are strict, rather than here."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert actual_cost(bad) == 0.0


def test_a_failed_attempt_is_charged_what_it_actually_spent(tmp_path):
    """A rate-limited run is not a free run.

    The sweep released the reservation for every failure, on the reasoning
    that a failed attempt measured nothing. It measured nothing and it
    still cost money: the provider refused service at iteration 30, and
    the record on disk says what the first 30 cost. Refunding that makes
    the budget cap stop binding.
    """
    from scripts.sweep import chargeable
    path = record(tmp_path, "f.json", cost_usd=0.07, outcome="no_verdict")
    assert chargeable(path) == pytest.approx(0.07)


def test_the_retry_does_not_erase_what_the_first_attempt_spent(tmp_path):
    """Both attempts write to the same path; only one record survives."""
    from scripts.sweep import chargeable
    path = record(tmp_path, "f.json", cost_usd=0.05, outcome="no_verdict")
    assert chargeable(path, sunk=0.09) == pytest.approx(0.14)


def test_an_attempt_that_left_no_record_is_charged_nothing(tmp_path):
    """Never reached the model; holding its reservation would shrink the
    budget for work that did happen."""
    from scripts.sweep import chargeable
    assert chargeable(tmp_path / "absent.json") == 0.0


def test_resume_skips_what_already_exists(tmp_path):
    """Re-running an existing cell is a silent protocol violation: repeats
    are meant to be independent samples, and replacing one with a fresh
    draw biases the set toward whatever the machine was doing when the
    sweep was interrupted."""
    cells = build_matrix(["002-databases"], ["m"], 3, tmp_path, [50])
    cells[0].path.write_text(json.dumps({"placeholder": True}))
    pending = [c for c in cells if not c.path.exists()]
    assert len(pending) == 2


# --- resuming safely --------------------------------------------------------

def test_a_truncated_record_is_re_run_not_skipped(tmp_path):
    """Existence is not completion. A record that is truncated, or written
    by an older schema, is a file that exists and cannot be scored — and a
    sweep resuming on existence alone would skip it forever, leaving a hole
    in the matrix that no amount of re-running fills."""
    from scripts.sweep import is_complete
    good = record(tmp_path, "good.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 3, "task_id": "002-datab')  # cut off
    assert is_complete(good)
    assert bad.exists() and not is_complete(bad)


def test_an_old_schema_record_is_re_run(tmp_path):
    import json
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"schema_version": 2, "task_id": "t"}))
    from scripts.sweep import is_complete
    assert not is_complete(old)


# --- budgeting on evidence --------------------------------------------------

def test_the_estimate_falls_back_to_the_cap_with_no_evidence(tmp_path):
    from scripts.sweep import planned_cost
    assert planned_cost(1.00, tmp_path) == 1.00


def test_the_estimate_learns_from_completed_runs(tmp_path):
    """A measured ds-flash run costs about $0.05 against a $1.00 cap.
    Reserving the cap would let a $20 budget authorise twenty attempts when
    it can afford several hundred."""
    from scripts.sweep import planned_cost
    for i in range(4):
        record(tmp_path, f"r{i}.json", cost_usd=0.05)
    estimate = planned_cost(1.00, tmp_path)
    assert 0.05 < estimate < 0.25          # the mean, with a margin
    assert int(20 // estimate) > 100


def test_an_unknown_cost_is_estimated_at_the_cap(tmp_path):
    """Not free. A run whose provider reported no usage could have cost
    anything up to its cap."""
    from scripts.sweep import planned_cost
    for i in range(4):
        record(tmp_path, f"u{i}.json", cost_known=False, cost_usd=0.0,
               max_cost_usd=1.0)
    assert planned_cost(1.00, tmp_path) == 1.00


def test_failed_infrastructure_does_not_unlock_a_cheap_forecast(tmp_path):
    from scripts.sweep import planned_cost
    for i in range(2):
        record(tmp_path, f"r{i}.json", cost_usd=0.05)
    record(tmp_path, "failed.json", cost_usd=0.001, harness_error="HTTP 429")
    assert planned_cost(1.0, tmp_path) == 1.0


def test_manifest_records_pricing_and_output_allowance(tmp_path, monkeypatch):
    from agentcheck.ledger import Ledger
    from scripts import sweep

    monkeypatch.setattr(sweep.Task, "load", lambda _: SimpleNamespace(
        image_id=lambda: "sha256:baseline"))
    monkeypatch.setattr(sweep.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="recorded"))
    args = SimpleNamespace(repeats=1, max_cost=1.0, max_wall=1800.0,
                           max_tokens=8192, budget=5.0, jobs=1)
    cells = build_matrix(["t"], ["qwen3-coder-next"], 1, tmp_path, [40])
    manifest = sweep.sweep_manifest(
        args, cells, sweep.SweepState(), Ledger(tmp_path / "ledger.json", 5), 0,
    )
    assert manifest["caps"]["max_tokens"] == 8192
    assert manifest["git_dirty"] is True
    spec = manifest["model_specs"]["qwen3-coder-next"]
    assert spec["id"] == "qwen/qwen3-coder-next"
    assert spec["input_price"] > 0 and spec["output_price"] > 0
    assert "api_key" not in spec


def test_a_cheap_model_does_not_forecast_an_expensive_one(tmp_path):
    """The forecast must be priced for the model being RUN.

    Every run in this project was `ds-flash` until a second model was
    added. Averaging across all records quoted `or-haiku` — ~12x the
    input and ~28x the output price — at ds-flash's $0.05, forecasting
    $0.80 for roughly $8 of work. That is the house bug: a missing
    distinction becoming a plausible number, on the exact operation the
    second model exists for. With no history for the model asked about,
    the honest answer is the cap.
    """
    from scripts.sweep import planned_cost
    for i in range(6):
        record(tmp_path, f"cheap{i}.json",
               model="deepseek/deepseek-v4-flash-0731", cost_usd=0.05)

    # asked about the model that HAS history: still learns from it
    assert planned_cost(1.00, tmp_path, ["or-ds-flash"]) < 0.25

    # asked about one that does not: falls back to the cap, not to the
    # cheap model's mean
    assert planned_cost(1.00, tmp_path, ["or-haiku"]) == 1.00

    # and the old callers, passing no models, keep the pooled behaviour
    assert planned_cost(1.00, tmp_path) < 0.25


# --- surviving a flaky daemon -----------------------------------------------

def test_wait_for_docker_returns_immediately_when_healthy():
    import subprocess

    from scripts.sweep import wait_for_docker
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker is not running")
    assert wait_for_docker(timeout=5.0)


def test_wait_for_docker_gives_up_rather_than_hanging(monkeypatch):
    """Bounded. A sweep that blocks forever on a dead daemon is worse than
    one that stops and says so."""
    import subprocess

    from scripts import sweep

    class Dead:
        returncode = 1
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Dead())
    monkeypatch.setattr(sweep.time, "sleep", lambda _: None)
    assert not sweep.wait_for_docker(timeout=0.2)


def test_the_sweep_retries_a_failed_attempt_once():
    """Docker Desktop stopped twice during one afternoon here. A harness
    error produced no sample at all, so re-running the cell is the first
    draw rather than a second — retrying a SUCCESSFUL run would bias the
    set, retrying one that never happened does not."""
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "sweep.py").read_text()
    body = source[source.index("for cell in cells:"):]
    assert "retrying once" in body
    assert body.index("wait_for_docker()") < body.index("run_cell(")


# --- phase 2: the ladder, throttling, and orphan safety ---------------------

from scripts.sweep import ORPHAN_MARGIN_SECONDS, looks_rate_limited


def test_a_single_rung_keeps_the_historical_filename(tmp_path):
    """Resume compatibility is not cosmetic.

    Adding the iteration budget to every filename would make all fifteen
    existing records look pending, and the sweep would re-run the whole
    archive at full cost.
    """
    cells = build_matrix(["002-databases"], ["ds-flash"], 2, tmp_path, [50])
    assert [c.path.name for c in cells] == [
        "002-databases-ds-flash-r01.json",
        "002-databases-ds-flash-r02.json",
    ]


def test_a_ladder_gives_each_rung_its_own_cell(tmp_path):
    cells = build_matrix(["002-databases"], ["ds-flash"], 1, tmp_path,
                         [5, 10, 50])
    names = [c.path.name for c in cells]
    assert names == [
        "002-databases-ds-flash-i5-r01.json",
        "002-databases-ds-flash-i10-r01.json",
        "002-databases-ds-flash-i50-r01.json",
    ]
    assert [c.iterations for c in cells] == [5, 10, 50]


def test_ladder_rungs_are_deduplicated_and_ordered(tmp_path):
    cells = build_matrix(["t"], ["m"], 1, tmp_path, [50, 5, 50])
    assert [c.iterations for c in cells] == [5, 50]


def test_an_empty_ladder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no iteration budget"):
        build_matrix(["t"], ["m"], 1, tmp_path, [])


def test_each_cell_has_a_distinct_ledger_key(tmp_path):
    cells = build_matrix(["a", "b"], ["m1", "m2"], 3, tmp_path, [5, 50])
    keys = [c.key for c in cells]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("text", [
    "openai.RateLimitError: Rate limit reached",
    "HTTP 429 Too Many Requests",
    "Error: server overloaded, please retry",
    "You exceeded your current quota",
])
def test_a_provider_refusal_is_recognised(text):
    """A rate-limited attempt measured nothing about the model.

    Scoring it as a failure would put the provider's capacity on a given
    afternoon into the results, where it is indistinguishable from the
    model having given up.
    """
    assert looks_rate_limited(text)


@pytest.mark.parametrize("text", [
    "AssertionError: expected 200",
    "docker: no such image",
    "the agent gave up: cannot determine the installed version",
])
def test_an_ordinary_failure_is_not_mistaken_for_throttling(text):
    assert not looks_rate_limited(text)


def test_the_orphan_margin_exceeds_no_running_attempt():
    """The safety property behind concurrent sweeps.

    `sweep_orphans` cannot tell an orphan from a sibling, so the sweep
    only reaps containers older than a run's own wall cap. The margin has
    to be positive, or a container created exactly at the cap boundary
    could be killed while still finishing.
    """
    assert ORPHAN_MARGIN_SECONDS > 0


def test_the_label_carries_the_rung_so_the_filename_cannot_diverge(tmp_path):
    """The bug that killed the first sweep, pinned.

    `run_agent` builds its output path as `{task}-{model}-{label}.json`.
    The sweep folded the rung into its own path but not into the label, so
    it looked for `...-i5-r01.json` while run_agent wrote `...-r01.json`.
    Every attempt reported "no record written" with the record sitting on
    disk beside it, and the next cell then collided with that file.

    Two places computing one name — the exact failure this project exists
    to talk about, in its own sweep. `path` is now derived from `label`.
    """
    cells = build_matrix(["t"], ["m"], 1, tmp_path, [5, 50])
    for cell in cells:
        assert cell.path.name == f"t-m-{cell.label}.json", (
            "the sweep's path must be run_agent's formula applied to the "
            "same label it passes on the command line")
        assert f"i{cell.iterations}" in cell.label


def test_a_single_rung_label_stays_backward_compatible(tmp_path):
    """Adding the rung unconditionally would make every archived record
    look pending and re-run the whole archive at full cost."""
    cell = build_matrix(["t"], ["m"], 1, tmp_path, [50])[0]
    assert cell.label == "r01"
    assert cell.path.name == "t-m-r01.json"


def test_a_mixed_matrix_is_detected(tmp_path, good_record):
    """Resuming must not assemble one matrix from two experiments.

    `is_complete` asks whether a cell holds a measurement. It never asked
    whether that measurement came from the configuration being run NOW, and
    those are different questions — which is how a ladder became sixteen
    configurations of at most seven runs instead of eight rungs at n=8.
    Nothing crashed; the numbers were simply spread over twice as many
    columns as anyone wanted.
    """
    import json
    import os
    import time

    from scripts.sweep import superseded_configs

    cells = build_matrix(["002-databases"], ["m"], 2, tmp_path, [5])
    for i, cell in enumerate(cells):
        record = dict(good_record)
        record["max_iterations"] = cell.iterations
        # Two cells of one rung, written under different configurations.
        record["config_version"] = "old-config" if i == 0 else "new-config"
        cell.path.write_text(json.dumps(record))
        os.utime(cell.path, (time.time() + i, time.time() + i))

    stale = superseded_configs(cells)
    assert stale == {"old-config": 1}, \
        "the older configuration is the superseded one"


def test_a_homogeneous_matrix_reports_nothing(tmp_path, good_record):
    import json

    from scripts.sweep import superseded_configs

    cells = build_matrix(["002-databases"], ["m"], 3, tmp_path, [5])
    for cell in cells:
        record = dict(good_record)
        record["max_iterations"] = cell.iterations
        cell.path.write_text(json.dumps(record))
    assert superseded_configs(cells) == {}


def test_different_rungs_are_not_mistaken_for_supersession(tmp_path,
                                                           good_record):
    """Each iteration cap is its own configuration by design.

    Comparing configs globally rather than per (task, rung) would flag
    every rung but the last as superseded, and archive the whole ladder.
    """
    import json

    from scripts.sweep import superseded_configs

    cells = build_matrix(["002-databases"], ["m"], 1, tmp_path, [5, 50])
    for cell in cells:
        record = dict(good_record)
        record["max_iterations"] = cell.iterations
        record["config_version"] = f"cfg-for-{cell.iterations}"
        cell.path.write_text(json.dumps(record))
    assert superseded_configs(cells) == {}


def test_an_experimental_arm_gets_its_own_filenames(tmp_path):
    """Two configurations that can coexist need distinguishable names.

    Without the marker, a single-rung run of the budget-announcement arm
    writes to exactly the paths a single-rung control run would want, and
    the second silently claims the first's cells. `config_version` catches
    the mix afterwards — but only once the runs have been spent.
    """
    control = build_matrix(["t"], ["m"], 1, tmp_path, [40])
    variant = build_matrix(["t"], ["m"], 1, tmp_path, [40], arm="budget")
    assert control[0].path != variant[0].path
    assert variant[0].label == "budget-r01"


def test_the_control_arm_keeps_its_names(tmp_path):
    assert build_matrix(["t"], ["m"], 1, tmp_path, [40])[0].label == "r01"


def test_orphan_reaping_never_uses_a_zero_threshold():
    """A serial sweep used to reap every labelled container.

    The reasoning was that one job means one container at a time. It
    ignores every other process — the container test suite, a second
    sweep, an interactive run_agent — each of which would have its
    containers destroyed mid-command and record a harness failure it did
    not cause. There is no case where reaping a container younger than the
    wall cap is correct.
    """
    import inspect

    from scripts import sweep as sweep_module
    source = inspect.getsource(sweep_module.main)
    assert "orphan_age = args.max_wall + ORPHAN_MARGIN_SECONDS" in source
    assert "if args.jobs > 1" not in source.split("orphan_age")[1][:200]
