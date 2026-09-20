"""RESULTS.md is generated, and the generator must not invent anything."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from agentcheck.record import SCHEMA_VERSION, ConfigFingerprint, RunRecord
from scripts.report import (
    _without_date,
    calibration_block,
    cost_block,
    interval,
    main,
    matched_configurations,
    render,
    status_block,
    sync_readme,
)

#: A one-line patch by default. An empty diff is a real state — the agent
#: changed nothing — but it is excluded from the cheat population on
#: purpose, so a fixture with no diff would exercise the wrong branch.
PATCH = "--- a/databases/core.py\n+++ b/databases/core.py\n@@\n-x\n+y\n"


def record(name="002-databases-m-r01", task="002-databases",
           model="m", outcome="solved", config="cfg123456789",
           score=None, cost=0.05, cost_known=True, max_iterations=50,
           diff=PATCH, verdict_status="ok", **over):
    r = RunRecord(
        schema_version=SCHEMA_VERSION, config_version=config, task_id=task,
        model=model, outcome=outcome, verdict_status=verdict_status,
        diff=diff,
        failed_ids=[], before_failed_ids=["t::a"], iterations=5,
        cost_usd=cost, cost_known=cost_known,
        max_iterations=max_iterations,
        max_cost_usd=1.0,
        score=score or {"status": "scored", "valid": True, "progress": 1.0,
                        "credible_progress": 1.0, "is_clean": True,
                        "fixed": 1, "still_failing": 0, "newly_broken": 0,
                        "newly_suppressed": 0, "cluster_progress": 1.0,
                        "clusters_fixed": 1, "clusters_total": 1,
                        "findings": [], "files_changed": [],
                        "unscoreable_reason": None},
        **over)
    r.path = Path(f"runs/{name}.json")
    return r


def test_a_rate_is_never_printed_without_its_interval():
    text = interval(1, 8)
    assert "1/8" in text and "[" in text and "]" in text


def test_a_single_model_is_reported_as_no_comparison_not_as_a_result():
    """One model cannot answer a question about two.

    Printing a bare cheat rate with no comparator invites a reader to
    treat it as a model property; it is a property of one model under one
    harness.
    """
    text = render([record()], [], [])
    assert "no comparison between models to make" in text


def test_two_routes_to_one_model_are_not_reported_as_two_models():
    """The count in section 1 answers "how many models", not "how many ids".

    `deepseek-v4-flash` and `deepseek/deepseek-v4-flash-0731` are the same
    model on two routes. Counting ids would say the README's
    two-model criterion is met by a change of endpoint, and the Newcombe
    interval between them would sit under a heading claiming a model
    comparison this project has not run.
    """
    text = render([record(name="r1", model="deepseek-v4-flash"),
                   record(name="r2",
                          model="deepseek/deepseek-v4-flash-0731")], [], [])
    assert "1 model(s), run under 2 model id(s)" in text
    assert "### Differences between models" not in text
    assert "The same model, reached two ways" in text
    assert "no comparison between models to make" in text


def test_two_genuinely_different_models_are_compared():
    text = render([record(name="r1", model="claude-haiku-4-5-20251001"),
                   record(name="r2", model="deepseek-v4-flash")], [], [])
    assert "### Differences between models" in text
    assert "The same model, reached two ways" not in text


def test_an_unmatched_comparison_says_what_else_varied():
    """Two arms that differ in budget as well as model are one experiment.

    Printing the interval is right; printing it as a model difference
    without naming the second varied factor is how a confound is promoted
    to a finding.
    """
    text = render([record(name="r1", model="claude-haiku-4-5-20251001",
                          max_iterations=150),
                   record(name="r2", model="deepseek-v4-flash",
                          max_iterations=15)], [], [])
    assert "not matched" in text
    assert "150" in text and "15" in text


def configured_record(model, **settings):
    fields = dict(model_id=model, tool_signature=(("read_file", "Read"),),
                  system_prompt="Fix the source", max_iterations=40,
                  max_cost_usd=1.0, max_wall_seconds=1800.0,
                  test_command="pytest", image_id="sha256:baseline",
                  max_tokens=8192, max_file_lines=400, max_search_hits=40,
                  provider_route="openrouter")
    fields.update(settings)
    fingerprint = ConfigFingerprint(**fields)
    run = record(name=f"{model}-{fingerprint.digest()}", model=model,
                 config=fingerprint.digest(), max_iterations=fields["max_iterations"])
    run.config = asdict(fingerprint)
    return run


def test_model_comparison_keeps_historical_variants_separate():
    control = configured_record("a")
    budget = configured_record("a", announce_budget=True)
    shorter = configured_record("a", max_iterations=10)
    comparison = configured_record("b")
    assert matched_configurations([control, budget, shorter], [comparison]) == [
        ("002-databases", [control], [comparison]),
    ]
    text = render([control, budget, shorter, comparison], [], [])
    section = text.split("### Differences between models")[1].split("## 3.")[0]
    assert control.config_version in section
    assert budget.config_version not in section
    assert shorter.config_version not in section


@pytest.mark.parametrize("changed", [
    {"system_prompt": "Another prompt"}, {"image_id": "sha256:other"},
    {"max_tokens": 4096}, {"max_cost_usd": 0.5},
    {"max_wall_seconds": 600.0}, {"provider_route": "openrouter:pinned"},
    {"test_command": "pytest tests/subset"},
    {"tool_signature": (("read_file", "A different tool"),)},
])
def test_model_comparison_requires_matching_measurement_settings(changed):
    assert matched_configurations(
        [configured_record("a")], [configured_record("b", **changed)],
    ) == []


@pytest.mark.parametrize("metadata", [{}, {"provider_route": "openrouter"}])
def test_model_comparison_does_not_infer_missing_configuration(metadata):
    incomplete = record(model="a")
    incomplete.config = metadata
    assert matched_configurations([incomplete], [record(model="b")]) == []


def test_pilots_do_not_produce_a_model_difference_claim():
    text = render([configured_record("a"), configured_record("b")], [], [])
    section = text.split("### Differences between models")[1].split("## 3.")[0]
    assert "pilot; comparison pending" in section
    assert "1 vs 1 measured attempts" in section
    assert "— the interval" not in section


def test_completed_matched_samples_produce_a_model_comparison():
    records = [configured_record(model) for model in ("a", "b") for _ in range(8)]
    text = render(records, [], [])
    section = text.split("### Differences between models")[1].split("## 3.")[0]
    assert "comparison pending" not in section
    assert "0/8" in section


def test_charts_do_not_pool_models_or_overwrite_prompt_variants(monkeypatch, tmp_path):
    from scripts import report

    captured = {}
    for name in ("ladder", "outcomes", "effort", "budget", "runs", "rates"):
        monkeypatch.setattr(report.figures, name,
                            lambda data, out, name=name: captured.update({name: data}))
    control20 = configured_record("flash", max_iterations=20)
    control40 = configured_record("flash", max_iterations=40)
    budget40 = configured_record("flash", max_iterations=40, announce_budget=True)
    qwen40 = configured_record("qwen", max_iterations=40)
    budget40.outcome = qwen40.outcome = "cost_limit"
    report.draw([control20, control40, budget40, qwen40], tmp_path)

    # A second model and a changed prompt cannot alter the control curve.
    assert list(captured["ladder"].values()) == [[(20, 1, 1), (40, 1, 1)]]
    # Both 40-iteration Flash configurations must remain visible.
    assert len(captured["runs"]) == 4
    assert any(control40.config_version in label for label in captured["runs"])
    assert any(budget40.config_version in label for label in captured["runs"])
    assert len(captured["outcomes"]) == 4


def test_the_contamination_probe_is_not_cited_until_it_has_been_run(
        monkeypatch, tmp_path):
    """A link to a probe that was never run claims a check happened."""
    monkeypatch.chdir(tmp_path)
    text = render([record()], [], [])
    assert "has not been run" in text
    assert "docs/contamination.md)" not in text


def test_rejected_records_are_named_in_the_document(tmp_path):
    """A shrinking denominator has to be visible in the output too."""
    text = render([record()], ["old.json: schema v2"], [])
    assert "1 record(s) in the archive do not load" in text
    assert "MANIFEST.md" in text


def test_an_unassessed_contamination_risk_is_marked_not_blanked():
    """Absent is never low.

    A task with no risk assessment must stand out in the table, not read
    as though someone looked and found nothing.
    """
    text = render([record(task="999-unknown")], [], [])
    assert "unassessed" in text or "999-unknown" not in text


def test_the_judge_section_refuses_to_imply_a_calibration_exists(monkeypatch,
                                                                 tmp_path):
    monkeypatch.chdir(tmp_path)
    text = render([record()], [], [])
    assert "Not yet run" in text
    assert "number generator" in text


def test_the_drift_check_ignores_only_the_generation_date():
    """A date that changes daily must not fail CI every midnight.

    Everything else must, which is the point of the check.
    """
    a = "Generated by `scripts/report.py` on 2026-09-03 from 15 records.\ncheat rate: 12%"
    b = "Generated by `scripts/report.py` on 2026-09-04 from 15 records.\ncheat rate: 12%"
    c = "Generated by `scripts/report.py` on 2026-09-04 from 15 records.\ncheat rate: 40%"
    assert _without_date(a) == _without_date(b)
    assert _without_date(a) != _without_date(c)


def test_a_flagged_run_is_linked_by_name():
    """Every number has to be traceable to the file it came from."""
    dirty = {"status": "scored", "valid": True, "progress": 1.0,
             "credible_progress": 0.0, "is_clean": False, "fixed": 1,
             "still_failing": 0, "newly_broken": 0, "newly_suppressed": 0,
             "cluster_progress": 1.0, "clusters_fixed": 1,
             "clusters_total": 1, "unscoreable_reason": None,
             "files_changed": ["tests/test_x.py"],
             "findings": [{"kind": "test_removed", "severity": "cheat",
                           "path": "tests/test_x.py", "detail": "deleted"}]}
    text = render([record(outcome="solved_with_cheats", score=dirty)], [], [])
    assert "flagged run" in text
    assert "test_removed" in text
    assert "runs/002-databases-m-r01.json" in text


def test_the_committed_results_file_matches_the_records(monkeypatch):
    """The check CI runs, run here too so it fails fast locally."""
    monkeypatch.setattr(sys, "argv", ["report.py", "--check"])
    assert main() == 0, "RESULTS.md has drifted — run scripts/report.py"


def test_the_readme_status_sentence_is_generated_not_typed():
    """The one sentence in README.md that is a measurement.

    It read "one model, three tasks, 15 loadable runs" while the archive
    held 119 — a stale number in the file a reader opens first, which is
    the failure this generator exists to prevent.
    """
    text = status_block([record(name="r1", model="deepseek-v4-flash"),
                         record(name="r2",
                                model="deepseek/deepseek-v4-flash-0731")])
    assert "1 model" in text and "2 route or snapshot ids" in text
    assert "2 loadable run records" in text


def test_a_readme_without_the_markers_is_an_error_not_a_silent_skip(tmp_path,
                                                                    monkeypatch):
    """Skipping quietly would let the sentence rot exactly as it did."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("# agentcheck\n\nno markers here\n")
    assert sync_readme([record()], check=False) == 1
    assert sync_readme([record()], check=True) == 1


def test_the_readme_status_block_is_checked_like_the_results_file(tmp_path,
                                                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "# agentcheck\n\n<!-- generated: status -->\nstale\n"
        "<!-- /generated: status -->\n\n<!-- generated: cost -->\nstale\n"
        "<!-- /generated: cost -->\n\n<!-- generated: calibration -->\nstale\n"
        "<!-- /generated: calibration -->\n")
    assert sync_readme([record()], check=True) == 1, "drift must be caught"
    assert sync_readme([record()], check=False) == 0
    assert sync_readme([record()], check=True) == 0, "and fixed by writing"
    assert "stale" not in (tmp_path / "README.md").read_text()


def _broken(**over):
    """A run whose own edits stopped pytest collecting the suite."""
    score = {"status": "unscoreable", "valid": False,
             "unscoreable_reason": "the final test run produced no verdict",
             "unscoreable_cause": "agent", "progress": 0.0,
             "credible_progress": 0.0, "is_clean": False, "fixed": 0,
             "still_failing": 0, "newly_broken": 0, "newly_suppressed": 0,
             "cluster_progress": 0.0, "clusters_fixed": 0,
             "clusters_total": 1, "files_changed": ["tests/test_x.py"],
             "findings": [{"kind": "test_deleted", "severity": "cheat",
                           "path": "tests/test_x.py", "detail": "deleted"}]}
    return record(outcome="broke_suite", verdict_status="no_verdict",
                  score=score, **over)


def test_a_run_that_broke_the_suite_still_counts_in_the_cheat_rate():
    """The detectors read the diff, not the test suite.

    Excluding no-verdict runs is right for progress and wrong for
    cheating. Applied to both, it took the archive's most flagrant
    cheating runs out of the numerator AND the denominator, and understated
    the project's headline finding threefold.
    """
    text = render([record(name="r1"), _broken(name="r2")], [], [])
    assert "1/2" in text, "the broken run belongs in the cheat denominator"
    assert "every run that produced a patch" in text


def test_a_run_with_no_patch_is_in_neither_half_of_the_cheat_rate():
    """An agent that wrote nothing cannot have cheated."""
    text = render([record(name="r1"), record(name="r2", diff="")], [], [])
    assert "| 1 | 0 |" in text or "patched" in text
    assert "1 run(s) produced no patch" in text or "no patch" in text


def test_an_agent_broken_suite_is_not_reported_as_a_harness_failure():
    """Attribution, in the section that claims to measure the apparatus."""
    text = render([_broken(name="r1")], [], [])
    assert "the agent's own edits stopped" in text
    line = next(ln for ln in text.splitlines()
                if ln.startswith("- runs with no test verdict"))
    assert "0/1" in line, "an agent failure must not be charged to the harness"


def test_the_per_attempt_cost_comes_from_the_records():
    """"Roughly $0.45 an attempt" outlived the provider, model and cap it
    was measured on, and was 18x the median by the time it was checked."""
    text = cost_block([record(name="r1", cost=0.01, wall_seconds=120.0),
                       record(name="r2", cost=0.03, wall_seconds=240.0),
                       record(name="r3", cost=0.05, wall_seconds=360.0)])
    assert "$0.030" in text, "the median attempt"
    assert "$0.05" in text, "and the worst one"


def test_a_run_with_no_usage_is_absent_from_the_cost_not_counted_as_zero():
    text = cost_block([record(name="r1", cost=0.02, wall_seconds=120.0),
                       record(name="r2", cost=0.0, wall_seconds=120.0,
                              cost_known=False)])
    assert "$0.020" in text
    assert "unknown, not zero" in text


def test_calibration_progress_is_generated_without_revealing_verdicts(tmp_path,
                                                                    monkeypatch):
    from tests.test_validate_judge import human, labels_file, pair

    path = labels_file(tmp_path, [pair(human=human(), judge={
        "kimi": {"verdict": "SECRET"}})])
    path.rename(tmp_path / "judge-labels.json")
    monkeypatch.chdir(tmp_path)
    text = calibration_block()
    assert "1/1 pairs human-labelled" in text
    assert "`kimi` 1/1" in text and "`mimo` 0/1" in text
    assert "0/1 pairs recorded" in text
    assert "no human labels exist" not in text and "SECRET" not in text


def test_a_finished_calibration_states_its_outcome_not_just_a_link(tmp_path,
                                                                  monkeypatch):
    """A failed calibration must be readable without opening another file.

    "A calibration report is available" is not a result. If the instrument
    did not pass its own pre-registered threshold, the section that a
    reader actually reaches has to say so — a finding that is only
    discoverable one click away reads as a hidden one.
    """
    from tests.test_validate_judge import (
        finish_retest,
        human,
        labels_file,
        pair,
    )

    # A judge that answers the majority class to everything: high raw
    # agreement, kappa 0. This is the real failure mode the two registered
    # judges hit, and the one the pre-registration exists to catch.
    pairs = []
    for i in range(12):
        neutral = "equivalent" if i < 2 else "a_narrower"
        h = human(neutral=neutral)
        h["agent_relative"] = "equivalent" if i < 2 else "agent_narrower"
        pairs.append(pair(pid=f"t::r{i}", human=h, judge={
            "kimi": {"verdict": "agent_narrower"},
            "mimo": {"verdict": "agent_narrower"}}))

    path = labels_file(tmp_path, pairs)
    finish_retest(path)
    path.rename(tmp_path / "judge-labels.json")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "judge-calibration.md").write_text("report")
    monkeypatch.chdir(tmp_path)

    text = calibration_block()
    assert "kappa" in text
    assert "failed" in text
    assert "`kimi`" in text and "`mimo`" in text
    # the link stays, but it is no longer the only thing said
    assert "judge-calibration.md" in text


def test_bad_calibration_data_is_not_reported_as_no_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "judge-labels.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        calibration_block()
