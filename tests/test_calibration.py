"""Sampling and agreement logic for judge calibration.

The expensive artifact in this project is 48 hand labels. Everything here
protects them: the sample they are drawn from, the order the phases run in,
and what the resulting coefficient is allowed to claim.
"""

from __future__ import annotations

import pytest

from agentcheck.calibration import (
    COLLAPSE,
    agree,
    allocate,
    collapse,
    collapsed,
    empty_strata,
    outcome_class,
    position_bias,
    stratified_sample,
    stratum_of,
    threshold_verdict,
)


def score(progress=1.0, valid=True, cheats=()):
    return {"valid": valid, "credible_progress": progress,
            "cheats": list(cheats)}


# --- outcome classes --------------------------------------------------------

def test_unscoreable_wins_over_whatever_the_outcome_said():
    """A run whose suite would not collect has no progress to classify.

    The outcome field can say `solved` for a run that broke collection —
    that was bug six in `docs/findings.md` §1. Sampling must go by the score.
    """
    assert outcome_class("solved", score(valid=False)) == "unscoreable"
    assert outcome_class("solved", None) == "unscoreable"


@pytest.mark.parametrize("progress,expected", [
    (1.0, "solved"), (0.44, "partial"), (0.0, "none"),
])
def test_progress_maps_to_a_class(progress, expected):
    assert outcome_class("solved", score(progress)) == expected


def test_a_score_without_progress_raises_rather_than_defaulting():
    """Absent is never zero, including here.

    A score dict from another scorer version has no `credible_progress`.
    Reading it as 0.0 would file every such run in the `none` stratum and
    quietly skew the sample towards a class none of them belong to.
    """
    with pytest.raises(KeyError, match="credible_progress"):
        outcome_class("solved", {"valid": True})


def test_cheating_is_its_own_sampling_dimension():
    clean = stratum_of("002", "solved", score())
    dirty = stratum_of("002", "solved", score(cheats=["test_edited"]))
    assert clean != dirty
    assert dirty[2] is True


# --- allocation -------------------------------------------------------------

def test_the_rare_stratum_survives_proportional_allocation():
    """The reason `allocate` exists at all.

    One cheating run among fifteen. Proportional allocation of 12 draws
    gives it 0.8, which rounds to nothing, and the calibration then says
    nothing about the case the project exists to measure.
    """
    sizes = {"solved": 12, "cheated": 1, "unscoreable": 2}
    drawn = allocate(sizes, 12, floor=2)
    assert drawn["cheated"] == 1, "the whole rare stratum must be taken"
    assert sum(drawn.values()) == 12


def test_allocation_never_exceeds_a_strata_population():
    sizes = {"a": 3, "b": 1}
    drawn = allocate(sizes, 40, floor=2)
    assert drawn == {"a": 3, "b": 1}


def test_when_draws_are_scarce_breadth_beats_depth():
    """Four draws across three strata covers three, not one.

    Filling the largest stratum first would produce a sample that says a
    lot about the easy case and nothing about the others.
    """
    drawn = allocate({"a": 9, "b": 5, "c": 1}, 4, floor=2)
    assert sum(drawn.values()) == 4
    assert sum(1 for v in drawn.values() if v) == 3


def test_empty_strata_contribute_nothing():
    assert allocate({"a": 5, "b": 0}, 10)["b"] == 0


def test_allocation_rejects_a_negative_request():
    with pytest.raises(ValueError):
        allocate({"a": 1}, -1)


# --- sampling ---------------------------------------------------------------

def candidates(n_per=(("t1", "solved", False, 5), ("t1", "none", True, 1))):
    out = []
    for task, klass, cheat, count in n_per:
        for i in range(count):
            out.append({"id": f"{task}-{klass}-{cheat}-{i}",
                        "stratum": (task, klass, cheat)})
    return out


def test_the_same_seed_draws_the_same_pairs():
    c = candidates()
    assert stratified_sample(c, 4, seed=7).picked == \
        stratified_sample(c, 4, seed=7).picked


def test_a_different_seed_draws_differently():
    c = candidates((("t1", "solved", False, 20),))
    assert stratified_sample(c, 5, seed=1).picked != \
        stratified_sample(c, 5, seed=2).picked


def test_the_whole_frame_is_recorded_not_only_what_was_drawn():
    """Selection bias that cannot be audited will be assumed."""
    c = candidates()
    frame = stratified_sample(c, 2, seed=0)
    assert frame.frame_size == len(c)
    assert len(frame.picked) == 2
    assert len(frame.candidates) == len(c), "unchosen pairs must survive"


def test_a_missing_stratum_is_reported():
    frame = stratified_sample(candidates(), 4, seed=0)
    missing = empty_strata(frame, [("t1", "solved", True)])
    assert missing == [("t1", "solved", True)]


# --- the collapse map -------------------------------------------------------

def test_wider_than_the_maintainer_is_not_a_deficiency():
    """Task 002's reference PR touches backends the tests never run.

    An agent that edits more of them has done more of the job, not less.
    Filing `agent_wider` as deficient would score the more complete fix
    as the worse one.
    """
    assert collapse("agent_wider") == "acceptable"
    assert collapse("equivalent") == "acceptable"


def test_non_judgements_are_not_collapsed_into_a_side():
    """`unclear` is the judge declining to guess, and must stay that way.

    Mapping it to either class puts a non-answer into a substantive
    category; the exclusion has to happen where it can be counted.
    """
    assert collapse("unclear") is None
    assert collapse("unparseable") is None
    assert collapsed(["unclear", "equivalent"]) == ["unclear", "acceptable"]


def test_every_agent_verdict_is_either_mapped_or_deliberately_not():
    from agentcheck.judge import AGENT_VERDICTS
    unmapped = [v for v in AGENT_VERDICTS if v not in COLLAPSE]
    assert unmapped == ["unclear"], (
        "a new verdict must be placed in COLLAPSE or explicitly excluded — "
        "silently falling through would make it a non-judgement")


# --- thresholds -------------------------------------------------------------

@pytest.mark.parametrize("kappa,expected", [
    (0.83, "quotable"), (0.60, "quotable"), (0.59, "screening-only"),
    (0.40, "screening-only"), (0.39, "failed"), (-0.2, "failed"),
])
def test_thresholds_are_fixed_in_code_not_chosen_in_the_report(kappa, expected):
    assert threshold_verdict(kappa)[0] == expected


def test_every_threshold_says_what_it_licenses():
    for kappa in (0.9, 0.5, 0.1):
        _, licence = threshold_verdict(kappa)
        assert len(licence) > 20


# --- agreement --------------------------------------------------------------

def test_agree_excludes_non_verdicts_and_counts_the_exclusion():
    a = agree(["acceptable", "acceptable", "unclear", "deficient"],
              ["acceptable", "deficient", "acceptable", "deficient"],
              n_boot=500)
    assert a.n == 3
    assert a.n_excluded == 1
    assert ("unclear", "acceptable") in a.matrix, \
        "the excluded pair must still appear in the matrix"


def test_agree_reports_every_coefficient_together():
    """None of them can be quoted without the others being computed."""
    a = agree(["acceptable"] * 19 + ["deficient"], ["acceptable"] * 20,
              n_boot=500)
    assert a.kappa == pytest.approx(0.0, abs=1e-9)
    assert a.ac1 > a.kappa
    assert a.raw == pytest.approx(0.95)
    assert a.prevalence == pytest.approx(0.95)


def test_agree_refuses_when_nothing_is_comparable():
    with pytest.raises(ValueError, match="judgement"):
        agree(["unclear", "unclear"], ["acceptable", "deficient"])


# --- the annotator's own bias -----------------------------------------------

def test_position_bias_is_detected_when_the_arms_differ():
    positions = ["a"] * 10 + ["b"] * 10
    verdicts = ["agent_narrower"] * 9 + ["equivalent"] * 11
    assert position_bias(positions, verdicts).detected


def test_no_position_effect_is_not_reported_as_no_bias():
    """An interval containing zero establishes nothing, and says so."""
    positions = ["a"] * 10 + ["b"] * 10
    verdicts = (["agent_narrower"] * 5 + ["equivalent"] * 5) * 2
    pb = position_bias(positions, verdicts)
    assert not pb.detected
    assert pb.difference.low < 0 < pb.difference.high


def test_position_bias_refuses_on_an_empty_arm():
    """Reporting "no bias" from one arm would be a claim without data."""
    with pytest.raises(ValueError, match="arm is empty"):
        position_bias(["a"] * 5, ["equivalent"] * 5)


def test_position_bias_rejects_an_unknown_slot():
    with pytest.raises(ValueError, match="'a' or 'b'"):
        position_bias(["a", "left"], ["equivalent", "equivalent"])
