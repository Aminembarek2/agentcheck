"""Figures are generated from records, so they cannot drift — but they can
still lie about what they were given. These are the ways they must not."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcheck import figures

pytest.importorskip("matplotlib",
                    reason="matplotlib is a dev extra; the harness and every "
                           "other test must work without it")


def test_a_fourth_series_is_refused_rather_than_given_a_new_hue():
    """Past three, the validated palette stops clearing the all-pairs gate.

    A generated fourth hue is indistinguishable from an existing one under
    CVD, so the honest options are faceting or folding the tail into
    'other' — never inventing a colour and hoping.
    """
    four = {str(i): [(5, 1, 4), (50, 3, 4)] for i in range(4)}
    with pytest.raises(ValueError, match="ceiling"):
        figures.ladder(four, Path("/tmp"))


def test_both_themes_are_written_and_are_not_the_same_image(tmp_path):
    """Dark mode is a separate render from the same ramps, not a flip."""
    written = figures.rates([("a", 1, 8), ("b", 0, 5)], tmp_path)
    assert {p.name for p in written} == {"rates-light.svg", "rates-dark.svg"}
    # sorted() puts "dark" before "light" alphabetically; unpacking them
    # the other way round is how this test failed the first time.
    dark, light = sorted(written)
    assert dark.read_text() != light.read_text()
    assert "1a1a19" in dark.read_text()
    assert "fcfcfb" in light.read_text()


def test_a_rate_is_never_drawn_without_its_interval(tmp_path):
    """1/8 and 100/800 are the same height on a bar chart and are not the
    same claim. The interval bounds must reach the output."""
    figures.rates([("only", 1, 8)], tmp_path)
    svg = (tmp_path / "rates-light.svg").read_text()
    assert "1/8" in svg
    assert "47%" in svg, "the Wilson upper bound must be visible"


def test_every_series_is_directly_labelled(tmp_path):
    """The relief rule. On the light surface the aqua slot sits at 2.74:1,
    below the 3:1 bar, so identity may not rest on colour alone."""
    figures.ladder({"task-a": [(5, 0, 8), (50, 8, 8)]}, tmp_path)
    assert "task-a" in (tmp_path / "ladder-light.svg").read_text()


def test_individual_runs_survive_into_the_plot(tmp_path):
    """The figure's whole purpose is to not be a mean."""
    figures.runs({"cfg": [0.0, 0.0, 0.44, 0.93, 1.0]}, tmp_path)
    svg = (tmp_path / "runs-light.svg").read_text()
    assert "n=5" in svg
    assert "median 44%" in svg


def test_the_picture_block_offers_both_themes():
    block = figures.picture("ladder", "a curve")
    assert "prefers-color-scheme: dark" in block
    assert "ladder-dark.svg" in block and "ladder-light.svg" in block
    assert 'alt="a curve"' in block


# --- outcome composition ----------------------------------------------------

def test_every_recorded_outcome_has_a_reportable_class():
    """A new outcome must not be swept into an existing bucket.

    `record.OUTCOMES` is the vocabulary; if someone adds one and forgets
    this table, the figure would quietly file it under something else and
    hide a behaviour that was added on purpose.
    """
    from agentcheck.record import OUTCOMES
    for outcome in OUTCOMES:
        assert figures.classify(outcome)


def test_an_unknown_outcome_raises_rather_than_bucketing():
    with pytest.raises(KeyError, match="no reportable class"):
        figures.classify("mysteriously_fine")


def test_cheating_is_never_folded_into_solved():
    """The headline finding of the project, protected as an assertion.

    Merging `solved_with_cheats` into `solved` is the exact mistake the
    whole harness exists to name, and it would be one dictionary edit away.
    """
    assert figures.classify("solved") != figures.classify("solved_with_cheats")


def test_stopping_early_is_one_class_however_it_was_expressed():
    """`gave_up` and `stopped_without_solving` are the same decision.

    One said so through the give_up tool, the other simply returned no
    tool calls; both stopped with budget remaining. They are kept apart in
    the record so this figure can choose to join them — joining them at
    record time would have destroyed the distinction permanently.
    """
    assert (figures.classify("gave_up")
            == figures.classify("stopped_without_solving"))


def test_a_measurement_that_did_not_happen_is_its_own_class():
    """Neither a success nor a failure. Counting it as either invents data."""
    no_verdict = figures.classify("no_verdict")
    assert no_verdict == figures.classify("harness_error")
    assert no_verdict != figures.classify("solved")
    assert no_verdict != figures.classify("iteration_limit")


def test_the_stack_reports_counts_beside_proportions(tmp_path):
    """A proportion without its denominator overstates a result."""
    figures.outcomes([("5it", ["solved", "gave_up", "gave_up"])], tmp_path)
    svg = (tmp_path / "outcomes-light.svg").read_text()
    assert "n=3" in svg
    assert "stopped short" in svg, "the legend must name the class"


def test_the_stack_refuses_more_classes_than_the_palette_validates():
    assert len(figures.OUTCOME_CLASSES) <= figures.MAX_STACKED


# --- was the budget binding -------------------------------------------------

def test_the_budget_plot_marks_the_cap_itself(tmp_path):
    """Without the reference line a reader cannot tell a run that was
    stopped by the budget from one that stopped on its own."""
    figures.budget({"t": [(150, 45), (150, 91)]}, tmp_path)
    assert "used the whole budget" in \
        (tmp_path / "budget-light.svg").read_text()


def test_the_budget_plot_holds_to_the_all_pairs_ceiling():
    """Every dot can land beside every other, so the adjacent-only slots
    do not apply here — three is the ceiling that validates."""
    four = {str(i): [(50, 10)] for i in range(4)}
    with pytest.raises(ValueError, match="all-pairs"):
        figures.budget(four, Path("/tmp"))


def test_the_same_data_draws_the_same_bytes(tmp_path):
    """Matplotlib stamps the save time into every SVG and salts its clip
    ids at random, so each report run rewrote ~1,200 lines of figures with
    no number changed. A figure's diff should mean its data moved."""
    first, second = tmp_path / "one", tmp_path / "two"
    series = {"t": [(5, 1, 8), (50, 6, 8)]}
    figures.ladder(series, first)
    figures.ladder(series, second)
    for svg in first.glob("*.svg"):
        assert svg.read_bytes() == (second / svg.name).read_bytes(), svg.name
        assert b"dc:date" not in svg.read_bytes()
