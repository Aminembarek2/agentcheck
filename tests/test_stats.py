"""Tests for the small-sample statistics.

The interval is the finding at n=4, not the point estimate.

    pytest test_stats.py -v
"""

import pytest

from agentcheck.stats import (
    Interval,
    bootstrap_mean_ci,
    claim,
    cohens_kappa,
    describe,
    kappa_band,
    wilson,
    wilson_interval,
)


def test_wilson_does_not_claim_certainty_at_the_boundary():
    """The reason Wald is not used. At 0 successes out of 4 it reports
    [0, 0] — an interval asserting the true rate is exactly zero on the
    strength of four runs."""
    ci = wilson(0, 4)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high > 0.4


def test_wilson_is_wide_at_the_sample_sizes_in_use():
    """2/4 is the observed cheat rate for one configuration. Anything that
    reads as a precise 50% is misreporting it."""
    ci = wilson(2, 4)
    assert ci.low < 0.2 and ci.high > 0.8
    assert ci.width > 0.6


def test_a_perfect_score_still_has_a_lower_bound_below_one():
    ci = wilson(4, 4)
    assert ci.high == 1.0
    assert ci.low < 0.6


def test_more_data_narrows_the_interval():
    assert wilson(20, 40).width < wilson(2, 4).width


def test_interval_shrinks_toward_the_point_estimate():
    ci = wilson(500, 1000)
    assert abs(ci.point - 0.5) < 1e-9
    assert ci.width < 0.07


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_higher_confidence_is_wider(level):
    assert wilson(2, 4, level).width >= wilson(2, 4, 0.90).width


def test_impossible_inputs_raise_rather_than_return_a_number():
    with pytest.raises(ValueError):
        wilson(0, 0)
    with pytest.raises(ValueError):
        wilson(5, 4)
    with pytest.raises(ValueError):
        wilson(1, 4, level=0.5)


def test_claim_reads_as_a_sentence():
    assert claim(2, 4, "cheat rate").startswith("cheat rate: 2/4 = 50% [")


# --- distributions ----------------------------------------------------------

def test_describe_leads_with_the_median_not_the_mean():
    """The stored values are bimodal — clusters at 0%, ~56%, ~93%, 100% —
    and the mean of a bimodal distribution names a value no run took."""
    out = describe([0.0, 0.0, 0.93, 1.0])
    assert out.index("median") < out.index("mean")


def test_describe_prints_every_value_at_small_n():
    out = describe([0.0, 0.564, 0.927, 1.0])
    assert "all:" in out
    assert "0%, 56%, 93%, 100%" in out


def test_describe_warns_below_ten_runs():
    assert "too few runs" in describe([0.1] * 9)
    assert "too few runs" not in describe([0.1] * 10)


def test_describe_handles_no_runs():
    assert describe([]) == "(no runs)"


def test_interval_formats_with_its_sample_size():
    assert str(Interval(0.5, 0.15, 0.85, 4)) == "50% [15%, 85%] (n=4)"


# --- wilson_interval (tuple form) -------------------------------------------

def test_wilson_interval_matches_the_dataclass_form():
    """One implementation, two shapes. Two implementations would be two
    chances to compute the same interval differently."""
    ci = wilson(2, 4, level=0.95)
    assert wilson_interval(2, 4, z=1.959963984540054) == (ci.low, ci.high)


def test_wilson_4_of_4_does_not_return_one_one():
    """The known value from the spec. A normal-approximation interval
    reports (1.0, 1.0) here, asserting certainty from four runs."""
    low, high = wilson_interval(4, 4)
    assert high == 1.0
    assert 0.4 < low < 0.7


def test_wilson_interval_never_leaves_zero_one():
    for successes in range(6):
        low, high = wilson_interval(successes, 5)
        assert 0.0 <= low <= high <= 1.0


# --- bootstrap --------------------------------------------------------------

def test_bootstrap_brackets_the_sample_mean():
    values = [0.0, 0.44, 0.56, 0.93, 1.0]
    low, high = bootstrap_mean_ci(values, n_boot=2000)
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_is_reproducible():
    """A benchmark that reports a different interval each run has added
    noise to its own results, and a reader cannot tell that from a real
    change."""
    values = [0.0, 0.0, 0.564, 0.927]
    assert bootstrap_mean_ci(values, n_boot=1000) == \
        bootstrap_mean_ci(values, n_boot=1000)


def test_bootstrap_of_a_single_value_is_that_value():
    assert bootstrap_mean_ci([0.42]) == (0.42, 0.42)


def test_bootstrap_cannot_exceed_the_observed_range():
    """At n=4 the resamples only ever contain the four observed values, so
    the interval describes those four numbers rather than a population."""
    values = [0.0, 0.0, 0.564, 0.927]
    low, high = bootstrap_mean_ci(values, n_boot=2000)
    assert min(values) <= low and high <= max(values)


def test_more_data_narrows_the_bootstrap():
    tight = bootstrap_mean_ci([0.5] * 40 + [0.6] * 40, n_boot=2000)
    loose = bootstrap_mean_ci([0.5, 0.6], n_boot=2000)
    assert (tight[1] - tight[0]) < (loose[1] - loose[0])


def test_bootstrap_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0], alpha=1.5)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0], n_boot=0)


# --- kappa ------------------------------------------------------------------

def test_perfect_agreement_is_one():
    a = ["equivalent", "agent_wrong", "unclear", "agent_narrower"]
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_a_lazy_judge_scores_zero_despite_high_raw_agreement():
    """The reason kappa is used at all. 80% of the validation set is
    "equivalent"; a judge that answers "equivalent" every time gets 80%
    raw agreement while knowing nothing."""
    gold = ["equivalent"] * 8 + ["agent_wrong", "agent_narrower"]
    lazy = ["equivalent"] * 10
    raw = sum(g == j for g, j in zip(gold, lazy, strict=True)) / len(gold)
    assert raw == 0.8
    assert cohens_kappa(gold, lazy) == pytest.approx(0.0)


def test_chance_level_agreement_is_near_zero():
    gold = ["a", "b"] * 20
    guess = ["a", "a", "b", "b"] * 10
    assert abs(cohens_kappa(gold, guess)) < 0.15


def test_systematic_disagreement_is_negative():
    assert cohens_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) < 0


def test_total_agreement_on_one_label_carries_no_information():
    """0/0. Perfect agreement, zero information — reported as 0.0 rather
    than as 1.0, which would advertise a judge that only knows one word."""
    assert cohens_kappa(["equivalent"] * 5, ["equivalent"] * 5) == 0.0


def test_kappa_rejects_mismatched_raters():
    with pytest.raises(ValueError):
        cohens_kappa(["a", "b"], ["a"])
    with pytest.raises(ValueError):
        cohens_kappa([], [])


@pytest.mark.parametrize("value,band", [
    (0.10, "slight"), (0.30, "fair"), (0.50, "moderate"),
    (0.70, "substantial"), (0.83, "almost perfect"),
])
def test_kappa_bands_match_landis_and_koch(value, band):
    assert kappa_band(value) == band


# --- phase 2: agreement beyond kappa ----------------------------------------

from agentcheck.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    confusion,
    difference_claim,
    gwets_ac1,
    kappa_ci,
    modal_prevalence,
    newcombe_diff,
    pabak,
    raw_agreement,
)


def test_kappa_ci_at_the_boundary_is_flagged_rather_than_believed():
    """Perfect agreement on 20 items is not proof of perfect agreement.

    Every resample of a perfectly-agreeing sample also agrees perfectly,
    so the percentile bootstrap returns [1.0, 1.0]. That is not a bug in
    the bootstrap — it cannot express uncertainty about a constant — but
    printed bare it asserts certainty about a quantity measured 20 times,
    which is the Wald failure this module exists to avoid, one level up.

    The interval is therefore flagged, not widened. Inventing a width
    would be fabricating the number instead of reporting the limitation.
    """
    a = ["eq"] * 12 + ["no"] * 8
    ci = kappa_ci(a, list(a), n_boot=2000)

    assert ci.point == pytest.approx(1.0)
    assert ci.width == 0.0
    assert ci.degenerate
    assert "cannot express uncertainty" in str(ci)


def test_an_ordinary_interval_is_not_flagged():
    ci = kappa_ci(["eq", "eq", "no", "no", "eq", "no", "eq", "no"],
                  ["eq", "no", "no", "eq", "eq", "no", "no", "no"],
                  n_boot=2000)
    assert not ci.degenerate
    assert ci.width > 0


def test_kappa_and_ac1_diverge_under_prevalence_skew():
    """The kappa paradox, pinned.

    19 of 20 items are the same label and the raters disagree on exactly
    one. Raw agreement is 95%. Kappa collapses because the marginals make
    chance agreement look near-total; AC1's chance term does not degrade
    that way. Neither number is wrong — they answer different questions,
    and reporting only one of them misleads in a predictable direction.
    """
    a = ["eq"] * 19 + ["no"]
    b = ["eq"] * 20

    assert raw_agreement(a, b) == pytest.approx(0.95)
    assert cohens_kappa(a, b) < 0.20
    assert gwets_ac1(a, b) > 0.80
    assert modal_prevalence(a) == pytest.approx(0.95)


def test_ac1_does_not_punish_a_constant_judge_and_that_is_why_kappa_leads():
    """AC1's own weakness, pinned so it cannot be quietly forgotten.

    The previous test shows kappa understating a good judge on a skewed
    sample. This is the mirror: a judge that answers "equivalent" to
    everything, on a set that is 80% equivalent, has learned nothing.
    Kappa says 0.00. AC1 says about 0.76 — "substantial" on the
    conventional bands — because AC1's chance term is deliberately
    insensitive to the marginal skew this rater is exploiting.

    Both coefficients are correct about different things and neither is
    sufficient alone. This is the concrete reason kappa is the
    pre-registered headline in docs/judge-protocol.md and AC1 is reported
    beside it: leading with AC1 would let a useless judge look moderate.
    """
    human = ["eq"] * 16 + ["narrower"] * 4
    judge = ["eq"] * 20

    assert raw_agreement(human, judge) == pytest.approx(0.80)
    assert cohens_kappa(human, judge) == pytest.approx(0.0, abs=1e-9)
    assert gwets_ac1(human, judge) > 0.70, (
        "if AC1 ever starts punishing a constant rater, the justification "
        "for kappa leading needs rewriting, not this assertion")


def test_pabak_reduces_to_the_two_category_formula():
    a = ["x"] * 8 + ["y"] * 2
    b = ["x"] * 9 + ["y"]
    assert pabak(a, b) == pytest.approx(2 * raw_agreement(a, b) - 1)


def test_non_verdicts_are_excluded_from_agreement_not_redistributed():
    """`unclear` is not a verdict, and must not become one.

    Counting an `unclear`/`equivalent` pair as a disagreement would
    penalise the judge for declining to guess — the option exists so a
    forced choice does not enter the data as noise. Counting it as
    agreement would be worse.
    """
    human = ["eq", "eq", "eq", "no"]
    judge = ["eq", "unclear", "eq", "no"]

    # Three usable pairs, all agreeing.
    assert raw_agreement(human, judge) == pytest.approx(1.0)
    # And the excluded pair is still visible in the matrix.
    assert confusion(human, judge)[("eq", "unclear")] == 1


def test_agreement_refuses_when_nothing_is_left_to_compare():
    with pytest.raises(ValueError, match="non-verdict"):
        raw_agreement(["unclear", "unclear"], ["eq", "eq"])


# --- comparing two rates ----------------------------------------------------

def test_newcombe_finds_a_difference_the_overlap_test_misses():
    """The whole reason this function exists.

    8/10 and 3/10: the two Wilson intervals overlap, so a reader
    comparing them side by side would conclude nothing is established.
    The interval on the difference excludes zero. Overlap is a
    conservative test and this pins the gap between the two.
    """
    low_a, _ = wilson_interval(8, 10)
    _, high_b = wilson_interval(3, 10)
    assert low_a < high_b, \
        "precondition: these Wilson intervals must overlap"

    d = newcombe_diff(8, 10, 3, 10)
    assert d.point == pytest.approx(0.5)
    assert d.low > 0, "the difference interval should exclude zero"


def test_newcombe_stays_inside_minus_one_to_one_at_the_boundary():
    """Where the normal-approximation difference interval breaks."""
    d = newcombe_diff(10, 10, 0, 10)
    assert d.point == pytest.approx(1.0)
    assert -1.0 <= d.low <= d.high <= 1.0


def test_newcombe_is_antisymmetric():
    forward = newcombe_diff(8, 10, 3, 10)
    back = newcombe_diff(3, 10, 8, 10)
    assert forward.point == pytest.approx(-back.point)
    assert forward.low == pytest.approx(-back.high)
    assert forward.high == pytest.approx(-back.low)


def test_difference_claim_says_when_nothing_is_established():
    """An interval containing zero is reported as such, not as 'no difference'."""
    line = difference_claim(5, 10, "a", 4, 10, "b")
    assert "not established" in line
    assert "no difference" not in line


# --- clustering -------------------------------------------------------------

def test_cluster_bootstrap_is_wider_than_pretending_independence():
    """55 tests from 2 root causes are not 55 trials.

    Same 55 values both ways. Resampling them individually gives a narrow
    interval that implicitly claims 55 independent observations;
    resampling the two clusters gives one that reflects the two things
    actually observed. If this ever stops holding, the honest denominator
    has been lost.
    """
    clusters = [[1.0] * 49, [0.0] * 6]
    flat = [v for c in clusters for v in c]

    naive = bootstrap_ci(flat, lambda s: sum(s) / len(s), n_boot=2000)
    clustered = cluster_bootstrap_ci(clusters, n_boot=2000)

    assert clustered.width > naive.width * 2


def test_cluster_bootstrap_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="non-empty"):
        cluster_bootstrap_ci([[], []])


# --- seeding ----------------------------------------------------------------

def test_bootstrap_is_reproducible_and_seed_dependent():
    values = [0.0, 0.44, 0.93, 1.0, 0.0, 1.0]
    def stat(sample):
        return sum(sample) / len(sample)

    assert bootstrap_ci(values, stat, n_boot=1000, seed=7) == \
        bootstrap_ci(values, stat, n_boot=1000, seed=7)
    assert bootstrap_ci(values, stat, n_boot=1000, seed=7) != \
        bootstrap_ci(values, stat, n_boot=1000, seed=8)


def test_bootstrap_refuses_when_the_statistic_is_mostly_undefined():
    """Skipping failed resamples silently would describe a different thing.

    With one distinct label, kappa is 0/0 on nearly every draw. Returning
    an interval built from the few draws where it happened to be defined
    would be an interval for a conditional distribution nobody asked for.
    """
    def only_defined_on_two_labels(sample):
        if len({x for x in sample}) < 2:
            raise ValueError("degenerate")
        return 1.0

    with pytest.raises(ValueError, match="undefined on the sample itself"):
        bootstrap_ci(["a"] * 10, only_defined_on_two_labels, n_boot=200)


def test_bootstrap_refuses_when_most_resamples_are_undefined():
    """Defined on the sample, undefined on nearly every draw.

    Skipping the failures and reporting an interval from the survivors
    would describe a conditional distribution nobody asked about, with a
    denominator that shrank silently — the phase-1 bug class.
    """
    def needs_both_rare_labels(sample):
        if "rare-a" not in sample or "rare-b" not in sample:
            raise ValueError("degenerate")
        return 1.0

    # Each singleton survives a resample with probability ~1 - 1/e, so
    # requiring both leaves roughly 40% of draws usable — under the half
    # this function insists on.
    items = ["common"] * 38 + ["rare-a", "rare-b"]
    with pytest.raises(ValueError, match="undefined on"):
        bootstrap_ci(items, needs_both_rare_labels, n_boot=500)
