"""Small-sample statistics for agentcheck.

At n=4 per configuration, with observed values spanning 0% to 100%, almost
nothing is claimable about the population. What IS claimable is the
observation itself plus an honest interval, and the job of this module is
to make the interval impossible to omit.

Three rules, the first two learned from the data already collected:

  * Proportions get a Wilson interval, never a normal approximation.
    Wald breaks down exactly where these runs live — at 0/4 and 4/4 it
    reports a width of zero, which reads as certainty about a quantity
    measured four times.
  * Per-test progress is NOT a proportion of independent trials. Task 002's
    55 failures come from two root causes; one fix moves 49 tests at once.
    An interval computed on n=55 would be roughly five times too narrow.
    The honest denominator is the number of root causes, and progress is
    reported as a distribution over runs rather than as a mean.
  * Agreement between two raters is reported as kappa, never as raw
    percent. If 80% of validation pairs are "equivalent", a judge that
    answers "equivalent" every time scores 80% raw agreement while knowing
    nothing at all. Kappa corrects for chance and gives it 0.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: Two-sided normal quantiles for the levels worth offering.
_Z = {0.90: 1.6448536269514722,
      0.95: 1.959963984540054,
      0.99: 2.5758293035489004}


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int
    level: float = 0.95
    #: Set when the interval carries no information about uncertainty:
    #: every bootstrap resample produced the same value, so the bounds are
    #: an artifact of the method rather than a measurement.
    #:
    #: This happens at the boundary. With 20 items on which two raters
    #: agree perfectly, every resample also agrees perfectly, and the
    #: percentile bootstrap returns [1.0, 1.0] — an interval that asserts
    #: certainty about a quantity measured 20 times. That is the Wald
    #: failure this module exists to avoid, reappearing one level up in a
    #: method that cannot be blamed for it: the bootstrap is not wrong,
    #: it simply cannot express uncertainty about a constant.
    #:
    #: The flag exists so no caller can print such an interval without
    #: knowing. It is never repaired by widening the bounds — an invented
    #: width is a fabricated number.
    degenerate: bool = False

    def __str__(self) -> str:
        body = (f"{self.point:.0%} "
                f"[{self.low:.0%}, {self.high:.0%}] (n={self.n})")
        if self.degenerate:
            body += ("  [no spread across resamples — the bootstrap cannot "
                     "express uncertainty here; read n, not the bounds]")
        return body

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict[str, float | int | bool]:
        return {"point": self.point, "low": self.low, "high": self.high,
                "n": self.n, "level": self.level,
                "degenerate": self.degenerate}


def wilson(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over Wald because these samples are tiny and the observed
    proportions sit at the boundaries. Wald gives [0, 0] for 0 successes
    out of 4 — an interval that asserts the true rate is exactly zero on
    the strength of four runs. Wilson gives [0%, 49%], which is the actual
    state of knowledge.
    """
    if n <= 0:
        raise ValueError("no observations")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n}")

    z = _Z.get(level)
    if z is None:
        raise ValueError(f"unsupported level {level}; use one of {sorted(_Z)}")

    low, high = wilson_interval(successes, n, z)
    return Interval(point=successes / n, low=low, high=high,
                    n=n, level=level)


def describe(values: Sequence[float]) -> str:
    """A distribution summary that does not lead with a mean.

    The stored progress values are bimodal — clusters at 0%, ~56%, ~93%
    and 100% — and the mean of a bimodal distribution names a value no run
    took. Median and full range first; the mean only alongside them.
    """
    if not values:
        return "(no runs)"
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    mean = sum(ordered) / n
    body = (f"median {median:.0%}   range {ordered[0]:.0%}–{ordered[-1]:.0%}"
            f"   mean {mean:.0%}   n={n}")
    if n < 10:
        body += ("\n    all: " + ", ".join(f"{v:.0%}" for v in ordered)
                 + "\n    too few runs for an interval on a continuous "
                   "measure — read the values, not the mean")
    return body


def claim(successes: int, n: int, label: str, level: float = 0.95) -> str:
    """One line stating a rate with its interval, ready to paste."""
    ci = wilson(successes, n, level)
    return (f"{label}: {successes}/{n} = {ci.point:.0%} "
            f"[{ci.low:.0%}, {ci.high:.0%}] {int(level * 100)}% Wilson")


def wilson_interval(successes: int, n: int, z: float = 1.96
                    ) -> tuple[float, float]:
    """(low, high) for a binomial proportion, as a plain tuple.

    THE implementation; `wilson` wraps it. Two functions computing the
    same interval is two chances to compute it differently — the first
    draft of this file had one using z=1.96 and the other the exact
    1.95996..., which disagreed in the fourth decimal for no reason
    anybody would ever have found.

    `z` rather than a confidence level so a caller can pass an
    unconventional one; 1.96 is the conventional 95%.
    """
    if n <= 0:
        raise ValueError("no observations")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n}")

    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom

    low = max(0.0, centre - spread)
    high = min(1.0, centre + spread)

    # Snap the boundaries exactly. At k=0 the arithmetic lands on 2.8e-17
    # instead of 0, and at k=n on 0.9999999999999999 instead of 1, which
    # breaks the one invariant every caller relies on: low <= p <= high.
    # The exact values are what Wilson gives at the boundaries, so this is
    # a correction, not a fudge.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return min(low, p), max(high, p)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 0
                      ) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    For continuous per-run measures — credible progress, cost, wall time —
    where the sample is far too small and too lumpy for a t interval. The
    stored progress values cluster at 0%, ~56%, ~93% and 100%; nothing
    about that is normal, and resampling makes no distributional claim.

    Seeded by default. A benchmark that reports a different interval each
    time it is run has added noise to its own results, and a reader cannot
    tell that noise from a real change.

    This is honest about its input but cannot rescue it: at n=4 the
    resamples only ever contain the four observed values, so the interval
    is a description of those four numbers, not an inference about a
    population. Report it next to the values, never instead of them.
    """
    if not values:
        raise ValueError("no observations")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    data = list(values)
    n = len(data)
    if n == 1:
        return data[0], data[0]

    rng = random.Random(seed)
    means = sorted(
        sum(data[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement between two raters over the same items.

    Raw percent agreement is not interpretable when the labels are
    unbalanced, and the judge validation set is very unbalanced: most
    agent patches that do anything at all are "equivalent" to the
    maintainer's. A judge that answers "equivalent" unconditionally scores
    80% raw agreement on such a set and has learned nothing. Kappa scores
    it 0.

    Conventional bands (Landis & Koch): <0.20 slight, 0.21-0.40 fair,
    0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.
    RoadmapBench reports 0.83 between two human annotators on a similar
    judgement, which is the number to aim at — a judge well below it is
    not measuring what the humans were measuring.

    Returns 0.0 when chance agreement is total (both raters used exactly
    one label for everything). The ratio is 0/0 there, and 0.0 is the
    honest reading: perfect agreement that carries no information.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no items to compare")

    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label]
                   for label in set(count_a) | set(count_b)) / (n * n)

    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def kappa_band(kappa: float) -> str:
    """The conventional label for a kappa, so a number is not left bare."""
    for threshold, name in ((0.80, "almost perfect"), (0.60, "substantial"),
                            (0.40, "moderate"), (0.20, "fair")):
        if kappa > threshold:
            return name
    return "slight"


# --- agreement, beyond kappa ------------------------------------------------

#: Labels that mean "no judgement was made", excluded from every agreement
#: statistic and reported as a rate of their own. Folding a non-answer into
#: a substantive category — or worse, redistributing it across the others —
#: puts harness behaviour into a result. `unclear` is the judge saying it
#: looked and could not decide; `unparseable` is never having received an
#: answer. Neither is a verdict.
NON_VERDICTS = frozenset({"unclear", "unparseable", ""})


def _paired(a: Sequence[str], b: Sequence[str]) -> list[tuple[str, str]]:
    """Aligned label pairs, with non-verdicts dropped and lengths checked.

    Two length checks, deliberately. The explicit one gives a message
    naming both lengths, which is what a caller needs; `strict=True` on
    the zip is the belt, because a silently truncated pairing would
    compute an agreement coefficient over a prefix of the data and return
    a perfectly plausible number. That is the failure mode this whole
    project is about, and it is not one to leave to a comment.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no items to compare")
    return [(x, y) for x, y in zip(a, b, strict=True)
            if x not in NON_VERDICTS and y not in NON_VERDICTS]


def raw_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """The uncorrected fraction of items both raters labelled the same.

    Never reported alone — it is the number kappa exists to deflate — but
    always reported alongside, because a kappa of 0.2 at 95% raw agreement
    and a kappa of 0.2 at 55% raw agreement describe very different
    situations and the scalar cannot tell them apart.
    """
    pairs = _paired(a, b)
    if not pairs:
        raise ValueError("every item was a non-verdict for one rater or both")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def gwets_ac1(a: Sequence[str], b: Sequence[str]) -> float:
    """Gwet's AC1: chance-corrected agreement that survives prevalence skew.

    Kappa has a documented pathology, and this validation set walks
    straight into it. When one category dominates — most agent patches
    that do anything at all are "equivalent" to the maintainer's — kappa's
    chance term is estimated from the marginals and becomes very large, so
    kappa collapses towards zero even when the two raters agree on nearly
    every item. That is the kappa paradox: high agreement, low kappa.

    The protocol anticipates the skew in one direction only — a judge
    that answers "equivalent" unconditionally must not score well, and
    kappa correctly gives it 0. The other direction is just as real: a
    genuinely good judge on a skewed sample is punished by the same
    arithmetic, and reporting only kappa would understate it.

    AC1 estimates chance agreement from the propensity of a random rating
    rather than from the product of the marginals, which does not degrade
    as prevalence rises. It is not a replacement for kappa. Report BOTH,
    plus raw agreement and the modal prevalence, and read them together.
    Reporting whichever of the two is higher is the thing this function
    exists to make harder, not easier.

    AC1 has a criticism of its own, and it is the reason kappa stays the
    headline statistic in `docs/judge-protocol.md` rather than being
    replaced. A judge that answers "equivalent" unconditionally on a set
    that is 80% equivalent scores kappa 0.00 — correctly, it has learned
    nothing — but AC1 about 0.76, because AC1's chance term is deliberately
    insensitive to exactly the marginal skew that constant rater exploits.
    That case is pinned in the tests. Leading with AC1 would let a useless
    judge look moderate, which is the specific failure the protocol names.

    So: kappa first, AC1 beside it, and where the two disagree, say which
    of the two situations the confusion matrix shows.

    Returns 0.0 when chance agreement is total, matching `cohens_kappa`.
    """
    pairs = _paired(a, b)
    if not pairs:
        raise ValueError("every item was a non-verdict for one rater or both")

    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n

    labels = {x for x, _ in pairs} | {y for _, y in pairs}
    k = len(labels)
    if k < 2:
        # One label in the whole set. Both raters agree perfectly and the
        # comparison carries no information; 0.0 is the honest reading,
        # and it is what kappa returns here too.
        return 0.0

    # pi_hat: the mean of each label's probability across the two raters.
    counts_a, counts_b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    chance = sum(
        ((counts_a[label] / n + counts_b[label] / n) / 2)
        * (1 - (counts_a[label] / n + counts_b[label] / n) / 2)
        for label in labels
    ) / (k - 1)

    if chance >= 1.0:
        return 0.0
    return (observed - chance) / (1 - chance)


def pabak(a: Sequence[str], b: Sequence[str]) -> float:
    """Prevalence-adjusted bias-adjusted kappa.

    The older and cruder correction for the same problem AC1 addresses:
    for two categories it reduces to `2 * observed - 1`, which is kappa
    computed as if the marginals were balanced. Included so the two
    adjustments can be compared rather than one being taken on faith, and
    because it is the one a reviewer is most likely to have heard of.

    Generalised to k categories as `(k * observed - 1) / (k - 1)`. It
    discards real information about the marginals — that is the criticism
    of it, and it is why AC1 is the one to lead with.
    """
    pairs = _paired(a, b)
    if not pairs:
        raise ValueError("every item was a non-verdict for one rater or both")

    observed = sum(1 for x, y in pairs if x == y) / len(pairs)
    k = len({x for x, _ in pairs} | {y for _, y in pairs})
    if k < 2:
        return 0.0
    return (k * observed - 1) / (k - 1)


def confusion(a: Sequence[str], b: Sequence[str]
              ) -> dict[tuple[str, str], int]:
    """Counts of every (rater A label, rater B label) combination.

    Includes non-verdicts, unlike the agreement statistics, because where
    the judge said `unclear` and the human did not is exactly the cell a
    reader wants to see. The matrix is more informative than any scalar
    derived from it and costs nothing to print; the scalars are what get
    quoted, and the matrix is what makes them checkable.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    return dict(Counter(zip(a, b, strict=True)))


def modal_prevalence(labels: Sequence[str]) -> float:
    """The share of the most common label.

    Reported next to kappa always. It is the single number that says how
    much to distrust kappa on this sample: at 0.5 the correction is mild,
    at 0.9 kappa is mostly measuring the marginals.
    """
    if not labels:
        raise ValueError("no labels")
    return Counter(labels).most_common(1)[0][1] / len(labels)


# --- bootstrap --------------------------------------------------------------

def bootstrap_ci(values: Sequence[Any],
                 statistic: Callable[[list[Any]], float],
                 n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Interval:
    """Percentile bootstrap for an arbitrary statistic over resampled items.

    Generalises `bootstrap_mean_ci` so agreement coefficients get intervals
    too. `values` is a sequence of ITEMS — whatever the unit of resampling
    is — and `statistic` maps a resampled list of them to a float.

    Resampling the items, not the labels, is the part that is easy to get
    wrong. For a rater-agreement interval the item is the labelled PAIR:
    resampling the two label lists independently would destroy the pairing
    and estimate the interval of a statistic nobody computed.

    Seeded, like everything else here, because a benchmark that reports a
    different interval on each invocation has added noise to its own
    results and a reader cannot separate that noise from a real change.

    Resamples on which the statistic is undefined — a bootstrap draw in
    which every pair happens to carry the same label, where kappa is 0/0 —
    are skipped and counted. If more than half the draws fail, the sample
    is too small or too degenerate for this interval and that raises,
    rather than returning an interval computed from the survivors, which
    would be an interval for a conditional distribution nobody asked about.
    """
    items = list(values)
    if not items:
        raise ValueError("no observations")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    try:
        point = float(statistic(items))
    except (ValueError, ZeroDivisionError) as e:
        # The statistic is undefined on the observed sample itself, so
        # there is nothing to build an interval around. Say that, rather
        # than letting the caller read a bare "degenerate" out of a
        # function it did not call.
        raise ValueError(
            f"the statistic is undefined on the sample itself ({e}) — "
            f"there is no point estimate to bootstrap around") from e

    n = len(items)
    if n == 1:
        return Interval(point=point, low=point, high=point, n=1,
                        level=1 - alpha, degenerate=True)

    rng = random.Random(seed)
    draws: list[float] = []
    failures = 0
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(float(statistic(sample)))
        except (ValueError, ZeroDivisionError):
            failures += 1

    if failures > n_boot / 2:
        raise ValueError(
            f"the statistic was undefined on {failures} of {n_boot} "
            f"resamples — the sample is too small or too degenerate for a "
            f"bootstrap interval. Report the point estimate and n instead.")

    draws.sort()
    m = len(draws)
    low = draws[int((alpha / 2) * m)]
    high = draws[min(m - 1, int((1 - alpha / 2) * m))]
    return Interval(point=point, low=low, high=high, n=n, level=1 - alpha,
                    degenerate=draws[0] == draws[-1])


def kappa_ci(a: Sequence[str], b: Sequence[str], n_boot: int = 10000,
             alpha: float = 0.05, seed: int = 0) -> Interval:
    """Bootstrap interval for Cohen's kappa, resampling labelled pairs.

    A point kappa at n=48 has a standard error wide enough to span two of
    the Landis–Koch bands, so "kappa = 0.61, substantial" and "kappa =
    0.58, moderate" can be the same result seen twice. Quoting a bare
    kappa in a project whose central claim is that single measurements are
    not evidence would be the same error one level up.
    """
    pairs = _paired(a, b)
    if not pairs:
        raise ValueError("every item was a non-verdict for one rater or both")
    return bootstrap_ci(
        pairs,
        lambda sample: cohens_kappa([x for x, _ in sample],
                                    [y for _, y in sample]),
        n_boot=n_boot, alpha=alpha, seed=seed)


def cluster_bootstrap_ci(
        clusters: Sequence[Sequence[float]],
        statistic: Callable[[list[Any]], float] | None = None,
        n_boot: int = 10000, alpha: float = 0.05,
        seed: int = 0) -> Interval:
    """Bootstrap that resamples CLUSTERS, not the observations inside them.

    Task 002's broken state is 55 failing tests arising from two root
    causes: one correct edit moves 49 of them at once. Treating those 55
    as independent trials inflates the effective sample size by roughly
    27x, and the interval that comes out is about five times too narrow.
    The README says this in prose; this is the arithmetic version.

    `clusters` is a list of lists — one inner list per root cause, holding
    that cause's per-test values. Each draw takes whole clusters with
    replacement and pools them, which propagates the fact that the tests
    within a cause move together.

    The default statistic is the mean over the pooled observations. Pass
    another for rates.
    """
    groups = [list(c) for c in clusters if len(c)]
    if not groups:
        raise ValueError("no non-empty clusters")

    def pooled_mean(sample: Sequence[Sequence[float]]) -> float:
        flat = [v for c in sample for v in c]
        if not flat:
            raise ValueError("empty resample")
        return sum(flat) / len(flat)

    return bootstrap_ci(groups, statistic or pooled_mean,
                        n_boot=n_boot, alpha=alpha, seed=seed)


# --- comparing two rates ----------------------------------------------------

def newcombe_diff(s1: int, n1: int, s2: int, n2: int,
                  level: float = 0.95) -> Interval:
    """Newcombe's hybrid-score interval for the DIFFERENCE of two rates.

    The right way to say "haiku cheats more often than ds-flash" at n=10
    each, and the reason it needs its own function is that the obvious
    alternative is wrong in both directions:

      * Two Wilson intervals that OVERLAP do not imply no difference. The
        overlap test is conservative, and at these sample sizes it will
        fail to detect differences that the paired interval finds.
      * Two Wilson intervals that do NOT overlap imply a larger difference
        than the data supports. Non-overlap is a stronger condition than
        the difference excluding zero.

    Neither is the interval on p1 - p2, and neither is what a reader
    assumes when shown two intervals side by side. Newcombe builds the
    difference interval from the two Wilson intervals directly, and it
    behaves at the boundaries where these samples live — which is where
    the normal-approximation difference interval extends past ±1.

    Returns an Interval whose `point` is p1 - p2, so it can be negative;
    `n` is n1 + n2, which is the total number of observations behind it
    and not a sample size in the usual sense. An interval containing 0 is
    a difference the data does not establish, and RESULTS.md says so in
    those words rather than calling it "no difference".
    """
    z = _Z.get(level)
    if z is None:
        raise ValueError(f"unsupported level {level}; use one of {sorted(_Z)}")

    l1, u1 = wilson_interval(s1, n1, z)
    l2, u2 = wilson_interval(s2, n2, z)
    p1, p2 = s1 / n1, s2 / n2

    # Newcombe's method 10: the distance from each proportion to its own
    # Wilson bound is combined in quadrature, taking the bound on each
    # side that pushes the difference outward.
    low = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return Interval(point=p1 - p2, low=max(-1.0, low), high=min(1.0, high),
                    n=n1 + n2, level=level)


def difference_claim(s1: int, n1: int, label1: str,
                     s2: int, n2: int, label2: str,
                     level: float = 0.95) -> str:
    """One line comparing two rates, which refuses to overstate itself."""
    d = newcombe_diff(s1, n1, s2, n2, level)
    establishes = not (d.low <= 0 <= d.high)
    verdict = ("the interval excludes 0"
               if establishes else
               "the interval contains 0 — this difference is not established")
    return (f"{label1} {s1}/{n1} = {s1 / n1:.0%} vs "
            f"{label2} {s2}/{n2} = {s2 / n2:.0%}; "
            f"difference {d.point:+.0%} [{d.low:+.0%}, {d.high:+.0%}] "
            f"{int(level * 100)}% Newcombe — {verdict}")
