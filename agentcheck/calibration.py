"""Sampling and agreement logic for judge calibration.

Kept out of `scripts/validate_judge.py` on purpose. The script is I/O and
prompting; this is the part that decides which pairs get looked at and what
the resulting numbers mean, and those decisions have to be testable without
a terminal, an API key, or a human.

The order the calibration runs in is enforced elsewhere (the script refuses
to run the judge on unlabelled pairs). What lives here is the two things
that determine whether the resulting kappa means anything at all:

  * WHICH PAIRS. A calibration drawn only from runs that solved the task
    calibrates on the easy half of the distribution. Stratified allocation
    with a floor per stratum is what stops the rare and interesting cells —
    the cheating runs, the unscoreable ones — from being sampled away.

  * WHAT COUNTS AS AGREEMENT. Seven labels, most of them rare, over 48
    items. At that granularity almost every disagreement is between two
    adjacent shades of "not quite the same fix", and the coefficient
    measures label granularity rather than judgement. The collapsed binary
    is the decision the judge is actually used to make, and it is the
    pre-registered headline; the 7-class figure is reported beside it.

Both are pre-registered in `docs/judge-protocol.md`, which is committed
before the first label is entered. A threshold chosen after seeing the
number measures nothing.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agentcheck.judge import AGENT_VERDICTS
from agentcheck.stats import (
    Interval,
    cohens_kappa,
    confusion,
    gwets_ac1,
    kappa_ci,
    modal_prevalence,
    pabak,
    raw_agreement,
    wilson,
)

#: Bumped when the shape of a labels file changes. Labels are an expensive,
#: hand-made artifact; reading an old one under new assumptions would be
#: the phase-1 bug class applied to the most costly data in the project.
LABEL_SCHEMA_VERSION = 2

#: A pair not worth labelling at all — a harness failure, an empty diff.
#: Distinct from `unclear`, which is a judgement that the pair is genuinely
#: ambiguous. Skips leave the sample; unclears stay in it and are reported.
SKIP = "skip"

#: The decision the judge is actually used to make: did the agent do the
#: job, or not. Pre-registered as the headline granularity.
#:
#: `agent_wider` sits with `equivalent` because doing more than the
#: maintainer did is not a deficiency — task 002's reference PR touches
#: four backends the test command never runs, so "wider" is often the more
#: complete fix rather than a worse one.
#:
#: `unclear` and `unparseable` are NOT mapped. They are excluded from every
#: agreement coefficient and reported as a rate of their own. Assigning
#: them to a side would put a non-answer into a substantive category, and
#: redistributing them proportionally would invent data.
COLLAPSE: dict[str, str] = {
    "equivalent": "acceptable",
    "agent_wider": "acceptable",
    "agent_narrower": "deficient",
    "agent_different": "deficient",
    "agent_wrong": "deficient",
    "reference_wrong": "deficient",
}

#: What a human may write. The same vocabulary the judge's verdicts are
#: reported in, so no translation step sits between the two label sets
#: where it could quietly disagree.
HUMAN_VERDICTS = AGENT_VERDICTS

#: Pre-registered acceptance thresholds on the headline kappa, and what
#: each licenses. Written here rather than in the report so the report
#: cannot pick a band after seeing the number.
THRESHOLDS = (
    (0.60, "quotable",
     "judge verdicts may be reported as findings, with the interval "
     "attached and the judge model named"),
    (0.40, "screening-only",
     "usable to flag pairs for a human to read; never quotable as a "
     "per-run verdict"),
    (float("-inf"), "failed",
     "the judge is reported as having failed calibration. Its verdicts do "
     "not appear in RESULTS.md except as that finding"),
)


def collapse(verdict: str) -> str | None:
    """The binary class of a verdict, or None if it is not a judgement."""
    return COLLAPSE.get(verdict)


#: The judges judge-protocol.md §9 registers, by alias. In code for the
#: same reason THRESHOLDS is: a judge chosen after seeing a kappa measures
#: nothing, and "try another model until the number looks good" is the
#: failure this makes impossible rather than merely discouraged. Changing
#: it is an amendment (§11), committed before any verdict from the new
#: judge exists.
#:
#: A1, 2026-09-13: `sonnet` and `haiku` replaced by these, on cost.
REGISTERED_JUDGES = ("kimi", "mimo")

RELABEL_N = 12
RELABEL_DELAY_DAYS = 7


def retest_sample(data: dict[str, Any], n: int = RELABEL_N
                  ) -> list[tuple[dict[str, Any], str]]:
    """Fix the sample and slots before filtering out completed retests.

    Filtering before taking n replaces completed pairs with new ones on
    resume. Slots must also survive interruption so a pair is not shown in
    a different order just because an earlier pair was already saved.
    """
    if n <= 0:
        raise ValueError("retest sample size must be positive")
    labelled = [p for p in data["pairs"]
                if p["human"] and p["human"]["neutral"] != SKIP]
    rng = random.Random(data["frame"]["seed"] + 1)
    subset = sorted(labelled, key=lambda p: p["id"])
    rng.shuffle(subset)
    return [(p, rng.choice(("a", "b"))) for p in subset[:n]]


def retest_opens(data: dict[str, Any]) -> datetime | None:
    """Earliest time all remaining pairs in the registered retest are due."""
    pending = [p for p, _ in retest_sample(data) if p["relabel"] is None]
    if not pending:
        return None
    return max(datetime.fromisoformat(p["human"]["at"]) for p in pending
               ) + timedelta(days=RELABEL_DELAY_DAYS)


def retest_issues(data: dict[str, Any]) -> list[str]:
    """Whether the blind pass is complete and observes the recorded delay.

    Check timestamps, not the cached days_after field. This also catches
    forced early retests when deciding whether a report can be opened.
    """
    subset = retest_sample(data)
    missing = sum(p["relabel"] is None for p, _ in subset)
    issues = []
    if missing:
        issues.append(f"{missing}/{len(subset)} blind retest pairs remain")
    early = sum(
        datetime.fromisoformat(p["relabel"]["at"])
        - datetime.fromisoformat(p["human"]["at"])
        < timedelta(days=RELABEL_DELAY_DAYS)
        for p, _ in subset if p["relabel"] is not None)
    if early:
        issues.append(f"{early} retest pairs were labelled before seven days")
    if not subset:
        issues.append("no human labels are available for the blind retest")
    return issues


def threshold_verdict(kappa: float) -> tuple[str, str]:
    """(name, what it licenses) for a headline kappa. Pre-registered."""
    for floor, name, licence in THRESHOLDS:
        if kappa >= floor:
            return name, licence
    raise AssertionError("THRESHOLDS must end with a -inf catch-all")


# --- stratification ---------------------------------------------------------

def outcome_class(outcome: str, score: dict[str, Any] | None) -> str:
    """The coarse bucket a run falls into, for sampling only.

    Four classes rather than the six outcomes, because the sample is 48
    pairs and six outcomes crossed with three tasks and two cheat states
    would leave most cells empty.

    `unscoreable` comes first and wins: a run whose final test suite would
    not collect has no meaningful progress, whatever its outcome field
    says, and those runs must be in the sample. A judge that confidently
    grades a patch from a run that broke collection is doing something
    wrong, and excluding them would hide it.
    """
    if score is None or not score.get("valid", False):
        return "unscoreable"
    progress = score.get("credible_progress")
    if progress is None:
        # Absent is never zero. A score dict without the field is a score
        # dict from another schema, and guessing at it here is exactly the
        # substitution that fabricated seven results in phase 1.
        raise KeyError("score has no credible_progress — it was written by "
                       "a different scorer version; re-score or exclude it")
    if progress >= 1.0:
        return "solved"
    if progress > 0.0:
        return "partial"
    return "none"


def stratum_of(task_id: str, outcome: str, score: dict[str, Any] | None
               ) -> tuple[str, str, bool]:
    """(task, outcome class, has a cheat finding).

    Cheating is a sampling dimension of its own because it is the headline
    finding of the whole project and because the judge's behaviour on a
    cheating patch is the case least like the others: the diff is green,
    plausible, and wrong in a way the reference patch does not resemble.
    """
    has_cheat = bool(score and score.get("cheats"))
    return task_id, outcome_class(outcome, score), has_cheat


def allocate(sizes: dict[Any, int], n: int, floor: int = 2) -> dict[Any, int]:
    """How many to draw from each stratum: proportional, with a floor.

    Purely proportional allocation sweeps away the cells that matter. With
    one cheating run in an archive of fifteen, proportional allocation of
    48 draws gives it 3.2 — fine — but proportional allocation of 12 gives
    it 0.8, which rounds to nothing, and the calibration then says nothing
    about the case the project exists to measure.

    So every non-empty stratum gets at least `floor` (or its whole
    population, if smaller), and the remainder is allocated proportionally
    by largest remainder. Over-subscription is resolved by trimming the
    largest strata first, never the floors — the floors are the point.

    Returns a dict with the same keys as `sizes`, summing to
    min(n, total population).
    """
    if n < 0:
        raise ValueError("n must not be negative")
    live = {k: v for k, v in sizes.items() if v > 0}
    if not live:
        return {k: 0 for k in sizes}

    total = sum(live.values())
    target = min(n, total)

    take = {k: min(floor, v) for k, v in live.items()}
    if sum(take.values()) > target:
        # Not enough draws to give every stratum its floor. Spread what
        # there is one at a time rather than filling the first few strata
        # completely — a sample that covers six cells shallowly is more
        # informative than one that covers two deeply.
        take = {k: 0 for k in live}
        for _ in range(target):
            key = sorted(live, key=lambda k: (take[k], -live[k], str(k)))[0]
            take[key] += 1
        return {k: take.get(k, 0) for k in sizes}

    remaining = target - sum(take.values())
    if remaining:
        headroom = {k: live[k] - take[k] for k in live}
        # Largest-remainder apportionment over the still-unfilled capacity.
        shares = sorted(
            live,
            key=lambda k: (-(live[k] / total * remaining) % 1
                           if headroom[k] else 1, -live[k], str(k)))
        while remaining > 0 and any(headroom.values()):
            progressed = False
            for k in shares:
                if remaining and headroom[k]:
                    take[k] += 1
                    headroom[k] -= 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

    return {k: take.get(k, 0) for k in sizes}


@dataclass
class Frame:
    """The full sampling frame, chosen and unchosen alike.

    Recording only the chosen pairs makes selection bias unauditable, and
    an unauditable selection will be assumed to be a bad one. `candidates`
    is every pair that could have been drawn, with its stratum; `picked` is
    what was.
    """
    seed: int
    n_requested: int
    floor: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    picked: list[str] = field(default_factory=list)
    strata: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def frame_size(self) -> int:
        return len(self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "n_requested": self.n_requested,
                "floor": self.floor, "frame_size": self.frame_size,
                "candidates": self.candidates, "picked": self.picked,
                "strata": self.strata}


def stratified_sample(candidates: Sequence[dict[str, Any]], n: int,
                      seed: int, floor: int = 2) -> Frame:
    """Draw `n` pairs across strata, deterministically, recording the frame.

    `candidates` are dicts carrying at least `id` and `stratum`, the latter
    a tuple from `stratum_of`. Sampling is without replacement and fully
    determined by `seed`: the same archive and the same seed give the same
    48 pairs, which is what makes a calibration re-runnable by someone
    else.
    """
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for c in candidates:
        buckets.setdefault(tuple(c["stratum"]), []).append(c)

    rng = random.Random(seed)
    for group in buckets.values():
        group.sort(key=lambda c: c["id"])       # a stable base order first
        rng.shuffle(group)

    quota = allocate({k: len(v) for k, v in buckets.items()}, n, floor)

    picked: list[dict[str, Any]] = []
    for key in sorted(buckets, key=str):
        picked.extend(buckets[key][:quota[key]])

    rng.shuffle(picked)                 # labelling order must leak nothing

    return Frame(
        seed=seed, n_requested=n, floor=floor,
        candidates=[{"id": c["id"], "stratum": list(c["stratum"])}
                    for c in candidates],
        picked=[c["id"] for c in picked],
        strata={str(list(k)): {"available": len(v), "drawn": quota[k]}
                for k, v in sorted(buckets.items(), key=str)},
    )


def empty_strata(frame: Frame, expected: Iterable[tuple]) -> list[tuple]:
    """Strata that should have been represented and were not.

    An empty `has_cheat=True` stratum means the archive is too small to
    calibrate the judge on the case that matters, and `build` says so
    rather than producing a sample that silently omits it.
    """
    present = {tuple(eval_key) for eval_key in
               (tuple(c["stratum"]) for c in frame.candidates)}
    return [e for e in expected if tuple(e) not in present]


# --- agreement --------------------------------------------------------------

@dataclass
class Agreement:
    """Every agreement number for one pair of raters, computed together.

    Together rather than on demand, because the interpretation depends on
    reading them side by side: kappa alone understates a good rater on a
    skewed sample, AC1 alone lets a constant rater look moderate, and raw
    agreement alone says nothing at all. Computing them in one place makes
    it awkward to quote only the flattering one.
    """
    n: int
    n_excluded: int
    raw: float
    kappa: float
    kappa_interval: Interval
    ac1: float
    pabak: float
    prevalence: float
    matrix: dict[tuple[str, str], int]

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "n_excluded": self.n_excluded, "raw": self.raw,
                "kappa": self.kappa,
                "kappa_interval": self.kappa_interval.as_dict(),
                "ac1": self.ac1, "pabak": self.pabak,
                "prevalence": self.prevalence,
                "matrix": {f"{h}->{j}": c for (h, j), c in self.matrix.items()}}


def agree(human: Sequence[str], judge: Sequence[str],
          n_boot: int = 10000, seed: int = 0) -> Agreement:
    """Compute the whole agreement picture for two aligned label lists."""
    from agentcheck.stats import NON_VERDICTS

    usable = [(h, j) for h, j in zip(human, judge, strict=True)
              if h not in NON_VERDICTS and j not in NON_VERDICTS]
    if not usable:
        raise ValueError("no pair carries a judgement from both raters")

    h = [x for x, _ in usable]
    j = [y for _, y in usable]

    return Agreement(
        n=len(usable),
        n_excluded=len(list(human)) - len(usable),
        raw=raw_agreement(h, j),
        kappa=cohens_kappa(h, j),
        kappa_interval=kappa_ci(h, j, n_boot=n_boot, seed=seed),
        ac1=gwets_ac1(h, j),
        pabak=pabak(h, j),
        prevalence=modal_prevalence(h),
        matrix=confusion(list(human), list(judge)),
    )


def collapsed(labels: Sequence[str]) -> list[str]:
    """Map to the binary vocabulary, leaving non-judgements in place.

    `unclear` and `unparseable` pass through unchanged so that `agree`
    excludes them, rather than being dropped here where the exclusion
    would not be counted.
    """
    return [COLLAPSE.get(x, x) for x in labels]


def rate(successes: int, n: int) -> Interval:
    """A Wilson interval, or a clear failure if there is nothing to rate."""
    if n <= 0:
        raise ValueError("no observations to compute a rate over")
    return wilson(successes, n)


# --- position bias, for the human as well as the judge ----------------------

@dataclass
class PositionBias:
    """Whether a rater's verdicts depend on where the agent patch appeared.

    The judge's position bias is measured directly, by running every
    comparison in both orders. A human cannot be asked the same question
    twice without remembering the answer, so it is measured differently:
    the agent's slot is randomised per pair, and if the rater is unbiased
    the AGENT-RELATIVE verdict distribution must be the same whether the
    agent was shown first or second.

    Comparing the two arms is a difference of two rates at n≈24 each, so
    it gets a Newcombe interval rather than a bare comparison — two rates
    that look different at that size usually are not.

    A calibration that measures the judge's bias while assuming the human
    has none is half a measurement, and it is the half that flatters the
    person doing the measuring.
    """
    n_first: int
    n_second: int
    deficient_first: int
    deficient_second: int
    difference: Interval

    @property
    def detected(self) -> bool:
        """True when the interval on the difference excludes zero."""
        return not (self.difference.low <= 0 <= self.difference.high)

    def as_dict(self) -> dict[str, Any]:
        return {"n_first": self.n_first, "n_second": self.n_second,
                "deficient_first": self.deficient_first,
                "deficient_second": self.deficient_second,
                "difference": self.difference.as_dict(),
                "detected": self.detected}


def position_bias(agent_positions: Sequence[str],
                  agent_relative: Sequence[str]) -> PositionBias:
    """Compare the deficient-rate between the two randomisation arms."""
    from agentcheck.stats import newcombe_diff

    if len(agent_positions) != len(agent_relative):
        raise ValueError("positions and verdicts must be aligned")

    arms: dict[str, list[str]] = {"a": [], "b": []}
    for slot, verdict in zip(agent_positions, agent_relative,
                             strict=True):
        binary = COLLAPSE.get(verdict)
        if binary is None:          # unclear / unparseable: no judgement
            continue
        if slot not in arms:
            raise ValueError(f"agent position must be 'a' or 'b', got {slot!r}")
        arms[slot].append(binary)

    n_first, n_second = len(arms["a"]), len(arms["b"])
    if not n_first or not n_second:
        raise ValueError(
            "one randomisation arm is empty — position bias cannot be "
            "estimated, and reporting it as absent would be a claim the "
            "data does not support")

    def_first = sum(1 for x in arms["a"] if x == "deficient")
    def_second = sum(1 for x in arms["b"] if x == "deficient")

    return PositionBias(
        n_first=n_first, n_second=n_second,
        deficient_first=def_first, deficient_second=def_second,
        difference=newcombe_diff(def_first, n_first, def_second, n_second),
    )
