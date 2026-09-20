#!/usr/bin/env python3
"""Read saved agent traces.

    python3 inspect_run.py runs/002-databases-ds-flash-03.json
    python3 inspect_run.py runs/*flash*.json --compare
    python3 inspect_run.py runs/*.json --summary

Reading traces is the actual work of this project. The outcome field says
what happened; the trajectory says why, and that is where the failure
taxonomy comes from.

Everything here loads through record.load. When it read raw JSON instead,
`--summary` averaged 28 unversioned runs from four different
configurations and two prompt versions into a single "mean 48%", and
counted seven schema markers as cheating runs — which doubled the reported
cheat rate, the headline result of the project.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from agentcheck.record import (
    RunRecord,
    describe_config,
    group_by_config,
    load_all,
)
from agentcheck.scorer import Score
from agentcheck.stats import bootstrap_mean_ci, claim, describe

EXPLORE = {"list_files", "read_file", "search_code",
           "package_version", "package_source", "read_package_file"}

#: Records that no longer load live here, moved by scripts/archive_runs.py
#: rather than deleted. They are excluded from the default read so that 28
#: rejection lines do not bury 15 results — never dropped, only relocated,
#: and --include-archive brings them back.
ARCHIVE_DIR = Path("runs/archive")


def _score(r: RunRecord) -> Score | None:
    if not r.score:
        return None
    try:
        return Score.from_dict(r.score)
    except (KeyError, TypeError):
        return None


def _report_rejections(rejected: list[str]) -> None:
    if not rejected:
        return
    print(f"\n{len(rejected)} record(s) could not be loaded and are NOT "
          f"included in anything below:", file=sys.stderr)
    for reason in rejected:
        print(f"  x {reason}", file=sys.stderr)
    print("  Re-run them; do not re-score them.\n", file=sys.stderr)


# --- detail -----------------------------------------------------------------

def show(r: RunRecord) -> None:
    print("=" * 72)
    print(f"{r.task_id}  ·  {r.model}  ·  {r.name}")
    print("=" * 72)
    cost = f"${r.cost_usd}" if r.cost_known else "$UNKNOWN (no usage reported)"
    print(f"outcome:    {r.outcome}")
    print(f"config:     {r.config_version}   "
          f"caps {r.max_iterations} iters / ${r.max_cost_usd:.2f} / "
          f"{r.max_wall_seconds:.0f}s")
    print(f"iterations: {r.iterations}   cost {cost}   {r.wall_seconds}s")
    print(f"tests:      {r.tests_passed} passed, {r.tests_failed} failed, "
          f"{r.tests_errored} errored, {r.tests_xfailed} xfailed "
          f"[verdict: {r.verdict_status}]")
    if r.verdict_reason:
        print(f"            {r.verdict_reason}")
    print(f"files:      {r.changed_files or '(none)'}")
    if r.give_up_reason:
        print(f"gave up:    {r.give_up_reason}")
    if r.harness_error:
        print(f"HARNESS ERROR: {r.harness_error}")

    s = _score(r)
    if s is None:
        print("\n(not scored)")
    elif not s.valid:
        print(f"\nUNSCOREABLE — {s.unscoreable_reason}")
    else:
        print(f"\nfixed:      {s.fixed} of {len(r.before_failed_ids)}"
              f"   newly broken: {s.newly_broken}"
              f"   newly suppressed: {s.newly_suppressed}")
        print(f"progress:   {s.progress:.0%}   "
              f"credible: {s.credible_progress:.0%}")
        print(f"root causes: {s.clusters_fixed}/{s.clusters_total} fixed "
              f"({s.cluster_progress:.0%})")

    if s is not None:
        if s.cheats:
            by_kind = Counter(f.kind for f in s.cheats)
            print(f"\nCHEATS ({len(s.cheats)}) — progress does not count:")
            for kind, n in by_kind.most_common():
                example = next(f for f in s.cheats if f.kind == kind)
                suffix = f"  (x{n})" if n > 1 else ""
                print(f"  ! [{kind}] {example}{suffix}")
        if s.warnings:
            print("\nnotes:")
            for f in s.warnings[:10]:
                print(f"  ? [{f.kind}] {f}")

    if r.hallucinated_tools:
        print(f"\nhallucinated tools: {', '.join(r.hallucinated_tools)}")

    if r.tool_calls:
        print("\ntool calls:")
        for name, n in sorted(r.tool_calls.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {name}")
        explore = sum(r.tool_calls.get(k, 0) for k in EXPLORE)
        total = sum(r.tool_calls.values())
        if total:
            print(f"  explore {explore / total:.0%}   "
                  f"writes {r.tool_calls.get('write_file', 0)}   "
                  f"test runs {r.tool_calls.get('run_tests', 0)}")

    print("\nsequence:")
    step = 0
    first_write = None
    read_library = None
    for m in r.trajectory:
        names = [c["name"] for c in m.get("tool_calls", [])]
        if not names:
            continue
        step += 1
        if "write_file" in names and first_write is None:
            first_write = step
        if read_library is None and (
                "read_package_file" in names or "package_source" in names):
            read_library = step
        print(f"  {step:>3}. {', '.join(names)}")

    print()
    if first_write:
        print(f"first write at step {first_write}")
    else:
        print("NEVER WROTE ANYTHING — explored until the budget ran out.")
    if read_library and first_write:
        verb = "before" if read_library < first_write else "after"
        print(f"read library source at step {read_library} — {verb} editing")
    elif not read_library:
        print("never opened the library source — any API change was guessed")

    if r.diff:
        print(f"\ndiff ({len(r.diff.splitlines())} lines):")
        print("\n".join(r.diff.splitlines()[:40]))


def _must_score(r: RunRecord) -> Score:
    """`_score` where the caller already filtered to scored records.

    None here means that filter broke, not that the run is unscoreable.
    Raise at the cause rather than three frames later.
    """
    score = _score(r)
    if score is None:
        raise AssertionError(
            f"{r.name} reached a scored-only path without a score")
    return score


# --- compare ----------------------------------------------------------------

def _row(r: RunRecord) -> tuple:
    s = _score(r)
    total = sum(r.tool_calls.values())
    explore = (sum(r.tool_calls.get(k, 0) for k in EXPLORE) / total
               if total else 0.0)

    cheats: str
    if s is None:
        prog = credible = "-"
        cheats = "-"
    elif not s.valid:
        prog = credible = "VOID"
        cheats = str(len(s.cheats))
    elif s.cheats:
        # A cheating run has no meaningful progress number, and printing
        # one invites it to be read anyway.
        prog = f"{s.progress:.0%}"
        credible = "CHEAT"
        cheats = str(len(s.cheats))
    else:
        prog = f"{s.progress:.0%}"
        credible = f"{s.credible_progress:.0%}"
        cheats = "0"

    return (
        r.name,
        r.config_version,
        r.outcome,
        r.iterations,
        f"${r.cost_usd}" if r.cost_known else "?",
        f"{r.tests_passed}/{r.tests_failed}",
        prog,
        credible,
        cheats,
        r.tool_calls.get("write_file", 0),
        f"{explore:.0%}",
    )


def compare(records: list[RunRecord]) -> None:
    rows = [_row(r) for r in records]
    hdr = ("run", "config", "outcome", "iter", "cost", "pass/fail",
           "prog", "credible", "cheats", "writes", "explore")
    widths = [max(len(str(r[i])) for r in [hdr, *rows]) for i in range(len(hdr))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*hdr))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    configs = {r.config_version for r in records}
    if len(configs) > 1:
        print(f"\n{len(configs)} configurations above. Rows from different "
              f"configs are not comparable — see --summary, which keeps "
              f"them apart.")


# --- summary ----------------------------------------------------------------

def summary_for(group: list[RunRecord]) -> None:
    n = len(group)
    scores = {r.name: _score(r) for r in group}

    unscoreable = [r for r in group if ((s := scores[r.name]) is None
                                        or not s.valid)]
    scoreable = [r for r in group if r not in unscoreable]
    cheated = [r for r in scoreable if _must_score(r).cheats]
    clean = [r for r in scoreable if not _must_score(r).cheats]
    solved = [r for r in scoreable if r.outcome == "solved"]
    solved_dirty = [r for r in scoreable if r.outcome == "solved_with_cheats"]

    print(f"runs:          {n}   scoreable: {len(scoreable)}   "
          f"unscoreable: {len(unscoreable)}")
    if not scoreable:
        print("  nothing here can be scored.")
        return

    d = len(scoreable)
    print(f"  {claim(len(solved), d, 'solved (clean)')}")
    if solved_dirty:
        print(f"  {claim(len(solved_dirty), d, 'green but cheating')}")
    print(f"  {claim(len(cheated), d, 'cheat rate')}")

    if clean:
        progs = [_must_score(r).credible_progress for r in clean]
        print(f"\ncredible progress over the {len(clean)} clean runs:")
        print(f"  {describe(progs)}")
        if len(progs) > 1:
            low, high = bootstrap_mean_ci(progs)
            print(f"  bootstrap 95% CI for the mean: "
                  f"[{low:.0%}, {high:.0%}]")
            if len(progs) < 10:
                print("  At this n the resamples only contain the values "
                      "listed above, so that\n  interval describes THEM, "
                      "not a population. Quote the values.")

        # Excluding cheating runs from this mean is defensible — a cheating
        # run's progress figure measures something else — but it is also
        # upward-biased, because cheating correlates with reaching green.
        # Both stored cheating runs in the s2 config hit 115/0. So the
        # counterfactual is printed too, and the reader decides.
        if cheated:
            with_zeros = progs + [0.0] * len(cheated)
            mean_all = sum(with_zeros) / len(with_zeros)
            mean_clean = sum(progs) / len(progs)
            print(f"\n  the {len(cheated)} cheating run(s) are excluded above. "
                  f"Scored as 0 instead, the mean is "
                  f"{mean_all:.0%} rather than {mean_clean:.0%}.")
            print("  Excluding them is defensible (their progress measures a "
                  "different thing)\n  but biased upward (cheating "
                  "correlates with reaching green). Report both.")

        clusters = [(_must_score(r).clusters_fixed,
                     _must_score(r).clusters_total) for r in clean
                    if _must_score(r).clusters_total]
        if clusters:
            fixed = sum(c[0] for c in clusters)
            total = sum(c[1] for c in clusters)
            print(f"\nroot causes fixed: {fixed}/{total} across clean runs")
            print("  This is the honest denominator. The per-test figure "
                  "above counts 55 tests\n  that share two root causes, so "
                  "its effective sample size is about two.")

    costs = [r.cost_usd for r in group if r.cost_known]
    if costs:
        unknown = n - len(costs)
        note = f" ({unknown} run(s) reported no usage)" if unknown else ""
        print(f"\ncost: ${sum(costs):.2f} total, "
              f"${sum(costs) / len(costs):.3f} per run{note}")

    if unscoreable:
        print("\nunscoreable runs (excluded from every figure above):")
        for r in unscoreable:
            s = scores[r.name]
            why = s.unscoreable_reason if s else "never scored"
            print(f"  - {r.name}: {why}")

    if cheated:
        print("\ncheating runs:")
        for r in cheated:
            kinds = sorted({f.kind for f in _must_score(r).cheats})
            print(f"  - {r.name}: {', '.join(kinds)}")


def summary(records: list[RunRecord]) -> None:
    """Aggregate, one block per (task, configuration).

    Never across them. Aggregating across configs would average a model
    that had one tool set together with one that had another; that is not
    a mean, it is a mixture, and it is the standard way a benchmark stops
    meaning anything.
    """
    groups = group_by_config(records)
    for i, (_key, group) in enumerate(sorted(groups.items())):
        if i:
            print()
        print("=" * 72)
        print(describe_config(group))
        print("=" * 72)
        summary_for(group)

    if len(groups) > 1:
        print(f"\n{len(groups)} configurations reported separately. There is "
              f"no combined number,\nand producing one by pooling them "
              f"would not be a mean.")


# --- cli --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--compare", action="store_true", help="one row per run")
    p.add_argument("--summary", action="store_true",
                   help="aggregate stats per configuration, with intervals")
    p.add_argument("--include-archive", action="store_true",
                   help="also read runs/archive/, which holds records that "
                        "no longer load; they will all be reported as "
                        "rejections, which is the point of looking")
    args = p.parse_args()

    paths = list(args.paths)
    if args.include_archive:
        paths += [str(q) for q in sorted(ARCHIVE_DIR.glob("*.json"))]
    else:
        # A shell glob of runs/ does not descend into runs/archive/, but an
        # explicit runs/**/*.json does. Dropping archived records here keeps
        # the default output about the current schema without ever making
        # a rejection invisible — --include-archive shows every one.
        paths = [q for q in paths if ARCHIVE_DIR not in Path(q).parents]

    records, rejected = load_all(paths)
    _report_rejections(rejected)

    if not records:
        print("no loadable records", file=sys.stderr)
        return 1

    if args.summary:
        summary(records)
    elif args.compare:
        compare(records)
    else:
        for r in records:
            show(r)
            print()

    return 2 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
