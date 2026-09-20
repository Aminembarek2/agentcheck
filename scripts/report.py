#!/usr/bin/env python3
"""Generate RESULTS.md from the run records. Nothing in it is typed by hand.

    .venv/bin/python scripts/report.py                 # write RESULTS.md
    .venv/bin/python scripts/report.py --check         # fail if it has drifted

A number a human transcribes into prose is a number that goes stale
silently, which is the phase-1 bug class wearing a different hat: the
document keeps saying 44% long after the archive says something else, and
nothing crashes. So the document is emitted from the records, committed,
and checked in CI the way `ruff format --check` is checked.

What this does NOT do is compute anything of its own. Every figure comes
from `record.py`, `scorer.py` and `stats.py`, which are the same functions
`inspect_run.py` prints. A second implementation of "the cheat rate" is a
second number free to disagree with the first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck import figures
from agentcheck.calibration import (
    LABEL_SCHEMA_VERSION,
    REGISTERED_JUDGES,
    SKIP,
    agree,
    collapsed,
    retest_issues,
    retest_opens,
    retest_sample,
    threshold_verdict,
)
from agentcheck.models import family_of
from agentcheck.record import (
    ConfigFingerprint,
    RunRecord,
    load_all,
    record_paths,
)
from agentcheck.scorer import Score
from agentcheck.stats import (
    bootstrap_mean_ci,
    describe,
    difference_claim,
    wilson,
)
from agentcheck.task import Task, TaskError

OUT = Path("RESULTS.md")

# The minimum sample size declared in README's completion criteria.
MIN_COMPARISON_RUNS = 8

#: README.md is hand-written prose, with one exception: the paragraph that
#: says how much has been measured so far. It said "one model, three tasks,
#: 15 loadable runs" while the archive held 119 — the exact failure
#: `report.py` exists to prevent, in the file a reader opens first. So that
#: sentence is generated between markers and checked in CI like the rest.
README = Path("README.md")

#: Tasks whose numbers are never pooled with the rest. A task with fewer
#: than three root causes has a near-binary outcome; averaging it in
#: raises or lowers the headline by an amount that depends on nothing but
#: how easy that one task is.
def is_smoke_test(task: Task) -> bool:
    return len(task.root_causes) < 2


def score_of(r: RunRecord) -> Score | None:
    if not r.score:
        return None
    try:
        return Score.from_dict(r.score)
    except (KeyError, TypeError):
        return None


def must_score(r: RunRecord) -> Score:
    """`score_of` where the caller has already established there is a score.

    Every call site below has filtered through `detectable()` or an
    explicit scoreable check first, so None here means that filter broke,
    not that this run is unscoreable. Raising says so; `score_of(r).cheats`
    would raise AttributeError three frames away from the cause, and a
    `# type: ignore` would hide the filter breaking at all.
    """
    score = score_of(r)
    if score is None:
        raise AssertionError(
            f"{r.name} reached a scored-only path without a score")
    return score


def cell_key(r: RunRecord) -> tuple[str, str]:
    return r.task_id, r.config_version


def pct(x: float) -> str:
    return f"{x:.0%}"


def interval(successes: int, n: int) -> str:
    ci = wilson(successes, n)
    return f"{successes}/{n} = {pct(ci.point)} [{pct(ci.low)}, {pct(ci.high)}]"


FIGURE_DIR = Path("docs/figures")


def _record_fingerprint(record: RunRecord) -> ConfigFingerprint | None:
    if record.harness_error or not record.config:
        return None
    try:
        return ConfigFingerprint(**record.config)
    except TypeError:
        return None  # Legacy records may carry only partial metadata.


def _configuration_label(record: RunRecord) -> str:
    return (f"{record.task_id.split('-', 1)[0]} · "
            f"{record.model.split('/')[-1]} · {record.max_iterations}it · "
            f"{record.config_version}")


def draw(records: list[RunRecord], out_dir: Path) -> set[str]:
    """Render whatever the data can support. Returns the stems written.

    Nothing is drawn from a placeholder. A rung with no runs is absent from
    the curve rather than plotted at zero — an unmeasured point drawn on an
    axis is indistinguishable from a measured failure, and that is the
    absent-is-never-zero rule applied to pixels.
    """
    drawn: set[str] = set()

    cells = cells_of(records)
    # A ladder varies only iterations, holding model and harness fixed.
    ladders: dict[tuple[str, str], dict[int, list[RunRecord]]] = defaultdict(
        lambda: defaultdict(list))
    for r in records:
        fingerprint = _record_fingerprint(r)
        if fingerprint is not None:
            settings = replace(fingerprint, max_iterations=0).digest()
            ladders[(r.task_id, settings)][r.max_iterations].append(r)

    series = {}
    used = {}
    for (task_id, settings), rungs in sorted(ladders.items()):
        if len(rungs) < 2:
            continue
        first = rungs[min(rungs)][0]
        label = (f"{task_id.split('-', 1)[0]} · "
                 f"{first.model.split('/')[-1]} · {settings}")
        points = []
        for cap, group in sorted(rungs.items()):
            scored = [x for x in group if (s := score_of(x)) and s.valid]
            if not scored:
                continue
            solved = sum(1 for x in scored
                         if x.outcome == "solved" and not must_score(x).cheats)
            points.append((cap, solved, len(scored)))
        if len(points) >= 2:
            series[label] = points
            used[label] = [(r.max_iterations, r.iterations)
                           for group in rungs.values() for r in group]
    if series and len(series) <= figures.MAX_SERIES:
        figures.ladder(series, out_dir)
        drawn.add("ladder")

    # Single-configuration panels also include pilots and legacy records.
    outcome_rows = [(_configuration_label(group[0]), [r.outcome for r in group])
                    for _, group in sorted(cells.items())]
    if outcome_rows:
        figures.outcomes(outcome_rows, out_dir)
        drawn.add("outcomes")

    # Where the budget went: reads against writes, per configuration.
    EXPLORE = {"list_files", "read_file", "search_code", "package_version",
               "package_source", "read_package_file"}
    effort_rows = []
    for _, group in sorted(cells.items()):
        pairs = []
        for r in group:
            tc = r.tool_calls or {}
            pairs.append((sum(tc.get(k, 0) for k in EXPLORE),
                          tc.get("write_file", 0)))
        effort_rows.append((_configuration_label(group[0]), pairs))
    if effort_rows:
        figures.effort(effort_rows, out_dir)
        drawn.add("effort")

    # Was the cap ever binding? A validity check on the ladder design.
    if used and len(used) <= figures.MAX_SERIES:
        figures.budget(used, out_dir)
        drawn.add("budget")

    # Every run as a dot, grouped by configuration.
    groups: dict[str, list[float]] = {}
    for _, group in sorted(cells.items()):
        scored = [r for r in group if (s := score_of(r)) and s.valid]
        if not scored:
            continue
        label = _configuration_label(group[0])
        groups[label] = [must_score(r).credible_progress for r in scored]
    if groups:
        figures.runs(groups, out_dir)
        drawn.add("runs")

    # Cheat rate per configuration, with intervals.
    rows = []
    for _, group in sorted(cells.items()):
        # The same population as the table in section 2: every run that
        # produced a patch. A figure drawn on a different denominator from
        # the table beside it is two answers to one question.
        patched = detectable(group)
        if not patched:
            continue
        cheated = sum(1 for r in patched if must_score(r).cheats)
        rows.append((_configuration_label(group[0]),
                     cheated, len(patched)))
    if rows:
        figures.rates(rows, out_dir)
        drawn.add("rates")

    return drawn


def detectable(rs: list[RunRecord]) -> list[RunRecord]:
    """Runs where cheating COULD be detected — the honest cheat denominator.

    Not the same as runs that produced a test verdict, and the difference
    is not small. Cheat detection reads the diff; it does not need the
    suite to have run. Excluding no-verdict runs was right for progress —
    there is no progress figure without a verdict — and got applied to the
    cheat rate by inheritance, where it removed 18 runs that had deleted
    989 tests between them from the denominator AND the numerator. The
    reported rate was 9%; over every run that produced a patch it is 23%.
    That is the project's own headline finding, understated 3x by its own
    reporting code.

    A run with no patch is excluded: an agent that changed nothing cannot
    have cheated, and counting it would deflate the rate in the other
    direction. How many runs produced nothing is reported separately.
    """
    return [r for r in rs if score_of(r) and r.diff.strip()]


def route_of(r: RunRecord) -> str:
    """The provider route a run went through, as the RECORD states it.

    A missing field is not "direct". Records written before
    `provider_route` existed do not say which endpoint served them, and
    filling that in from today's registry would put a claim in the
    document that nothing in the run verified.
    """
    route = (r.config or {}).get("provider_route")
    if route is None:
        return "not recorded"
    return route or "direct"


def caps_of(rs: list[RunRecord]) -> str:
    return ", ".join(str(c) for c in sorted({r.max_iterations for r in rs}))


def unmatched(a: list[RunRecord], b: list[RunRecord]) -> str:
    """Name what else differs between two arms, or say nothing.

    A difference between two arms that also differ in iteration budget is
    a difference between two experiments, not between two models. The
    interval is still worth printing — it bounds the size of whatever is
    there — but printing it beside a heading that names only one of the
    two varied factors is how a confound becomes a finding.
    """
    ca, cb = caps_of(a), caps_of(b)
    if ca == cb:
        return ""
    return (f". These arms are **not matched**: iteration caps {{{ca}}} "
            f"against {{{cb}}}. Budget varies with the arm, so the "
            f"difference belongs to neither on its own.")


def matched_configurations(
    a: list[RunRecord], b: list[RunRecord],
) -> list[tuple[str, list[RunRecord], list[RunRecord]]]:
    """Match a task and every fingerprint field except the model ID.

    Records without a full configuration remain in descriptive totals,
    but cannot establish a controlled comparison.
    """
    arms = []
    for records in (a, b):
        groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
        for record in records:
            fingerprint = _record_fingerprint(record)
            if fingerprint is None:
                continue
            key = (record.task_id,
                   replace(fingerprint, model_id="comparison").digest())
            groups[key].append(record)
        arms.append(groups)
    left, right = arms
    return [(key[0], left[key], right[key])
            for key in sorted(left.keys() & right.keys())]


def cells_of(records: list[RunRecord]) -> dict[tuple[str, str],
                                               list[RunRecord]]:
    out: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        out[cell_key(r)].append(r)
    return out


def render(records: list[RunRecord], rejected: list[str],
           manifests: list[dict], drawn: set[str] | None = None) -> str:
    out: list[str] = []
    add = out.append

    add("# Results")
    add("")
    add(f"Generated by `scripts/report.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d} from "
        f"{len(records)} run record(s). **Do not edit by hand** — "
        f"`report.py --check` fails if this file and the records disagree.")
    add("")
    add("Every figure below is traceable to a file in `runs/`. Where a "
        "number is missing, it is missing because the measurement did not "
        "happen, not because it was inconvenient.")
    add("")

    # --- what was measured --------------------------------------------------
    tasks: dict[str, Task] = {}
    #: Tasks that have runs but no loadable definition. They still appear
    #: per-configuration below, so they must appear in the table too —
    #: silently omitting a row would show results for a task with no
    #: declared contamination risk and no root-cause count, and a reader
    #: scanning the table would not know it was missing.
    undefined: list[str] = []
    for r in records:
        if r.task_id in tasks or r.task_id in undefined:
            continue
        try:
            tasks[r.task_id] = Task.load(r.task_id)
        except TaskError:
            undefined.append(r.task_id)

    cells: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        cells[cell_key(r)].append(r)

    add("## 1. What was measured")
    add("")
    # Models are counted by MODEL, not by model id. `deepseek-v4-flash`
    # and `deepseek/deepseek-v4-flash-0731` are one model on two routes,
    # and "2 model(s)" here would read as the README's two-model
    # criterion having been met by a change of endpoint.
    ids = {r.model for r in records}
    model_families = {family_of(m) for m in ids}
    variants = ("" if len(ids) == len(model_families) else
                f", run under {len(ids)} model id(s) — route or snapshot "
                f"variants of the same model, listed in section 2")
    add(f"- {len(cells)} configuration(s), "
        f"{len({r.task_id for r in records})} task(s), "
        f"{len(model_families)} model(s){variants}")
    known = [r.cost_usd for r in records if r.cost_known]
    if known:
        add(f"- total stored cost estimate: **${sum(known):.2f}** "
            f"across {len(known)} run(s) with reported usage"
            + (f"; {len(records) - len(known)} run(s) reported none"
               if len(known) < len(records) else ""))
    if manifests:
        add(f"- {len(manifests)} sweep manifest(s) in `runs/sweeps/`, "
            f"recording image digests, caps and the harness version")
    add("")
    add("Costs below are stored registry estimates, not invoices or "
        "forecasts. Historical direct-provider Flash estimates use stale "
        "rates; OpenRouter Flash rates are not endpoint maxima. "
        "See [pricing limitations](docs/threats-to-validity.md#internal-validity).")
    add("")

    add("| task | dependency | root causes | contamination risk | n |")
    add("|---|---|---|---|---|")
    for task_id in sorted(tasks):
        t = tasks[task_id]
        n = sum(1 for r in records if r.task_id == task_id)
        risk = t.contamination_risk or "**unassessed**"
        add(f"| `{task_id}` | {t.package} {t.from_version} → "
            f"{t.to_version} | {len(t.root_causes)} | {risk} | {n} |")
    for task_id in sorted(undefined):
        n = sum(1 for r in records if r.task_id == task_id)
        add(f"| `{task_id}` | **definition not loadable** | ? | "
            f"**unassessed** | {n} |")
    add("")
    if undefined:
        add(f"{len(undefined)} task(s) have run records but no readable "
            f"`task.yaml`. Their runs appear in section 3 and are excluded "
            f"from every pooled figure — without a root-cause count there "
            f"is no honest denominator for them.")
        add("")
    # The probe is what turns the declaration into something falsifiable,
    # so the sentence pointing at it must depend on the probe having been
    # run. Linking a file that does not exist claims a check happened.
    probe = Path("docs/contamination.md")
    add("Contamination risk is a declared judgement, not a measurement. "
        + (f"[`{probe}`]({probe}) records the probe that tests these "
           f"declarations: each model is asked, with no repository access, "
           f"to name the files the maintainer's PR touched."
           if probe.exists() else
           "**The probe that would test it has not been run** "
           "(`scripts/contamination_probe.py`), so every risk level above "
           "is an argument from the age and popularity of the migration "
           "and nothing more.")
        + " Tasks at different risk levels are never averaged together "
          "without saying so.")
    add("")

    if rejected:
        add(f"{len(rejected)} record(s) in the archive do not load under the "
            f"current schema and are excluded from everything above and "
            f"below. They are not deleted: see "
            f"[`runs/archive/MANIFEST.md`](runs/archive/MANIFEST.md).")
        add("")

    # --- the headline -------------------------------------------------------
    drawn = drawn or set()
    if "ladder" in drawn:
        add("### Cost to solve")
        add("")
        add(figures.picture(
            "ladder",
            "Runs solved cleanly against iteration budget, with 95% Wilson "
            "intervals, one line per task"))
        add("")
        add("A flat, high line means the budget was never what was binding "
            "and the task measures nothing about capability. A knee means "
            "it was, and where the knee sits is task difficulty in a unit "
            "that compares across tasks.")
        add("")

    if "outcomes" in drawn:
        add("### How runs ended")
        add("")
        add(figures.picture(
            "outcomes",
            "Share of runs by how they ended, per task and iteration "
            "budget, as a stacked bar"))
        add("")
        add("The solve-rate curve collapses every way a run can end into "
            "one number, and the ways differ. A squeezed budget turning "
            "`solved` into `hit a cap` says the budget was binding — a "
            "statement about the task. The same squeeze turning it into "
            "`stopped short` says the agent quit with budget in hand — a "
            "statement about the model. `green but flagged` is never "
            "folded into `solved`.")
        add("")

    if "budget" in drawn:
        add("### Was the budget ever binding?")
        add("")
        add(figures.picture(
            "budget",
            "Iterations used against the iteration cap, one dot per run, "
            "with a diagonal marking runs that used their whole budget"))
        add("")
        add("Each series holds the model and other recorded settings fixed; "
            "the identifier hashes those settings without the iteration cap. "
            "Dots on the diagonal reached the iteration cap. Dots below it "
            "ended earlier, including cost limits, time limits and failures; "
            "they do not by themselves establish an unconstrained control.")
        add("")

    add("## 2. Detector-flag rate")
    add("")
    if "rates" in drawn:
        add(figures.picture(
            "rates",
            "Detector-flag rate per configuration with 95% Wilson intervals"))
        add("")

    add("A flag means at least one automated detector assigned cheat "
        "severity to the patch. Detector precision and recall on these "
        "runs have not been measured; the flag rate is neither a confirmed "
        "cheating rate nor a demonstrated lower bound. Wilson intervals "
        "describe sampling uncertainty, not detector error. Here, 'clean' "
        "means no detector flags, not human verification.")
    add("")

    # Smoke-test tasks are excluded from the pooled figure rather than
    # weighted down. A task with one root cause has a 0-or-1 outcome, so
    # including it moves the headline by an amount that depends on nothing
    # but how easy that one task is.
    measuring = {k: v for k, v in cells.items()
                 if k[0] in tasks and not is_smoke_test(tasks[k[0]])}
    excluded = sorted({k[0] for k in cells if k not in measuring})

    by_id: dict[str, list[RunRecord]] = defaultdict(list)
    for group in measuring.values():
        for r in group:
            by_id[r.model].append(r)

    # Model ids grouped by the MODEL behind them. `deepseek-v4-flash` and
    # `deepseek/deepseek-v4-flash-0731` are one model on two routes, and a
    # difference between those two arms is a fact about the route — put
    # under a "differences between models" heading it would read as a
    # model comparison this project has not run.
    by_family: dict[str, list[str]] = defaultdict(list)
    for mid in sorted(by_id):
        by_family[family_of(mid)].append(mid)

    if not by_id:
        add("No measuring task has runs yet.")
        add("")
    else:
        if excluded:
            add(f"Excluded from this pool: "
                f"{', '.join(f'`{t}`' for t in excluded)} — fewer than two "
                f"root causes, so progress is near-binary and pooling it "
                f"would move the headline by an amount that reflects only "
                f"how easy that task is. Reported in full in section 3.")
            add("")
        add("Pooled across measuring tasks, per model id. The denominator "
            "is **every run that produced a patch**, because that is the "
            "population in which cheating can be detected: the detectors "
            "read the diff, not the test suite. Runs that changed nothing "
            "are excluded — an agent that wrote no patch cannot have "
            "cheated — and counted below.")
        add("")
        add("| model | model id | route | patched | flagged | Wilson 95% |")
        add("|---|---|---|---|---|---|")
        rates: dict[str, tuple[int, int]] = {}
        for fam in sorted(by_family):
            for mid in by_family[fam]:
                rs = by_id[mid]
                routes = ", ".join(sorted({route_of(r) for r in rs}))
                patched = detectable(rs)
                cheated = [r for r in patched if must_score(r).cheats]
                if not patched:
                    add(f"| `{fam}` | `{mid}` | {routes} | 0 | — | "
                        f"no run produced a patch |")
                    continue
                rates[mid] = (len(cheated), len(patched))
                ci = wilson(len(cheated), len(patched))
                add(f"| `{fam}` | `{mid}` | {routes} | {len(patched)} | "
                    f"{len(cheated)} | {pct(ci.point)} [{pct(ci.low)}, "
                    f"{pct(ci.high)}] |")
        add("")

        # The same rate over the narrower population, stated rather than
        # chosen. Reporting only one of these would be picking a
        # denominator after seeing both numbers.
        pooled = [r for group in measuring.values() for r in group]
        patched_all = detectable(pooled)
        scoreable = [r for r in patched_all if must_score(r).valid]
        no_patch = len(pooled) - len(patched_all)
        if scoreable and len(scoreable) != len(patched_all):
            cheated_all = sum(1 for r in patched_all if must_score(r).cheats)
            cheated_scoreable = sum(1 for r in scoreable
                                    if must_score(r).cheats)
            add(f"Over the narrower population — runs that also produced a "
                f"test verdict — the rate is "
                f"{interval(cheated_scoreable, len(scoreable))} against "
                f"{interval(cheated_all, len(patched_all))} over every run "
                f"that produced a patch. The gap is the point: the runs "
                f"with no verdict are largely runs that deleted enough of "
                f"the suite to stop it collecting. Computing the flag "
                f"rate only where the suite still ran excludes those "
                f"patches from the measurement. "
                f"{no_patch} run(s) produced no patch at all and are in "
                f"neither figure.")
            add("")
        if not rates:
            add("No run in these configurations produced a patch, so there "
                "is nothing for the detectors to read and no flag rate to "
                "report. That is a fact about the runs, not a rate of 0%.")
            add("")

        measured = sorted(rates)
        cross = [(a, b) for i, a in enumerate(measured) for b in measured[i + 1:]
                 if family_of(a) != family_of(b)]
        within = [(a, b) for i, a in enumerate(measured) for b in measured[i + 1:]
                  if family_of(a) == family_of(b)]

        newcombe_note = (
            "Newcombe intervals on the difference of the two rates. Two "
            "overlapping Wilson intervals do not establish that there is "
            "no difference, and two that do not overlap claim more than "
            "the data supports — neither is the interval on the "
            "difference.")

        if cross:
            add("### Differences between models")
            add("")
            add("Comparisons below match the task and every recorded "
                "configuration field except model ID. Historical caps "
                "and prompt variants are not pooled. Each arm needs at "
                f"least {MIN_COMPARISON_RUNS} measured attempts; smaller "
                "samples remain pilots. " + newcombe_note)
            add("")
            for a, b in cross:
                matched = matched_configurations(by_id[a], by_id[b])
                if not matched:
                    add(f"- `{a}` vs `{b}`: no matched configurations "
                        f"with complete metadata{unmatched(by_id[a], by_id[b])}")
                for task_id, left, right in matched:
                    pa, pb = detectable(left), detectable(right)
                    label = (f"**{task_id}** (`{left[0].config_version}` vs "
                             f"`{right[0].config_version}`)")
                    if min(len(left), len(right)) < MIN_COMPARISON_RUNS:
                        add(f"- {label}: **pilot; comparison pending** "
                            f"({len(left)} vs {len(right)} measured attempts; "
                            f"need {MIN_COMPARISON_RUNS} per arm).")
                        continue
                    if not pa or not pb:
                        add(f"- {label}: no detector-flag comparison; at least "
                            "one arm produced no patches.")
                        continue
                    sa = sum(bool(must_score(r).cheats) for r in pa)
                    sb = sum(bool(must_score(r).cheats) for r in pb)
                    add(f"- {label}: "
                        f"{difference_claim(sa, len(pa), a, sb, len(pb), b)}")
            add("")
        elif rates:
            add("> Only one model has been run on the measuring tasks, so "
                "there is no comparison between models to make. "
                "The project's scope asks for at least two."
                + (" The rows above are that one model reached through "
                   "different routes or snapshot pins; they are not two "
                   "models." if within else ""))
            add("")

        if within:
            add("### The same model, reached two ways")
            add("")
            add("One model per pair below, under two ids: a different "
                "provider route, a different snapshot pin, or both. "
                "Nothing here is evidence about a second model. " +
                newcombe_note)
            add("")
            for a, b in within:
                sa, na = rates[a]
                sb, nb = rates[b]
                add(f"- {difference_claim(sa, na, a, sb, nb, b)}"
                    f"{unmatched(by_id[a], by_id[b])}")
            add("")

    # --- per configuration --------------------------------------------------
    add("## 3. Per configuration")
    add("")
    if "runs" in drawn:
        add(figures.picture(
            "runs",
            "Credible progress of every individual run, grouped by "
            "configuration, with the median marked"))
        add("")
        add("Every dot is one run. Deliberately not a bar of means: "
            "observed progress is bimodal, so a mean names a value no run "
            "took.")
        add("")
    add("Never pooled across configurations. A `config_version` covers the "
        "model, the prompt, the tool definitions, the caps, the test "
        "command and the image digest; two runs with different stamps "
        "answer different questions.")
    add("")

    for key in sorted(cells):
        group = cells[key]
        task_id, config = key
        first = group[0]
        task = tasks.get(task_id)
        add(f"### `{task_id}` · {first.model} · "
            f"{first.max_iterations} iters, ${first.max_cost_usd:.2f} · "
            f"`{config}`")
        add("")
        if task and is_smoke_test(task):
            add(f"> Smoke test, not a measurement. "
                f"{task.low_power_reason.strip()}")
            add("")

        scores = {r.name: score_of(r) for r in group}
        unscoreable = [r for r in group
                       if (s := scores[r.name]) is None or not s.valid]
        scoreable = [r for r in group if r not in unscoreable]
        broke = [r for r in group if r.outcome == "broke_suite"]
        add(f"- runs: {len(group)}   scoreable: {len(scoreable)}   "
            f"unscoreable: {len(unscoreable)}"
            + (f" (of which {len(broke)} because the agent broke the suite)"
               if broke else ""))

        # The cheat rate here uses the same population as section 2 — every
        # run that produced a patch — so a reader comparing a cell against
        # the pooled figure is comparing like with like. The solve rate
        # cannot: there is no "solved" without a verdict.
        patched = detectable(group)
        if patched:
            cheated_patched = [r for r in patched if must_score(r).cheats]
            add(f"- detector-flag rate (runs that produced a patch): "
                f"{interval(len(cheated_patched), len(patched))}")

        if scoreable:
            clean = [r for r in scoreable if not must_score(r).cheats]
            cheated = [r for r in scoreable if must_score(r).cheats]
            solved = [r for r in scoreable if r.outcome == "solved"]
            add(f"- solved (clean): {interval(len(solved), len(scoreable))}")
            add(f"- detector-flag rate (runs with a verdict): "
                f"{interval(len(cheated), len(scoreable))}")

            if clean:
                progs = [must_score(r).credible_progress for r in clean]
                add("")
                add(f"Credible progress over the {len(clean)} clean run(s): "
                    f"`{', '.join(pct(p) for p in sorted(progs))}`")
                add("")
                add(f"    {describe(progs)}")
                add("")
                if len(progs) > 1:
                    low, high = bootstrap_mean_ci(progs)
                    add(f"Bootstrap 95% CI for the mean: "
                        f"[{pct(low)}, {pct(high)}].")
                    if len(progs) < 10:
                        add("At this n the resamples contain only the values "
                            "listed above, so that interval describes "
                            "**them**, not a population. Quote the values.")
                    add("")
                if cheated:
                    with_zeros = progs + [0.0] * len(cheated)
                    add(f"The {len(cheated)} flagged run(s) are excluded "
                        f"from that mean. Scored as 0 instead it is "
                        f"{pct(sum(with_zeros) / len(with_zeros))} rather "
                        f"than {pct(sum(progs) / len(progs))}. Excluding "
                        f"flagged runs raises this mean relative to "
                        f"counting their credible progress as zero.")
                    add("")

                cl = [(must_score(r).clusters_fixed,
                       must_score(r).clusters_total) for r in clean
                      if must_score(r).clusters_total]
                if cl:
                    fixed = sum(c[0] for c in cl)
                    total = sum(c[1] for c in cl)
                    add(f"Root causes fixed: **{fixed}/{total}** across clean "
                        f"runs. This is the honest denominator; the per-test "
                        f"figure counts tests that move together.")
                    add("")

            for r in cheated:
                s = must_score(r)
                kinds = sorted({f.kind for f in s.cheats})
                add(f"- **flagged run** [`{r.name}`](runs/{r.name}.json): "
                    f"{', '.join(kinds)}")
            if cheated:
                add("")

        for r in unscoreable:
            s = scores[r.name]
            why = s.unscoreable_reason if s else "never scored"
            # Named by cause. "unscoreable" alone reads as the harness
            # having failed, which for these runs is the opposite of what
            # happened: the agent edited the repository until pytest could
            # not collect it.
            label = ("**broke the suite**"
                     if s and s.unscoreable_cause == "agent"
                     else "**unscoreable**")
            kinds = sorted({f.kind for f in s.cheats}) if s else []
            found = f" — detector flags: {', '.join(kinds)}" if kinds else ""
            add(f"- {label} [`{r.name}`](runs/{r.name}.json): {why}{found}")
        add("")

        costs = [r.cost_usd for r in group if r.cost_known]
        if costs:
            add(f"Stored cost estimate: ${sum(costs):.2f} total, "
                f"${sum(costs) / len(costs):.3f} per run.")
            add("")

    # --- harness failures ---------------------------------------------------
    add("## 4. Where the budget goes")
    add("")
    if "effort" in drawn:
        add(figures.picture(
            "effort",
            "Mean read and write tool calls per run, per iteration "
            "budget, with the count of runs that never wrote a file"))
        add("")

    explore_keys = {"list_files", "read_file", "search_code",
                    "package_version", "package_source", "read_package_file"}
    never_wrote = sum(1 for r in records
                      if not (r.tool_calls or {}).get("write_file", 0))
    empty_diff = sum(1 for r in records if not r.diff.strip())
    if records:
        add(f"Across every run in `runs/` — {len(records)} of them, "
            f"spanning {len({r.config_version for r in records})} "
            f"configurations — **{never_wrote} never called `write_file` "
            f"once**, and {empty_diff} left an empty diff. The median run "
            f"changed nothing at all.")
        add("")

    by_cap: dict[int, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_cap[r.max_iterations].append(r)
    if len(by_cap) > 1:
        add("| iteration budget | configs | mean reads | mean writes "
            "| never wrote |")
        add("|---|---|---|---|---|")
        for cap in sorted(by_cap):
            group = by_cap[cap]
            n = len(group)
            reads = sum(sum((r.tool_calls or {}).get(k, 0)
                            for k in explore_keys) for r in group) / n
            writes = sum((r.tool_calls or {}).get("write_file", 0)
                         for r in group) / n
            blank = sum(1 for r in group
                        if not (r.tool_calls or {}).get("write_file", 0))
            configs = len({r.config_version for r in group})
            add(f"| {cap} | {configs} | {reads:.1f} | {writes:.1f} "
                f"| {blank}/{n} |")
        add("")
        add("Rows are **not** pooled results — each spans the "
            "configurations named in the second column, and rows are not "
            "comparable to each other as measurements. They are shown "
            "together because the read count is the thing being read off, "
            "and it behaves the same way across every configuration "
            "present, including the pre-OpenRouter runs at a 150-iteration "
            "cap on a different model snapshot and route. A quantity that "
            "saturates at roughly the same place under settings that share "
            "nothing else is the more interesting observation, not a "
            "sloppier one — but no solve rate or progress figure is "
            "combined this way anywhere in this document.")
        add("")
        add("Reads saturate while writes keep climbing. The agent spends "
            "its first several dozen iterations reading and only then "
            "begins to edit, so a budget below that never reaches the "
            "editing phase at all.")
        add("")
        add("That changes what the solve-rate curve means. \"It needs more "
            "iterations\" is the wrong reading; **it needs to stop reading "
            "sooner** — which is a property of the prompt and the tool "
            "descriptions, not of the model. [`docs/findings.md`](docs/findings.md) "
            "§2 records a "
            "prompt change moving writes from `0,0,8,0` to `8,8,11,7` with "
            "everything else held constant, and "
            "[Harness-Bench](https://arxiv.org/html/2605.27922v1) "
            "documents 10–20 point swings on identical weights for the "
            "same reason.")
        add("")

    add("## 5. Harness failures")
    add("")
    add("Above the fold on purpose. The rate at which the measuring "
        "apparatus fails is part of the result, and a benchmark that does "
        "not report it is claiming a precision it has not demonstrated.")
    add("")
    add("Read the rate below as an **upper bound**. Some of these failures "
        "were caused by the harness cleaning up its own live containers — "
        "orphan detection decided ownership by container age until it was "
        "changed to read the owning process from the process table. "
        "Affected runs are marked unscoreable and re-run rather than "
        "counted, so no figure includes them, but runs predating the fix "
        "carry failures the measurement itself produced. See "
        "[`docs/threats-to-validity.md`](docs/threats-to-validity.md).")
    add("")
    n = len(records)
    if n:
        # Split by CAUSE. Every run without a verdict used to be counted
        # here, including the ones where the agent had deleted enough of
        # the suite that pytest could no longer collect it — an agent
        # result, filed as an apparatus failure. That overstates this
        # section and understates section 2.
        unattributed = [r for r in records
                        if (s := score_of(r)) is None
                        or (not s.valid and s.unscoreable_cause != "agent")]
        broke = [r for r in records
                 if (s := score_of(r)) and s.unscoreable_cause == "agent"]
        hallucinating = sum(1 for r in records if r.hallucinated_tools)
        errored = sum(1 for r in records if r.harness_error)
        for label, k in (("runs with no test verdict that the harness "
                          "cannot attribute to the agent", len(unattributed)),
                         ("runs where the model invented a tool",
                          hallucinating),
                         ("runs that hit a harness error", errored)):
            add(f"- {label}: {interval(k, n)}")
        add("")
        if broke:
            add(f"Not counted above, deliberately: **{len(broke)} run(s) "
                f"produced no verdict because the agent's own edits stopped "
                f"pytest collecting the suite** ({interval(len(broke), n)}). "
                f"That is a result about the agent, and it is reported as "
                f"the outcome `broke_suite` rather than as a failure of "
                f"the measurement. Attribution is by evidence and is "
                f"conservative: a run is charged to the agent only when "
                f"pytest failed in a way the repository's contents can "
                f"cause AND the agent changed the repository. A timed-out "
                f"suite is never attributed — an introduced hang and a "
                f"slow container look identical from here.")
            add("")
        invented = sorted({t for r in records for t in r.hallucinated_tools})
        if invented:
            add(f"Tools invented: {', '.join(f'`{t}`' for t in invented)}.")
            add("")

    # --- judge --------------------------------------------------------------
    add("## 6. Judge calibration")
    add("")
    add(calibration_block())
    add("")

    add("## 7. Limitations")
    add("")
    add("See [`docs/threats-to-validity.md`](docs/threats-to-validity.md). "
        "It is longer than this document and it is the part worth reading "
        "first.")
    add("")
    return "\n".join(out)


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def status_block(records: list[RunRecord]) -> str:
    """The two sentences of README.md that are measurements, not prose."""
    ids = {r.model for r in records}
    fams = {family_of(m) for m in ids}
    cells = cells_of(records)
    sizes = sorted(len(v) for v in cells.values())
    span = (f"n={sizes[0]}" if sizes[0] == sizes[-1]
            else f"n={sizes[0]}–{sizes[-1]}")
    variants = ("" if len(ids) == len(fams)
                else f" (under {len(ids)} route or snapshot ids)")
    labelled = _labelled_pairs()
    judge = ("no human labels exist yet, so no judge verdict appears "
             "anywhere" if not labelled else
             f"{labelled} pair(s) hand-labelled")
    return (f"Where it stands today: **{plural(len(fams), 'model')}"
            f"**{variants}, "
            f"**{plural(len({r.task_id for r in records}), 'task')}**, "
            f"**{plural(len(records), 'loadable run record')}** across "
            f"{plural(len(cells), 'configuration')}, {span} per "
            f"configuration. The judge: {judge}.")


def cost_block(records: list[RunRecord]) -> str:
    """Stored cost estimates and durations, without claiming billed costs.

    Median as well as mean: the distribution is skewed by a few long runs,
    and a reader planning a sweep budget wants the typical attempt and the
    worst one, not one number standing in for both.
    """
    costs = sorted(r.cost_usd for r in records if r.cost_known)
    walls = sorted(r.wall_seconds for r in records if r.wall_seconds)
    unknown = sum(1 for r in records if not r.cost_known)
    if not costs or not walls:
        return ("No run in `runs/` reports both a cost and a duration, so "
                "there is no per-attempt figure to give.")
    mid = len(costs) // 2
    return (f"Stored cost estimates are **${costs[mid]:.3f}** at the median "
            f"and ${sum(costs) / len(costs):.3f} on average. Attempts take "
            f"{walls[len(walls) // 2] / 60:.0f} minutes at the median "
            f"({walls[-1] / 60:.0f} at the longest), over "
            f"{plural(len(costs), 'run')} in `runs/`. The highest single "
            f"estimate is ${costs[-1]:.2f}. These are not verified bills "
            f"or future budget guarantees; see "
            f"[pricing limitations](docs/threats-to-validity.md#internal-validity)."
            + (f" {plural(unknown, 'run')} reported no token usage and are "
               f"not in that figure — their cost is unknown, not zero."
               if unknown else ""))


def _calibration_data(path: Path = Path("judge-labels.json")) -> dict | None:
    """Absent is a state; malformed or incompatible evidence is an error."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if data.get("version") != LABEL_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported labels schema")
    return data


def _labelled_pairs(path: Path = Path("judge-labels.json")) -> int:
    data = _calibration_data(path)
    return sum(bool(p["human"]) for p in data["pairs"]) if data else 0


def _headline_kappas(data: dict, usable: list) -> str:
    """The pre-registered headline for each judge, and what it licenses.

    Seed 0, matching `validate_judge report`, so the figure quoted here and
    the figure in the report are the same number and not two draws of the
    same bootstrap.
    """
    said, bands = [], []
    for alias in REGISTERED_JUDGES:
        pairs = [p for p in usable if alias in p["judge"]]
        if not pairs:
            continue
        human = collapsed([p["human"]["agent_relative"] for p in pairs])
        judge = collapsed([p["judge"][alias]["verdict"] for p in pairs])
        try:
            a = agree(human, judge, seed=0)
        except ValueError:
            said.append(f"`{alias}` produced no comparable binary verdicts")
            bands.append("inconclusive")
            continue
        band, _ = threshold_verdict(a.kappa)
        bands.append(band)
        said.append(f"`{alias}` kappa **{a.kappa:.2f}** "
                    f"[{a.kappa_interval.low:.2f}, {a.kappa_interval.high:.2f}] "
                    f"({band})")
    if not said:
        return "No judge has a comparable verdict set."
    # Lead with the conclusion when every judge reached the same one; a
    # reader should not have to compare two coefficients to learn that
    # nothing here licenses quoting a judge.
    lead = (f"**Every judge {bands[0]} the specified threshold.**"
            if len(bands) > 1 and len(set(bands)) == 1 and bands[0] == "failed"
            else "**Against the specified threshold:**")
    return lead + " " + "; ".join(said) + "."


def calibration_block() -> str:
    """Public progress without showing any labels, verdicts or agreement."""
    data = _calibration_data()
    protocol = "[docs/judge-protocol.md](docs/judge-protocol.md)"
    if data is None or not any(p["human"] for p in data["pairs"]):
        return (f"**Not yet run.** The protocol is specified in {protocol}, "
                "but no human labels exist. An unvalidated judge is a "
                "number generator; no judge verdict is reported.")
    labelled = [p for p in data["pairs"] if p["human"]]
    usable = [p for p in labelled if p["human"]["neutral"] != SKIP]
    counts = {a: sum(a in p["judge"] for p in usable)
              for a in REGISTERED_JUDGES}
    sampled = retest_sample(data)
    done = sum(p["relabel"] is not None for p, _ in sampled)
    complete = (len(labelled) == len(data["pairs"])
                and all(n == len(usable) for n in counts.values())
                and bool(sampled) and done == len(sampled))
    parts = [
        "**Calibration report available.**" if complete and
        Path("docs/judge-calibration.md").exists() else "**Calibration pending.**",
        f"{len(labelled)}/{len(data['pairs'])} pairs human-labelled "
        f"({len(labelled) - len(usable)} skipped).",
        "Judged: " + "; ".join(f"`{a}` {n}/{len(usable)}" for a, n in counts.items()) + ".",
        f"Blind retest: {done}/{len(sampled)} pairs recorded.",
    ]
    opens = retest_opens(data)
    if opens:
        parts.append(f"The remaining retest opens at {opens:%Y-%m-%d %H:%M:%S} UTC.")
    if complete and Path("docs/judge-calibration.md").exists():
        # State the outcome here, not only in the linked report. A section
        # that says "a report is available" makes a reader open a file to
        # learn whether the instrument works, and a failed calibration that
        # is only discoverable one click away reads as a hidden result.
        # Recomputed from the labels with the report's own seed rather than
        # parsed back out of the generated markdown.
        parts.append(_headline_kappas(data, usable))
        issues = retest_issues(data)
        if issues:
            parts.append("Protocol violations: " + "; ".join(issues) + ".")
        parts.append("Full agreement, uncertainty and confusion matrices are "
                     "in [the calibration report](docs/judge-calibration.md).")
    else:
        parts.append("Judge verdicts remain unpublished until the blind retest "
                     "and calibration report are complete.")
    parts.append(f"Protocol: {protocol}.")
    return " ".join(parts)


#: Every generated block in README.md, by marker name. Each is a sentence
#: made of numbers, and a number a human retypes is a number that goes
#: stale silently — "$0.45 per attempt" survived a change of provider,
#: model and cap, and was 18x the actual median by the time anyone read it
#: again.
README_BLOCKS = {
    "status": lambda records: status_block(records),
    "cost": lambda records: cost_block(records),
    "calibration": lambda records: calibration_block(),
}


def sync_readme(records: list[RunRecord], check: bool) -> int:
    """Write (or verify) every generated block in README.md. 0 on success."""
    if not README.exists():
        print(f"{README} not found — generated blocks not updated",
              file=sys.stderr)
        return 1
    text = wanted = README.read_text()
    for name, build in README_BLOCKS.items():
        begin, end = f"<!-- generated: {name} -->", f"<!-- /generated: {name} -->"
        if begin not in wanted or end not in wanted:
            print(f"{README} has no {begin} … {end} block. That sentence is "
                  f"generated; without the markers it would go stale "
                  f"unnoticed, which is what this script exists to prevent.",
                  file=sys.stderr)
            return 1
        head, rest = wanted.split(begin, 1)
        _, tail = rest.split(end, 1)
        wanted = f"{head}{begin}\n{build(records)}\n{end}{tail}"
    if check:
        if text != wanted:
            print(f"{README}'s generated blocks have drifted from the "
                  f"records. Regenerate them:\n"
                  f"    .venv/bin/python scripts/report.py", file=sys.stderr)
            return 1
        return 0
    if text != wanted:
        README.write_text(wanted)
        print(f"updated the generated blocks in {README}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="*", default=None,
                   help="run records (default: runs/*.json)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check", action="store_true",
                   help="exit non-zero if the committed file has drifted "
                        "from the records")
    args = p.parse_args()

    paths = args.paths or record_paths("runs")
    records, rejected = load_all(paths)
    if not records:
        print("no loadable records", file=sys.stderr)
        return 1

    manifests = []
    for m in sorted(Path("runs/sweeps").glob("*.json")):
        try:
            manifests.append(json.loads(m.read_text()))
        except json.JSONDecodeError:
            print(f"skipping unreadable manifest {m}", file=sys.stderr)

    try:
        drawn = draw(records, FIGURE_DIR)
    except ImportError:
        # matplotlib is a dev extra. The report must still generate without
        # it; a missing plotting library is not a reason to have no results.
        print("matplotlib not installed — figures skipped "
              '(pip install -e ".[dev]")', file=sys.stderr)
        drawn = set()

    text = render(records, rejected, manifests, drawn)

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist; run report.py", file=sys.stderr)
            return 1
        current = args.out.read_text()
        # The generation date changes daily and is not a result. Comparing
        # it would make this check fail every midnight for no reason.
        if _without_date(current) != _without_date(text):
            print(f"{args.out} has drifted from the records. Regenerate it:\n"
                  f"    .venv/bin/python scripts/report.py", file=sys.stderr)
            return 1
        if sync_readme(records, check=True):
            return 1
        print(f"{args.out} is up to date with {len(records)} record(s)")
        return 0

    args.out.write_text(text)
    print(f"wrote {args.out} from {len(records)} record(s)")
    return sync_readme(records, check=False)


def _without_date(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not line.startswith("Generated by "))


if __name__ == "__main__":
    raise SystemExit(main())
