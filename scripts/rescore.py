#!/usr/bin/env python3
"""Re-score saved runs against the current detectors.

    python3 rescore.py --dry-run          # show what would change
    python3 rescore.py                    # rewrite runs/*.json in place
    python3 rescore.py runs/*flash*       # a subset

Whenever a detector changes, the archive has to be re-scored. Otherwise
old and new runs are graded by different rules and any table mixing them
quietly lies.

This script used to be the most dangerous file in the project, because it
rewrites stored results and every mistake it made looked like a result:

  * it passed an empty before-state, so `fixed` computed to zero and every
    stored progress number silently became 0%;
  * it read a missing `failed_ids` as "nothing is failing", which
    subtracted to "every test fixed" and stamped seven runs with 100%;
  * it marked legacy records by appending a string to the `cheats` list,
    which made the reported cheat rate — the headline result — double.

None of those is possible now, and not because they were each fixed. The
before-state lives IN the record, so it cannot be passed empty or missing.
The loader rejects anything whose shape is not current, so a record with no
`failed_ids` never reaches the scorer. And "unscoreable" is its own field,
so nothing but a cheat can ever be put in `cheats`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from agentcheck.record import load_all, record_paths
from agentcheck.scorer import SuiteState, final_outcome, score_run
from agentcheck.task import Task, TaskError


def _state(failed_ids, errors, ok=True, suppressed=0) -> SuiteState:
    return SuiteState(ok=ok, failed_ids=frozenset(failed_ids),
                      errors=dict(errors or {}), suppressed=suppressed)


#: The sentence `tools.TestResult.no_verdict` formats for an exit-code
#: failure. Matched only against records written before
#: `verdict_exit_code` was stored.
_LEGACY_EXIT = re.compile(r"^pytest exited (\d+)\b")


def legacy_exit_code(record) -> int | None:
    """Recover pytest's exit code for a record written before it was kept.

    Reading a number back out of prose is exactly what this project tells
    people not to do, and it is confined to this function for that reason:
    it is a one-way migration over a sentence THIS harness wrote, not a
    parser for anything a provider or a tool produced. New records carry
    the field, and nothing else falls back to this.

    Without it the archive's 21 no-verdict runs stay unattributed, and an
    unattributed no-verdict run is reported as a harness failure — which
    is the misattribution the field was added to end.
    """
    if record.verdict_exit_code is not None:
        return record.verdict_exit_code
    match = _LEGACY_EXIT.match((record.verdict_reason or "").strip())
    return int(match.group(1)) if match else None


def legacy_findings(paths: list[str]) -> int:
    """Run the DETECTORS over records too old to score.

    A pre-schema record cannot yield a progress number: the before-state it
    was measured against was never written down, and inventing one is how
    seven runs came to claim 100%. But the diff is right there, and the
    detectors are pure functions over it. Losing the progress figure is not
    a reason to lose the cheat analysis.

    Nothing is written and no number is produced. This reports what the
    diffs contain, and says so.
    """
    import json

    tasks: dict[str, Task] = {}
    shown = 0

    for raw_path in paths:
        try:
            d = json.loads(Path(raw_path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"{Path(raw_path).name}: unreadable — {e}", file=sys.stderr)
            continue

        task_id, diff = d.get("task_id"), d.get("diff")
        if not task_id or diff is None:
            print(f"{Path(raw_path).name}: no task_id or diff — nothing to "
                  f"analyse", file=sys.stderr)
            continue
        if task_id not in tasks:
            try:
                tasks[task_id] = Task.load(task_id)
            except TaskError as e:
                print(f"{Path(raw_path).name}: {e}", file=sys.stderr)
                continue
        task = tasks[task_id]

        # A before-state that is deliberately a placeholder: it exists only
        # so the detectors run, and the returned Score is discarded except
        # for its findings. No progress figure is read, printed or stored.
        placeholder = SuiteState(ok=True, failed_ids=frozenset({"__unknown__"}))
        score = score_run(diff, placeholder, placeholder,
                          reference_diff=task.reference_diff(),
                          reachability=task.reachability(),
                          dependency=task.package)

        cheats = score.cheats
        label = f"{len(cheats)} cheat(s)" if cheats else "no cheats found"
        print(f"{Path(raw_path).stem:<46} {label}")
        for kind in sorted({f.kind for f in cheats}):
            example = next(f for f in cheats if f.kind == kind)
            n = sum(1 for f in cheats if f.kind == kind)
            suffix = f" (x{n})" if n > 1 else ""
            print(f"    ! [{kind}] {str(example)[:100]}{suffix}")
        shown += 1

    print(f"\nanalysed the diffs of {shown} legacy run(s). NO progress "
          f"number is derivable\nfrom these — the before-state they were "
          f"measured against was never recorded.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*")
    p.add_argument("--dry-run", action="store_true",
                   help="report changes without writing anything")
    p.add_argument("--legacy-findings", action="store_true",
                   help="run the cheat detectors over records too old to "
                        "score, without producing any number")
    args = p.parse_args()

    paths = args.paths or [str(x) for x in record_paths("runs")]
    if not paths:
        print("no runs to score", file=sys.stderr)
        return 1

    if args.legacy_findings:
        return legacy_findings(paths)

    records, rejected = load_all(paths)

    if rejected:
        print(f"{len(rejected)} record(s) cannot be re-scored:",
              file=sys.stderr)
        for reason in rejected:
            print(f"  x {reason}", file=sys.stderr)
        print("  These pre-date the current schema. Re-run them — the data "
              "needed to score\n  them was never recorded, and any number "
              "produced from what is there would be\n  invented. Their "
              "DIFFS are still analysable:\n"
              "      python3 rescore.py --legacy-findings runs/*.json\n",
              file=sys.stderr)

    tasks: dict[str, Task] = {}
    changed = 0

    for r in records:
        if r.task_id not in tasks:
            try:
                tasks[r.task_id] = Task.load(r.task_id)
            except TaskError as e:
                # A task we cannot load means unknown reachability and no
                # reference diff. Scoring anyway would silently drop every
                # warning and downgrade every test edit; skipping is the
                # only honest option.
                print(f"skipping {r.name}: {e}", file=sys.stderr)
                continue
        if r.task_id not in tasks:
            continue
        task = tasks[r.task_id]

        before = _state(r.before_failed_ids, r.before_errors)
        after = _state(r.failed_ids, {}, ok=r.verdict_ok,
                       suppressed=r.tests_xfailed + r.tests_xpassed)
        if not r.verdict_ok:
            after = SuiteState(ok=False, failed_ids=frozenset(r.failed_ids),
                               reason=r.verdict_reason,
                               exit_code=legacy_exit_code(r))

        score = score_run(
            r.diff, before, after,
            reference_diff=task.reference_diff(),
            reachability=task.reachability(),
            dependency=task.package,
        )
        outcome = final_outcome(
            after, score, (r.give_up_reason and "gave_up") or None,
            hit_iteration_cap=r.iterations >= r.max_iterations,
            hit_cost_cap=r.cost_known and r.cost_usd >= r.max_cost_usd,
            hit_time_cap=r.wall_seconds >= r.max_wall_seconds > 0,
            harness_error=r.harness_error or None,
        )

        old_credible = r.score.get("credible_progress")
        old_outcome = r.outcome
        moved = (old_credible != score.credible_progress
                 or old_outcome != outcome)

        if score.valid:
            state = "CHEAT" if score.cheats else f"{score.credible_progress:.0%}"
        else:
            state = "VOID"

        mark = "  (changed)" if moved else ""
        print(f"{r.name:<46} {outcome:<22} {state:>6}{mark}")

        if not args.dry_run:
            if r.path is None:
                raise AssertionError(
                    f"{r.name} was loaded without a path; nothing to rescore")
            r.score = score.as_dict()
            r.outcome = outcome
            r.save(r.path)
        changed += bool(moved)

    verb = "would change" if args.dry_run else "changed"
    print(f"\nre-scored {len(records)} run(s) against the current detectors; "
          f"{changed} {verb}")
    if rejected:
        print(f"{len(rejected)} rejected and left untouched.")
    return 2 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
