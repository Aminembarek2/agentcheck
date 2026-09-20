#!/usr/bin/env python3
"""Run the agent against one task, score the result, and save the trace.

    python3 run_agent.py 002-databases
    python3 run_agent.py 002-databases --model ds-flash --label ds-06
    python3 run_agent.py 002-databases --model opus --iterations 100 --max-cost 5

Two numbers come out of every run: progress (how many failing tests now
pass) and credible progress (the same, but zero if the diff contains a
cheat). Reporting both is the point of the project.

Read a trace afterwards with:

    python3 inspect_run.py runs/<file>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentcheck import tracing
from agentcheck.agent import MAX_TOKENS, SYSTEM_PROMPT, Agent
from agentcheck.models import MODELS, build_model
from agentcheck.models import model_id as resolve_model_id
from agentcheck.record import ConfigFingerprint, RunRecord, holds_measurement
from agentcheck.sandbox import Sandbox, SandboxError, sweep_orphans
from agentcheck.scorer import SuiteState, final_outcome, score_run
from agentcheck.task import Task, TaskError
from agentcheck.tools import MAX_FILE_LINES, MAX_SEARCH_HITS, Tools

# --- reporting --------------------------------------------------------------

def _print_findings(score) -> None:
    if score.cheats:
        print("\nCHEATS DETECTED — progress does not count:")
        for f in score.cheats[:12]:
            print(f"  ! {f}")
        if len(score.cheats) > 12:
            print(f"  ! ... and {len(score.cheats) - 12} more")
    if score.warnings:
        print("\nnotes:")
        for f in score.warnings[:12]:
            print(f"  ? {f}")
        if len(score.warnings) > 12:
            print(f"  ? ... and {len(score.warnings) - 12} more")


# --- main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=Task.available())
    p.add_argument("--model", choices=sorted(MODELS), default="or-ds-flash",
                   help="default is the cheapest arm; a forgotten "
                        "--model should not cost 7x")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--max-cost", type=float, default=1.00,
                   help="estimated USD cap; the final call may exceed it")
    p.add_argument("--max-wall", type=float, default=1800.0,
                   help="hard wall-clock cap for this run, seconds")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--no-cache", action="store_true",
                   help="disable provider prompt caching (costs more)")
    p.add_argument("--label", default="run")
    p.add_argument("--announce-budget", action="store_true",
                   help="tell the model its remaining steps each turn. A "
                        "separate configuration, not an improvement to the "
                        "default: it changes behaviour and its runs are "
                        "never pooled with the control arm")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    try:
        task = Task.load(args.task)
    except TaskError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = Path(args.out_dir) / f"{task.id}-{args.model}-{args.label}.json"
    if holds_measurement(out):
        # Refuse only where writing would destroy a RESULT. A file that
        # merely exists is not one: a record from an older schema, or one
        # whose run the harness could not complete, holds no measurement
        # and is exactly what a re-run is meant to replace.
        #
        # Guarding on existence alone made this script veto the sweep's own
        # decisions — the sweep found 22 unusable records, scheduled them,
        # and got "already exists" back for every one.
        print(f"{out} already holds a completed run — pick another --label, "
              f"or delete it deliberately", file=sys.stderr)
        return 1
    if out.exists():
        print(f"NOTE: overwriting {out.name}, which holds no measurement "
              f"(unreadable, or the harness could not finish it)",
              file=sys.stderr)

    for note in task.describe_artifacts():
        print(f"NOTE: {note}", file=sys.stderr)

    try:
        image_id = task.image_id()
    except TaskError as e:
        print(f"error: {e}\nBuild it first: .venv/bin/python scripts/prepare_task.py {task.id} --build",
              file=sys.stderr)
        return 1

    model = build_model(args.model, max_tokens=args.max_tokens)
    print(f"model id:   {MODELS[args.model].id}")
    print(f"image:      {image_id[:19]}")

    swept = sweep_orphans()
    if swept:
        print(f"swept {len(swept)} orphaned container(s) from an earlier run")

    try:
        with Sandbox(task.image, env=task.env) as box:
            tools = Tools(box, task.test_command)

            # The before-state is captured here, not read from task.yaml:
            # the score must compare against what THIS container actually
            # did, not against a number recorded on another machine.
            print("recording the starting state...")
            before_result = tools.run_tests()
            if before_result.is_error:
                print(f"the suite could not run before the agent started: "
                      f"{before_result.reason}", file=sys.stderr)
                return 1
            before = SuiteState.from_result(before_result)
            print(f"  {before_result.passed} passed, "
                  f"{before_result.failed} failed, "
                  f"{before_result.errored} errored")

            if not before.failed_ids:
                print("nothing is failing — this task measures nothing as "
                      "configured. Refusing to run.", file=sys.stderr)
                return 1

            expected = task.expected_broken.get("failed")
            if expected is not None and expected != len(before.failed_ids):
                print(f"WARNING: task.yaml records {expected} failing tests, "
                      f"this container has {len(before.failed_ids)}. The "
                      f"image or the pins have drifted.", file=sys.stderr)

            # Everything untracked at this point is an artifact of running
            # the suite (task 002 leaves a `testsuite` SQLite file), not of
            # the agent's work. Excluded from the diff from here on.
            artifacts = box.mark_baseline()
            if artifacts:
                print(f"  test artifacts excluded from the diff: "
                      f"{', '.join(sorted(artifacts))}")

            failures = tools.run_tests_summary()

            agent = Agent(tools, model=model,
                          max_iterations=args.iterations,
                          max_cost_usd=args.max_cost,
                          max_wall_seconds=args.max_wall,
                          # Anthropic-only syntax. Sending cache_control
                          # blocks to an OpenAI-compatible endpoint is not a
                          # no-op — it is a different content shape that the
                          # provider may reject or silently mangle.
                          cache_prompt=(not args.no_cache
                                        and MODELS[args.model].supports_caching),
                          announce_budget=args.announce_budget)

            fingerprint = ConfigFingerprint(
                model_id=resolve_model_id(model),
                tool_signature=agent.tool_signature(),
                system_prompt=SYSTEM_PROMPT,
                max_iterations=args.iterations,
                max_cost_usd=args.max_cost,
                max_wall_seconds=args.max_wall,
                test_command=task.test_command,
                image_id=image_id,
                max_tokens=args.max_tokens,
                max_file_lines=MAX_FILE_LINES,
                max_search_hits=MAX_SEARCH_HITS,
                # Empty for a direct key; "openrouter:<provider>" when
                # routed. Two runs served by different upstreams are two
                # configurations, not two samples of one.
                provider_route=MODELS[args.model].route,
                announce_budget=args.announce_budget,
            )

            # Attached after the fingerprint, because a trace has to carry
            # the config_version to be findable from a record — and the
            # fingerprint is not known until the agent's tool signature is.
            # Empty unless LANGFUSE_PUBLIC_KEY is set; nothing here can
            # produce or change a number.
            agent.callbacks = tracing.callbacks(
                run_name=out.stem, task_id=task.id, model=args.model,
                config_version=fingerprint.digest())

            print(f"config:     {fingerprint.digest()}")
            print(f"running {args.model} (max {args.iterations} iterations, "
                  f"${args.max_cost:.2f}, {args.max_wall:.0f}s)...")

            run = agent.run(failures)
            after = SuiteState.from_result(run.verdict)

            score = score_run(
                run.diff, before, after,
                reference_diff=task.reference_diff(),
                reachability=task.reachability(),
                dependency=task.package,
                # A run the harness could not finish measured nothing. Its
                # progress would compute to a clean 0.0 and read as a model
                # failure, which is a different claim entirely.
                interrupted=run.harness_error or "",
            )

            outcome = final_outcome(
                after, score, run.stopped,
                hit_iteration_cap=run.hit_iteration_cap,
                hit_cost_cap=run.hit_cost_cap,
                hit_time_cap=run.hit_time_cap,
                harness_error=run.harness_error or None,
            )

            record = RunRecord(
                config_version=fingerprint.digest(),
                task_id=task.id,
                model=resolve_model_id(model),
                outcome=outcome,
                verdict_status=run.verdict.status,
                verdict_reason=run.verdict.reason,
                verdict_exit_code=run.verdict.exit_code,
                failed_ids=sorted(after.failed_ids),
                before_failed_ids=sorted(before.failed_ids),
                before_errors=dict(before.errors),
                tests_passed=run.verdict.passed,
                tests_failed=run.verdict.failed,
                tests_errored=run.verdict.errored,
                tests_skipped=run.verdict.skipped,
                tests_xfailed=run.verdict.xfailed,
                tests_xpassed=run.verdict.xpassed,
                iterations=run.iterations,
                cost_usd=run.cost_usd,
                cost_known=run.cost_known,
                wall_seconds=run.wall_seconds,
                max_iterations=args.iterations,
                max_cost_usd=args.max_cost,
                max_wall_seconds=args.max_wall,
                give_up_reason=run.give_up_reason,
                harness_error=run.harness_error,
                diff=run.diff,
                changed_files=run.changed_files,
                tool_calls=run.tool_calls,
                hallucinated_tools=run.hallucinated_tools,
                score=score.as_dict(),
                config=fingerprint.as_dict(),
                trajectory=run.trajectory,
            )

            # --- report ---------------------------------------------------
            cost = (f"${run.cost_usd}" if run.cost_known
                    else "$UNKNOWN (provider reported no usage)")
            print()
            print(f"outcome:    {outcome}")
            print(f"iterations: {run.iterations}/{args.iterations}   "
                  f"cost {cost}   {run.wall_seconds}s")
            print(f"tests:      {run.verdict.passed} passed, "
                  f"{run.verdict.failed} failed, "
                  f"{run.verdict.errored} errored, "
                  f"{run.verdict.xfailed} xfailed")

            if not score.valid:
                print(f"\nUNSCOREABLE — {score.unscoreable_reason}\n"
                      f"Discard this run; do not average it in.")
            else:
                print(f"fixed:      {score.fixed} of "
                      f"{len(before.failed_ids)}   "
                      f"newly broken: {score.newly_broken}   "
                      f"newly suppressed: {score.newly_suppressed}")
                print(f"progress:   {score.progress:.0%}"
                      f"   credible: {score.credible_progress:.0%}")
                print(f"root causes fixed: {score.clusters_fixed} of "
                      f"{score.clusters_total} "
                      f"({score.cluster_progress:.0%})")

            if run.give_up_reason:
                print(f"gave up:    {run.give_up_reason}")
            if run.harness_error:
                print(f"HARNESS ERROR: {run.harness_error}")

            _print_findings(score)

            if run.hallucinated_tools:
                print(f"\nhallucinated tools: "
                      f"{', '.join(run.hallucinated_tools)}")

            if run.tool_calls:
                print("\ntool calls:")
                for name, n in sorted(run.tool_calls.items(),
                                      key=lambda kv: -kv[1]):
                    print(f"  {n:>3}  {name}")
            else:
                print("\nNo tool calls at all — the model is probably not "
                      "handling the tool schemas. Check before trusting "
                      "this run.")

            # Langfuse batches, so a short run can exit before anything
            # is sent — which looks exactly like tracing being broken.
            tracing.flush()
            record.save(out)
            print(f"\ntrace: {out}")
            return 0

    except SandboxError as e:
        print(f"sandbox failure: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
