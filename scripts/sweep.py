#!/usr/bin/env python3
"""Run a matrix of agent attempts under a total budget.

    .venv/bin/python scripts/sweep.py --repeats 10 --budget 20
    .venv/bin/python scripts/sweep.py --plan ...        # print the matrix, spend nothing
    .venv/bin/python scripts/sweep.py --resume ...      # continue an interrupted sweep

Each attempt costs roughly $0.50 and six minutes, so a 2x4 matrix is about
$4 and forty minutes and a careless 4x4x3 is $24 and four hours. Three
properties follow from that, and all three are enforced rather than
documented:

  * A TOTAL BUDGET, checked before each attempt and again after. The
    per-run cap bounds one run; nothing bounded the sweep, and the way you
    discover that is the bill.

  * RESUMABLE. Runs are written one file per attempt and the sweep skips
    what already exists, so an interrupted sweep is continued rather than
    restarted. Re-running an existing cell would also be a silent protocol
    violation: repeats are supposed to be independent samples, and quietly
    replacing one with a fresh draw biases the set toward whatever the
    machine was doing when you pressed Ctrl-C.

  * ONE RECORD PER RUN, no aggregate file. The aggregate is derived from
    the records by inspect_run.py, so there is exactly one place a number
    can come from. A summary written here would be a second one, free to
    disagree.

A failed attempt does not stop the sweep — one container that would not
start should not cost you the other fifteen — but every failure is
reported at the end, and the exit status is non-zero if any occurred.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck import __version__ as HARNESS_VERSION
from agentcheck.ledger import Ledger
from agentcheck.models import DEFAULT_MAX_TOKENS, MODELS, family_of
from agentcheck.record import (
    IncompatibleRecord,
    holds_measurement,
    load,
    record_paths,
)
from agentcheck.sandbox import sweep_orphans
from agentcheck.task import Task, TaskError

#: Strings that identify a provider refusing service rather than a model
#: failing a task. A run killed by a rate limit measured NOTHING, and
#: scoring it as a failure would put the provider's capacity into the
#: results. It is recorded as a harness error and retried.
RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "too many requests",
                      "overloaded", "capacity", "quota")

#: Margin above the per-run wall cap before a container counts as an
#: orphan. A live sibling's container is younger than its own cap by
#: definition, so anything older cannot belong to a running attempt.
ORPHAN_MARGIN_SECONDS = 120.0

#: How long to wait after a provider refuses service before
#: retrying. Coming straight back at a rate limiter turns one
#: 429 into three and can extend the block.
RATE_LIMIT_BACKOFF_SECONDS = 60.0

#: Estimate each attempt from measured runs, with the cap as a fallback.
#: The ledger settles recorded costs after each attempt.
def planned_cost(max_cost: float, out_dir: Path | None = None,
                 models: Sequence[str] | None = None) -> float:
    """Expected cost of one attempt, in dollars, for the models being run.

    Reserving the full per-run CAP before each attempt is what a budget
    check has to do with no evidence — but it is wildly pessimistic once
    there is some. A measured ds-flash run costs about $0.05 against a
    $1.00 cap, so reserving the cap would let a $20 budget authorise
    twenty attempts when it can afford four hundred.

    So: the mean of what completed runs actually cost, with a margin, and
    never more than the cap. Falls back to the cap when nothing has run.

    `models` is not optional in spirit. This used to average EVERY record
    regardless of model, which is fine while one model has produced every
    run and silently wrong the moment a second arrives: `or-haiku` is
    ~12x the input and ~28x the output price of `or-ds-flash`, so history
    from the cheap one forecast the dear one at roughly a tenth of its
    cost. A forecast that quotes $0.80 for $8 of work is the house bug —
    a missing distinction becoming a plausible number — and it lands on
    the exact operation the second model exists for. With no history for
    the models asked about, the honest answer is the cap.
    """
    families = None
    if models:
        families = {MODELS[m].family or MODELS[m].id
                    for m in models if m in MODELS}

    observed = []
    for path in record_paths(out_dir or Path("runs")):
        try:
            record = load(path)
        except IncompatibleRecord:
            continue
        if record.harness_error:
            continue
        if families is not None and family_of(record.model) not in families:
            continue
        observed.append(record.cost_usd if record.cost_known
                        else record.max_cost_usd)

    if len(observed) < 3:
        return max_cost
    mean = sum(observed) / len(observed)
    return min(max_cost, max(mean * 2.0, 0.01))


#: Stop after this many attempts fail back to back. An expired key or a
#: dead daemon fails every remaining cell identically, and recording that
#: thirty times helps nobody.
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True)
class Cell:
    task_id: str
    model: str
    repeat: int
    out_dir: Path
    #: The iteration cap this attempt runs under. A dimension of the matrix
    #: rather than a global setting, so a budget ladder — the same task and
    #: model at 5, 10, 20 and 50 iterations — is one sweep. `config_version`
    #: already includes the cap, so the rungs land in separate cells and
    #: cannot be pooled by accident.
    iterations: int = 50

    #: True when the sweep spans more than one iteration budget, so the
    #: rung has to appear in the filename to keep the cells apart.
    laddered: bool = False

    #: A short marker for a non-default experimental arm, empty for the
    #: control. It belongs in the filename for the same reason the rung
    #: does: two configurations that can coexist in one directory need
    #: distinguishable names, or the second one silently claims the
    #: first's cells. `config_version` would catch the mix afterwards,
    #: but only after the runs had been spent.
    arm: str = ""

    @property
    def label(self) -> str:
        """What is passed to run_agent as --label.

        THE source of the filename. `run_agent` builds its output path as
        `{task}-{model}-{label}.json`, so anything this does not carry
        cannot appear in the name — and the sweep's idea of where the
        record lives would diverge from where it actually lands.

        That is exactly what happened: the rung was folded into the sweep's
        path but not into the label, so the sweep looked for
        `...-i5-r01.json` while run_agent wrote `...-r01.json`. Every
        attempt "failed" with a record sitting on disk beside it, and the
        second cell then collided with the first cell's file.

        It is the bug this project was built to talk about, in its own
        sweep: two places computing one name, free to disagree, with the
        disagreement surfacing as a plausible-looking failure rather than a
        crash. There is now one formula, and `path` is derived from it.
        """
        arm = f"{self.arm}-" if self.arm else ""
        rung = f"i{self.iterations}-" if self.laddered else ""
        return f"{arm}{rung}r{self.repeat:02d}"

    @property
    def path(self) -> Path:
        """Derived from the label, using run_agent's own formula."""
        return self.out_dir / f"{self.task_id}-{self.model}-{self.label}.json"

    @property
    def name(self) -> str:
        return (f"{self.task_id} · {self.model} · {self.iterations}it · "
                f"{self.label}")

    @property
    def key(self) -> str:
        """Ledger key. The record path, which is unique per cell."""
        return self.path.name


@dataclass
class SweepState:
    done: list[Cell] = field(default_factory=list)
    skipped: list[Cell] = field(default_factory=list)
    failed: list[tuple[Cell, str]] = field(default_factory=list)
    spent: float = 0.0

    @property
    def attempted(self) -> int:
        return len(self.done) + len(self.failed)


def build_matrix(tasks: list[str], models: list[str], repeats: int,
                 out_dir: Path, ladder: list[int], arm: str = "") -> list[Cell]:
    """Task-major, then repeat, then model, then iteration budget.

    Repeat before model on purpose: an interrupted sweep then has an equal
    number of samples per model rather than all of one and none of the
    other, so a partial sweep is still comparable.

    The filename carries the iteration budget ONLY when more than one rung
    is being run. A single-rung sweep keeps the historical
    `<task>-<model>-rNN.json` name, so resuming over records written before
    the ladder existed still finds them — a naming change would make every
    completed attempt look pending and re-run the whole archive.
    """
    if not ladder:
        raise ValueError("no iteration budget given")
    rungs = sorted(set(ladder))
    laddered = len(rungs) > 1
    cells = []
    for task_id in tasks:
        for repeat in range(1, repeats + 1):
            for model in models:
                for iterations in rungs:
                    cells.append(Cell(task_id, model, repeat, out_dir,
                                      iterations, laddered, arm))
    return cells


def looks_rate_limited(text: str) -> bool:
    """Did the provider refuse service, rather than the model fail?

    The distinction matters more than it looks. A run ended by a 429
    produced no sample: the agent did not decline, did not run out of
    budget, and did not fail the task. Recording it as any agent outcome
    would put the provider's capacity on that afternoon into the results,
    where it is indistinguishable from model behaviour.
    """
    low = text.lower()
    return any(marker in low for marker in RATE_LIMIT_MARKERS)


def run_cell(cell: Cell, max_cost: float, max_wall: float,
             out_dir: Path, extra: list[str]) -> tuple[bool, str, bool]:
    """(ok, why, was a provider refusal).

    stderr is captured rather than inherited so the failure can be
    classified; it is echoed on failure so nothing is lost.
    """
    cmd = [sys.executable, str(Path(__file__).with_name("run_agent.py")),
           cell.task_id, "--model", cell.model, "--label", cell.label,
           "--out-dir", str(out_dir),
           "--iterations", str(cell.iterations),
           "--max-cost", f"{max_cost}",
           "--max-wall", f"{max_wall}", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return (False, f"run_agent.py exited {proc.returncode}",
                looks_rate_limited(proc.stderr + proc.stdout))
    if not cell.path.exists():
        return False, f"no record written at {cell.path}", False

    # A written record is not a completed attempt. `run_agent` catches a
    # harness failure, records it, and exits 0 — so a rate-limited run that
    # never made a single model call arrived here as a success, and the
    # first ladder sweep reported "64 completed, 0 failed" while 22 of them
    # had been refused service by the provider.
    #
    # Reading the record is the only way to tell. The subprocess's exit
    # status describes whether the SCRIPT worked; the record describes
    # whether the RUN did, and those are different questions.
    try:
        record = load(cell.path)
    except IncompatibleRecord as e:
        return False, f"record is unreadable: {e}", False

    if record.harness_error:
        throttled = looks_rate_limited(record.harness_error)
        # The record stays on disk. It is evidence of a harness failure and
        # is exactly what the harness-failure rate in RESULTS.md counts —
        # deleting it would shrink the denominator, which is its own way of
        # fabricating a result. But the cell is NOT complete, so `is_complete`
        # rejects it and a resumed sweep will run it again.
        return False, f"harness error: {record.harness_error[:160]}", throttled

    return True, "", False


def wait_for_docker(timeout: float = 600.0) -> bool:
    """Block until the daemon answers, or give up.

    Docker Desktop stops and restarts on its own — twice during one
    afternoon of building this. A sweep that starts an attempt against a
    daemon that is not there burns the attempt, and three of those in a row
    end the sweep. Waiting costs nothing when the daemon is healthy and
    saves the remaining hours when it is not.
    """
    deadline = time.time() + timeout
    warned = False
    while time.time() < deadline:
        probe = subprocess.run(["docker", "info"], capture_output=True)
        if probe.returncode == 0:
            if warned:
                print("  docker is back", file=sys.stderr)
            return True
        if not warned:
            print(f"  waiting for the docker daemon (up to "
                  f"{timeout / 60:.0f} min)...", file=sys.stderr)
            warned = True
        time.sleep(10)
    return False


def superseded_configs(cells: list[Cell]) -> dict[str, int]:
    """Configurations present in the matrix other than the newest per cell.

    Resuming asks "does this cell hold a measurement". It never asked
    "was it produced by the configuration I am running NOW", and those are
    different questions — which is how a ladder ended up as sixteen groups
    of at most seven runs instead of eight rungs at n=8.

    Removing a provider pin changed `config_version`, correctly. The sweep
    then skipped 42 cells belonging to the previous experiment and ran only
    the 22 that happened to be missing, assembling a matrix from two
    incompatible halves. Nothing crashed; `group_by_config` would have
    split them later and every rung would simply have been short.

    Detected by comparing, per (task, iteration cap), the config of the
    most recently written record against the others. Returns the losers
    with their counts, empty when the matrix is homogeneous.
    """
    newest: dict[tuple[str, int], tuple[float, str]] = {}
    seen: list[tuple[tuple[str, int], str]] = []
    for cell in cells:
        if not cell.path.exists():
            continue
        try:
            record = load(cell.path)
        except IncompatibleRecord:
            continue
        key = (cell.task_id, cell.iterations)
        stamp = cell.path.stat().st_mtime
        seen.append((key, record.config_version))
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, record.config_version)

    stale: dict[str, int] = {}
    for key, config in seen:
        if config != newest[key][1]:
            stale[config] = stale.get(config, 0) + 1
    return stale


def is_complete(path: Path) -> bool:
    """Is this cell genuinely done, i.e. did it produce a MEASUREMENT?

    Delegates to `record.holds_measurement`, which is where the question is
    decided. It is asked here and again inside `run_agent`, and the two
    must not be able to disagree — when they did, the sweep scheduled 22
    cells and `run_agent` refused every one of them.
    """
    return holds_measurement(path)


def actual_cost(path: Path) -> float:
    """What a finished attempt really cost, from its record.

    An unknown cost is charged at the per-run CAP for budget purposes. It
    is not free, and treating it as zero is how a sweep with a broken
    usage report runs until the budget is exhausted in the other direction.
    """
    try:
        record = load(path)
    except IncompatibleRecord:
        return 0.0
    return record.cost_usd if record.cost_known else record.max_cost_usd


def chargeable(path: Path, sunk: float = 0.0) -> float:
    """What to put on the ledger for an attempt — finished OR failed.

    A failed attempt is not necessarily a free one. A run refused by a
    rate limiter at iteration 30 has paid for 30 iterations, and its
    record is on disk with the bill. The sweep used to release the
    reservation for every failure, which refunds money that was really
    spent — and a budget that refunds itself is not a cap. That is the
    same failure the ledger was written to prevent, one layer up.

    `sunk` is what an earlier attempt on this cell spent before the retry
    overwrote its record. Read then, because after the retry it is gone.
    """
    return sunk + (actual_cost(path) if path.exists() else 0.0)


def sweep_manifest(args, cells, state: SweepState, ledger: Ledger,
                   started: float) -> dict:
    """Everything needed to say what was running when these records appeared.

    Reproducibility here does not mean a reader can re-run the sweep and
    get the same numbers — agent runs are not deterministic and the whole
    project is about their variance. It means a reader can tell what was
    running: which images, which harness, which caps, which seeds, what it
    cost. A results table whose provenance is "some afternoon in September"
    is not checkable.
    """
    images = {}
    for task_id in sorted({c.task_id for c in cells}):
        try:
            images[task_id] = Task.load(task_id).image_id()
        except (TaskError, OSError) as e:
            images[task_id] = f"(unavailable: {e})"

    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True)

    return {
        "harness_version": HARNESS_VERSION,
        "git_sha": git.stdout.strip() if git.returncode == 0 else "",
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "model_specs": {alias: asdict(MODELS[alias])
                        for alias in sorted({c.model for c in cells})},
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(
            timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "matrix": {
            "tasks": sorted({c.task_id for c in cells}),
            "models": sorted({c.model for c in cells}),
            "iteration_ladder": sorted({c.iterations for c in cells}),
            "repeats": args.repeats,
            "cells": len(cells),
        },
        "caps": {"max_cost_usd": args.max_cost,
                 "max_wall_seconds": args.max_wall,
                 "max_tokens": args.max_tokens,
                 "budget_usd": args.budget},
        "jobs": args.jobs,
        "image_ids": images,
        "ledger": ledger.snapshot().as_dict(),
        "outcome": {"completed": [c.key for c in state.done],
                    "skipped": [c.key for c in state.skipped],
                    "failed": {c.key: why for c, why in state.failed}},
        # The harness venv, not the container's. What ran the agent is a
        # different question from what the agent was working against, and
        # a langchain upgrade between two sweeps is exactly the kind of
        # change that makes two columns incomparable without anyone
        # noticing.
        "pip_freeze": (frozen.stdout.splitlines()
                       if frozen.returncode == 0 else []),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(Task.available()),
                   help="comma-separated task ids")
    p.add_argument("--models", default="or-ds-flash",
                   help="comma-separated model aliases; defaults to "
                        "the cheapest, since a 30-run sweep on the "
                        "wrong one costs 2x")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--budget", type=float, default=10.00,
                   help="hard total spend cap for the whole sweep, USD")
    p.add_argument("--iterations", default="50",
                   help="iteration cap, or several separated by commas for "
                        "a budget ladder (e.g. 5,10,20,50). Each rung is a "
                        "separate configuration and is never pooled with "
                        "another")
    p.add_argument("--max-cost", type=float, default=1.00,
                   help="per-run spend cap, USD")
    p.add_argument("--max-wall", type=float, default=1800.0)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="maximum output tokens per model call")
    p.add_argument("--jobs", type=int, default=1,
                   help="attempts to run concurrently. Each holds a "
                        "container running a real test suite, so this is "
                        "bounded by memory, not by cores")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--manifest-dir", type=Path, default=None,
                   help="where the sweep manifest is written "
                        "(default: <out-dir>/sweeps)")
    p.add_argument("--plan", action="store_true",
                   help="print the matrix and the forecast, spend nothing")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--announce-budget", action="store_true",
                   help="run the budget-announcement arm; a separate "
                        "configuration from the default")
    p.add_argument("--allow-mixed", action="store_true",
                   help="proceed even when the existing records span "
                        "several configurations. They will not be pooled; "
                        "each is reported separately")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        ladder = [int(x) for x in str(args.iterations).split(",") if x.strip()]
    except ValueError:
        print(f"--iterations must be integers, got {args.iterations!r}",
              file=sys.stderr)
        return 1
    if not ladder or any(v < 1 for v in ladder):
        print("--iterations needs at least one positive value",
              file=sys.stderr)
        return 1
    if args.jobs < 1:
        print("--jobs must be at least 1", file=sys.stderr)
        return 1
    if args.max_tokens < 1:
        print("--max-tokens must be positive", file=sys.stderr)
        return 1

    unknown_tasks = [t for t in tasks if t not in Task.available()]
    unknown_models = [m for m in models if m not in MODELS]
    if unknown_tasks or unknown_models:
        print(f"unknown task(s) {unknown_tasks} model(s) {unknown_models}",
              file=sys.stderr)
        return 1

    # Fail before spending anything, not after the third attempt.
    for task_id in tasks:
        try:
            task = Task.load(task_id)
            task.image_id()
        except TaskError as e:
            print(f"error: {e}\n  .venv/bin/python scripts/prepare_task.py "
                  f"{task_id} --build", file=sys.stderr)
            return 1
        for note in task.describe_artifacts():
            print(f"NOTE [{task_id}]: {note}", file=sys.stderr)

    cells = build_matrix(tasks, models, args.repeats, args.out_dir, ladder,
                         arm="budget" if args.announce_budget else "")

    mixed = superseded_configs(cells)
    if mixed and not args.allow_mixed:
        print(f"\nREFUSING: the records this sweep would skip span "
              f"{len(mixed) + 1} configurations.\n", file=sys.stderr)
        for cfg, count in sorted(mixed.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3} record(s) under cfg {cfg}", file=sys.stderr)
        print(
            "\nResuming would assemble a matrix out of two different "
            "experiments. `config_version`\ncovers the model, prompt, "
            "tools, caps, test command, image digest and route —\nso a "
            "record made under different settings answers a different "
            "question, and\n`group_by_config` will refuse to pool it later "
            "anyway, leaving every rung short.\n\n"
            "Either archive the superseded records and re-run those cells:\n"
            "    .venv/bin/python scripts/archive_runs.py "
            "runs/*.json --superseded --apply\n\n"
            "or pass --allow-mixed if you genuinely want several "
            "configurations side by side\nand will report them separately.",
            file=sys.stderr)
        return 1

    pending = [c for c in cells if not is_complete(c.path)]
    existing = [c for c in cells if is_complete(c.path)]
    unusable = [c for c in cells
                if c.path.exists() and not is_complete(c.path)]
    if unusable:
        print(f"unusable: {len(unusable)} existing record(s) cannot be "
              f"loaded and will be re-run:")
        for c in unusable[:5]:
            print(f"  ! {c.path.name}")
    per_run = planned_cost(args.max_cost, args.out_dir, models)
    forecast = len(pending) * per_run

    print(f"matrix:   {len(tasks)} task(s) x {len(models)} model(s) "
          f"x {len(set(ladder))} budget(s) x {args.repeats} repeat(s) = "
          f"{len(cells)} attempts")
    if len(set(ladder)) > 1:
        print(f"ladder:   {', '.join(str(v) for v in sorted(set(ladder)))} "
              f"iterations — separate configurations, never pooled")
    if existing:
        print(f"existing: {len(existing)} already recorded, will be skipped")
    print(f"pending:  {len(pending)}")
    basis = ("the per-run cap (insufficient history or the estimate "
             "reaches the cap)" if per_run >= args.max_cost
             else f"2x the mean of completed runs for "
                  f"{', '.join(sorted(models))} (${per_run / 2:.3f})")
    print(f"forecast: ${forecast:.2f} for {len(pending)} attempt(s), "
          f"estimating ${per_run:.3f} each\n          from {basis}; "
          f"budget ${args.budget:.2f}")
    wall_hours = len(pending) * args.max_wall / 3600 / args.jobs
    print(f"time:     up to {wall_hours:.1f}h at the {args.max_wall:.0f}s "
          f"per-run cap with {args.jobs} job(s)")

    if forecast > args.budget:
        affordable = int(args.budget // per_run)
        print(f"\nWARNING: the full matrix cannot fit in the budget. "
              f"{affordable} of {len(pending)} attempts\nwill run before it "
              f"is exhausted, leaving an UNBALANCED matrix — which is worse "
              f"than\na smaller balanced one. Either raise --budget to "
              f"${forecast:.2f} or lower --repeats.",
              file=sys.stderr)

    if args.plan:
        for cell in cells:
            print(f"  {'skip' if is_complete(cell.path) else 'run '}  "
                  f"{cell.name}")
        return 0

    extra = ["--max-tokens", str(args.max_tokens)]
    if args.no_cache:
        extra.append("--no-cache")
    if args.announce_budget:
        extra.append("--announce-budget")
    state = SweepState()
    started = time.time()

    # The ledger lives beside the records, so two sweeps against the same
    # out-dir share one budget rather than each authorising the full amount.
    ledger = Ledger(args.out_dir / ".sweep-ledger.json", args.budget)

    # Spend that already happened has to be in the ledger before anything is
    # reserved, or a resumed sweep authorises the budget a second time.
    for cell in cells:
        if is_complete(cell.path):
            state.skipped.append(cell)
            ledger.adopt(cell.key, actual_cost(cell.path))
        else:
            # A cell that is pending but already settled is one a previous
            # sweep believed it had finished. That happened: 22 attempts
            # were settled before the harness learned to tell a rate-limited
            # run from a run that measured nothing. Retire the old entry so
            # the cell can be reserved again — the spend stays on the books
            # under a sunk key, because the money was spent either way.
            retired = ledger.retire(cell.key)
            if retired:
                print(f"  retired ${retired:.4f} of prior spend on "
                      f"{cell.key} — that attempt produced no measurement "
                      f"and the cell will run again")
    state.spent = ledger.snapshot().settled

    # A container older than the per-run wall cap cannot belong to a live
    # attempt, so reaping by age is safe alongside anything else running.
    #
    # This used to drop to 0 — reap everything — whenever --jobs was 1, on
    # the reasoning that a serial sweep has only one container at a time.
    # That reasoning ignores every OTHER process: the container test suite,
    # a second sweep, an interactive run_agent. Any of them would have its
    # containers destroyed mid-command by a serial sweep that believed it
    # was alone, and the victim records a harness failure it did not cause.
    #
    # There is no case where reaping a container younger than the wall cap
    # is correct, so the threshold now applies always.
    orphan_age = args.max_wall + ORPHAN_MARGIN_SECONDS

    stop = threading.Event()
    lock = threading.Lock()
    consecutive_failures = 0
    printed = threading.Lock()

    def attempt(cell: Cell) -> None:
        nonlocal consecutive_failures
        if stop.is_set():
            return

        snapshot = ledger.snapshot()
        estimate = min(args.max_cost,
                       max(planned_cost(args.max_cost, args.out_dir, models), 0.01))
        granted, why = ledger.reserve(cell.key, estimate)
        if not granted:
            with printed:
                print(f"\nCANNOT RUN {cell.name}: {why}.\nStopping.")
            stop.set()
            return

        settled = False
        #: Spend by an attempt whose record the retry will overwrite. Set
        #: before the try, because the `finally` below reads it on every
        #: path out — including one that raised before the first attempt.
        sunk = 0.0
        try:
            with printed:
                print(f"\n{'=' * 72}\n{cell.name}   "
                      f"(${snapshot.committed:.2f}/${args.budget:.2f} "
                      f"committed)\n{'=' * 72}")

            if not wait_for_docker():
                with lock:
                    state.failed.append((cell, "docker daemon unavailable"))
                stop.set()
                return

            sweep_orphans(orphan_age)
            ok, why, throttled = run_cell(cell, args.max_cost, args.max_wall,
                                          args.out_dir, extra)

            #: What a failed first attempt already spent. A run refused at
            #: iteration 30 by a rate limiter has paid for 30 iterations,
            #: and the retry below overwrites its record — so the figure is
            #: read here or it is lost. Charging it to nobody is how a cap
            #: stops binding, which is the one thing the ledger exists to
            #: prevent.
            if not ok:
                sunk = actual_cost(cell.path)
                # One retry, and only for a failure. A harness error
                # produced no sample at all — the attempt measured nothing
                # — so running it again is not a second draw from the same
                # cell, it is the first. Retrying a run that SUCCEEDED
                # would bias the set; retrying one that never happened
                # does not.
                #
                # A provider refusal waits before retrying. Coming straight
                # back at a rate limiter converts one 429 into three.
                if throttled:
                    with printed:
                        print(f"  provider refused service (rate limit); "
                              f"this attempt measured nothing. Waiting "
                              f"{RATE_LIMIT_BACKOFF_SECONDS:.0f}s.",
                              file=sys.stderr)
                    time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                else:
                    with printed:
                        print(f"  attempt failed ({why}); retrying once",
                              file=sys.stderr)
                wait_for_docker()
                sweep_orphans(orphan_age)
                ok, why, throttled = run_cell(cell, args.max_cost,
                                              args.max_wall, args.out_dir,
                                              extra)

            if ok:
                # The failed first attempt's spend rides along. It is real
                # money against this cell, whatever the retry produced.
                ledger.settle(cell.key, chargeable(cell.path, sunk))
                settled = True
                with lock:
                    state.done.append(cell)
                    state.spent = ledger.snapshot().settled
                    consecutive_failures = 0
            else:
                # A rate-limited attempt is a harness failure, not a model
                # result, and is named as one so it cannot be read as the
                # model having given up.
                reason = (f"provider refused service (rate limit) — no "
                          f"sample: {why}" if throttled else why)
                with lock:
                    state.failed.append((cell, reason))
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\nSTOPPING: {consecutive_failures} attempts "
                              f"failed in a row. Something is wrong that "
                              f"re-running will not fix —\ncheck the last "
                              f"error above, then resume this sweep with the "
                              f"same command.", file=sys.stderr)
                        stop.set()
                with printed:
                    print(f"FAILED: {reason}", file=sys.stderr)
        finally:
            if not settled:
                # A failed attempt is not necessarily a free one. If it
                # left a record, the model was called and the bill is
                # real: settle it, so a resumed sweep retires it and the
                # money stays on the books rather than being refunded by
                # the accounting. Only an attempt that produced no record
                # at all is released.
                spent = chargeable(cell.path, sunk)
                if spent:
                    ledger.settle(cell.key, spent)
                else:
                    ledger.release(cell.key)

    todo = [c for c in cells if not is_complete(c.path)]
    if args.jobs == 1:
        for cell in todo:
            if stop.is_set():
                break
            attempt(cell)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(attempt, cell) for cell in todo]
            for future in as_completed(futures):
                future.result()

    elapsed = (time.time() - started) / 60
    final = ledger.snapshot()
    print(f"\n{'=' * 72}")
    print(f"sweep finished in {elapsed:.0f} min")
    print(f"  completed: {len(state.done)}   skipped: {len(state.skipped)}"
          f"   failed: {len(state.failed)}")
    print(f"  spent:     ${final.settled:.2f} of ${args.budget:.2f}")
    if final.open_reservations:
        print(f"  WARNING: {final.open_reservations} reservation(s) left "
              f"open — a worker died without settling. Inspect "
              f"{ledger.path}.", file=sys.stderr)

    manifest_dir = args.manifest_dir or (args.out_dir / "sweeps")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / f"{stamp}.json"
    manifest_path.write_text(
        json.dumps(sweep_manifest(args, cells, state, ledger, started),
                   indent=2))
    print(f"  manifest:  {manifest_path}")

    if state.failed:
        print("\nfailed attempts (re-run the sweep to retry them):")
        for cell, why in state.failed:
            print(f"  x {cell.name}: {why}")

    print(f"\nAggregate with:\n"
          f"    .venv/bin/python scripts/inspect_run.py "
          f"{args.out_dir}/*.json --summary")
    print("No summary is written here on purpose — the records are the only "
          "source of\nany number, and a second one would be free to "
          "disagree.")
    return 1 if state.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
