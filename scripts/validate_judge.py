#!/usr/bin/env python3
"""Calibrate the judge against human labels.

    .venv/bin/python scripts/validate_judge.py build   runs/*.json -n 48
    .venv/bin/python scripts/validate_judge.py label   judge-labels.json
    .venv/bin/python scripts/validate_judge.py relabel judge-labels.json
    .venv/bin/python scripts/validate_judge.py status  judge-labels.json
    .venv/bin/python scripts/validate_judge.py run     judge-labels.json --model mimo
    .venv/bin/python scripts/validate_judge.py report  judge-labels.json

An unvalidated LLM judge is a number generator. These phases turn it into a
measurement, and the ORDER of them is the whole point.

  build    stratified sample of comparison pairs, with the frame recorded
  label    YOU label them, blind, in the judge's own vocabulary
  relabel  a subset again, days later, to measure your own reliability
  run      the judge labels the same pairs
  report   agreement, kappa with an interval, and every disagreement

Rules the phases enforce rather than recommend:

  * HUMAN LABELS FIRST. `run` refuses on any unlabelled pair, and `label`
    never shows a judge verdict. Seeing the judge's answer moves yours,
    and the agreement number then measures how persuasive the judge was.

  * THE HUMAN LABELS BLIND, IN THE SAME VOCABULARY. Neither the test
    results nor which patch is the maintainer's is shown, and the agent's
    slot is randomised per pair. Labelling in agent-relative terms while
    the judge labels in neutral terms would compare two different tasks;
    knowing which patch is upstream's is the strongest anchor available.

  * THE PROTOCOL IS PRE-REGISTERED. `docs/judge-protocol.md` fixes the
    headline statistic and the acceptance thresholds before the first
    label is entered, and `build` records its git hash. A threshold chosen
    after seeing the number measures nothing, and a reader who can check
    from `git log` that it was not has reason to believe the rest.

  * PAIRS DRAWN ACROSS ALL STRATA. Task, outcome class, and whether the
    run cheated — with a floor per cell. Calibrating only on runs that
    solved the task calibrates on the easy half of the distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck.calibration import (
    LABEL_SCHEMA_VERSION,
    REGISTERED_JUDGES,
    RELABEL_DELAY_DAYS,
    RELABEL_N,
    SKIP,
    agree,
    collapsed,
    position_bias,
    rate,
    retest_issues,
    retest_opens,
    retest_sample,
    stratified_sample,
    stratum_of,
    threshold_verdict,
)
from agentcheck.judge import (
    MAX_DIFF_CHARS,
    NEUTRAL_VERDICTS,
    SYSTEM_PROMPT,
    judge_patches,
    to_agent_relative,
    truncate,
)
from agentcheck.models import DEFAULT_MAX_TOKENS, MODELS, build_model, model_id
from agentcheck.record import load_all
from agentcheck.stats import kappa_band
from agentcheck.task import Task, TaskError

PROTOCOL = Path("docs/judge-protocol.md")
CALIBRATION_REPORT = Path("docs/judge-calibration.md")

#: Minimum characters of written rationale per label. Not a formality: the
#: rationale is what makes a disagreement adjudicable three weeks later,
#: and an unrationalised label is indistinguishable from clicking through.
#: 20 characters is roughly one clause — enough to say which call site was
#: missed, too short to be a burden.
MIN_RATIONALE = 20

#: The floor the project's scope sets on the calibration set, and clause 10 of
#: the protocol makes invalidating. Below it the kappa interval is wide
#: enough to span two Landis-Koch bands and the result cannot be placed.
#: `build` will still write a smaller file — a pilot on 15 pairs is a
#: reasonable thing to want — but it says loudly that what comes out is not
#: the calibration the protocol describes.
MIN_PAIRS = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_hash(path: Path) -> str:
    """The blob hash of a file as committed, or "" if it is not committed.

    Used to record which version of the protocol was in force. An
    uncommitted protocol is not pre-registered — nothing outside this
    machine can attest to when it was written — and `build` says so.
    """
    try:
        out = subprocess.run(["git", "rev-parse", f"HEAD:{path}"],
                             capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _committed_clean(path: Path) -> str:
    """The HEAD blob hash of `path` if it is committed with no local edits.

    "" if it is untracked, modified, or outside a repository. That is the
    whole difference between "the labels predate the judge" as a claim
    and as something a reader can check from `git log`.
    """
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(path)],
                               capture_output=True, text=True, timeout=30,
                               check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if dirty.returncode != 0 or dirty.stdout.strip():
        return ""
    return _git_hash(path)


def _committed_labels(path: Path) -> str:
    """Prove the human labels and sampling inputs predate judge output.

    Judge output is appended to the same file. Requiring that output to
    be committed before every resume adds no evidence about WHEN the human
    labelled it. Compare all input fields against HEAD, excluding only the
    judge-output dictionaries, and keep the exact committed blob reference.
    """
    clean = _committed_clean(path)
    if clean:
        return clean
    baseline_hash = _git_hash(path)
    if not baseline_hash:
        return ""
    try:
        committed = subprocess.run(["git", "cat-file", "blob", baseline_hash],
                                   capture_output=True, text=True, timeout=30,
                                   check=False)
        if committed.returncode:
            return ""
        baseline = json.loads(committed.stdout)
        current = json.loads(path.read_text())

        def inputs(data: dict) -> dict:
            return {**data, "pairs": [
                {k: v for k, v in pair.items() if k != "judge"}
                for pair in data["pairs"]]}

        return baseline_hash if inputs(baseline) == inputs(current) else ""
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return ""


def _read(path: Path) -> dict:
    data = json.loads(path.read_text())
    version = data.get("version")
    if version != LABEL_SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: labels file is v{version}, this harness writes "
            f"v{LABEL_SCHEMA_VERSION}. Hand labels are too expensive to "
            f"reinterpret under new assumptions — rebuild and relabel, or "
            f"check out the harness version that wrote it.")
    return data


def _write(path: Path, data: dict) -> None:
    """Write atomically. A Ctrl-C mid-write must not cost the labels."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# --- build ------------------------------------------------------------------

def build(paths: list[str], out: Path, n: int, seed: int, floor: int) -> int:
    records, rejected = load_all(paths)
    for reason in rejected:
        print(f"  x {reason}", file=sys.stderr)
    if not records:
        print("no loadable records", file=sys.stderr)
        return 1

    protocol_hash = _git_hash(PROTOCOL)
    if not protocol_hash:
        print(f"REFUSING: {PROTOCOL} is not committed.\n"
              f"The protocol fixes the headline statistic and the acceptance\n"
              f"thresholds, and its value comes from provably predating the\n"
              f"labels. Commit it first.", file=sys.stderr)
        return 1

    tasks: dict[str, Task] = {}
    candidates: list[dict[str, Any]] = []
    for r in records:
        if r.task_id not in tasks:
            try:
                tasks[r.task_id] = Task.load(r.task_id)
            except TaskError as e:
                print(f"skipping {r.name}: {e}", file=sys.stderr)
                continue
        task = tasks[r.task_id]
        if task.reference_diff() is None:
            print(f"skipping {r.name}: no reference diff for {r.task_id} "
                  f"(prepare_task.py --reference)", file=sys.stderr)
            continue
        if not r.diff.strip():
            # Nothing to compare. Not a skip to be labelled — there is no
            # patch, so there is no judgement to make about one.
            continue
        candidates.append({
            "id": f"{r.task_id}::{r.name}",
            "stratum": stratum_of(r.task_id, r.outcome, r.score),
            "record": r,
        })

    if not candidates:
        print("nothing to validate against", file=sys.stderr)
        return 1

    frame = stratified_sample(candidates, n, seed, floor)
    by_id = {c["id"]: c["record"] for c in candidates}

    # An absent cheating stratum is not a detail. It means the archive
    # cannot calibrate the judge on the case the project exists to measure,
    # and a sample that quietly omits it would read as a full calibration.
    cheat_available = sum(1 for c in candidates if c["stratum"][2])
    if not cheat_available:
        print("WARNING: no cheating run is available to sample.\n"
              "  The judge will be calibrated only on honest patches, which "
              "is the case\n  it finds easiest. Say so in the report, or "
              "run the sweep first.", file=sys.stderr)

    rng = random.Random(seed ^ 0x5EED)      # a separate stream from sampling
    pairs: list[dict[str, Any]] = []
    for pid in frame.picked:
        r = by_id[pid]
        pairs.append({
            "id": pid,
            "task_id": r.task_id,
            "run": r.name,
            "path": str(r.path),
            "model": r.model,
            "agent_diff": r.diff,
            # Which slot the agent's patch occupies in the labelling view.
            # Randomised so the human's own position preference becomes
            # measurable rather than assumed away.
            "agent_position": rng.choice(("a", "b")),
            "human": None,
            "relabel": None,
            "judge": {},
        })

    payload = {
        "version": LABEL_SCHEMA_VERSION,
        "created": _now(),
        "protocol": str(PROTOCOL),
        "protocol_git_hash": protocol_hash,
        "frame": frame.as_dict(),
        "pairs": pairs,
    }
    _write(out, payload)

    print(f"wrote {out} — {len(pairs)} pairs drawn from a frame of "
          f"{frame.frame_size}, seed {seed}")
    print(f"protocol {PROTOCOL} @ {protocol_hash[:12]}")
    print("\nstrata (available -> drawn):")
    for key, counts in frame.strata.items():
        print(f"  {key:<44} {counts['available']:>3} -> {counts['drawn']}")
    if len(pairs) < MIN_PAIRS:
        print(f"\nWARNING: {len(pairs)} pairs, below the {MIN_PAIRS} the "
              f"protocol requires (clause 10).\n"
              f"  The whole frame is {frame.frame_size} candidates, so this "
              f"is not a sampling\n  choice — the archive is too small. "
              f"Run the sweep first and rebuild.\n"
              f"  Anything produced from this file is a PILOT and must be "
              f"labelled as one;\n  a kappa at n={len(pairs)} has an "
              f"interval spanning two agreement bands.",
              file=sys.stderr)

    print(f"\nLabel them BEFORE running the judge:\n"
          f"    .venv/bin/python scripts/validate_judge.py label {out}")
    return 0


# --- labelling --------------------------------------------------------------

def _show_diff(diff: str, title: str) -> str:
    """One patch, truncated by exactly the rule the judge is truncated by.

    It used to be the first 120 LINES. The judge reads 60,000 characters,
    which is every patch in this sample — so the human was labelling a
    ninth of task 002's reference patch while the judge read all of it,
    and the kappa between them would have carried that difference as if it
    were disagreement about the patches.

    `judge-protocol.md` §4 lists what is hidden from the annotator on
    purpose. "Most of the diff" is not on that list, and a limit chosen
    for terminal comfort is not a blinding decision.
    """
    truncated: list[str] = []
    body = truncate(diff, title, MAX_DIFF_CHARS, truncated)
    head = f"\n--- {title} ({len(diff.splitlines())} lines) " + "-" * 30
    return f"{head}\n{body}"


def _token(pair: dict) -> str:
    """A short, stable name for a pair that reveals nothing about it.

    The pair's own id is `task::run-name`, and the run name carries the
    model and the configuration — two of the four anchors `judge-protocol`
    §4 hides. A hash of it names the pair for cross-checking a written
    sheet against the prompt without putting either back on screen.
    """
    return hashlib.sha256(pair["id"].encode()).hexdigest()[:6]


def _render(pair: dict, task: Task, index: str,
            show_token: bool = True) -> str:
    """The labelling view. Everything it does NOT show is deliberate.

    Hidden: the test results, the outcome, the score, the run's model, and
    which of the two patches is the maintainer's. Each of them is an
    anchor. Knowing a patch reached 115 passed / 0 failed answers "did it
    work"; the question here is "is it the same fix".

    Printed, not paged. A pager was tried and was worse than either
    alternative: `less` restores the terminal when you quit it, so the two
    patches disappeared at the exact moment the verdict prompt appeared,
    and the annotator was answering from memory. Printing leaves them in
    the scrollback, where they can be read while the answer is typed —
    and `v` at the prompt prints them again.

    Returns the view as well as printing it, so the prompt can re-show it.
    """
    agent_first = pair["agent_position"] == "a"
    reference = task.reference_diff() or ""

    view = "\n".join([
        "=" * 72,
        f"[{index}]  {pair['task_id']}   "
        f"{task.package} {task.from_version} -> {task.to_version}"
        + (f"   ·   pair {_token(pair)}" if show_token else ""),
        "=" * 72,
        _show_diff(pair["agent_diff"] if agent_first else reference,
                   "PATCH A"),
        _show_diff(reference if agent_first else pair["agent_diff"],
                   "PATCH B"),
    ])
    print(view)
    return view


#: Re-print the patches instead of answering. Not a verdict, and not in
#: NEUTRAL_VERDICTS, so it can never be recorded as one.
REVIEW = "v"


def _ask_verdict(view: str = "") -> str:
    options = sorted(NEUTRAL_VERDICTS)
    print(f"\n  {' | '.join(options)} | {SKIP}"
          + (f"    ({REVIEW} = show the patches again)" if view else ""))
    while True:
        answer = input("  verdict: ").strip().lower()
        if answer in NEUTRAL_VERDICTS or answer == SKIP:
            return answer
        if answer == REVIEW and view:
            print(view)
            continue
        print(f"  not one of: {', '.join(options)}, {SKIP}")


def _ask_rationale() -> str:
    while True:
        text = input(f"  why (>= {MIN_RATIONALE} chars): ").strip()
        if len(text) >= MIN_RATIONALE:
            return text
        print(f"  {len(text)} characters. A label you cannot explain in a "
              f"clause is one you\n  will not be able to adjudicate later.")


def _label_entry(pair: dict, neutral: str, rationale: str,
                 source: str = "prompt") -> dict:
    return {
        "neutral": neutral,
        # Stored alongside, not instead: the neutral label is what was
        # actually entered, and the agent-relative one is derived. Keeping
        # both means a bug in the mapping is visible rather than baked in.
        "agent_relative": (SKIP if neutral == SKIP
                           else to_agent_relative(neutral,
                                                  pair["agent_position"])),
        "rationale": rationale,
        "at": _now(),
        # Where the label was entered. A sheet filled in offline and
        # imported carries one timestamp for the batch rather than one per
        # pair, and the report should be able to say so rather than
        # implying a precision the file does not have.
        "source": source,
    }


def label(path: Path, dry_run: bool) -> int:
    data = _read(path)
    tasks: dict[str, Task] = {}

    todo = [p for p in data["pairs"] if p["human"] is None]
    if dry_run:
        if not data["pairs"]:
            print("no pairs in this file", file=sys.stderr)
            return 1
        pair = data["pairs"][0]
        tasks[pair["task_id"]] = Task.load(pair["task_id"])
        _render(pair, tasks[pair["task_id"]], "dry run")
        print("\n(dry run — nothing was recorded)")
        return 0

    if not todo:
        print("every pair is already labelled")
        return 0

    print(f"{len(todo)} pair(s) to label. Ctrl-C saves and stops.\n"
          f"Two patches for the same migration. Which is the maintainer's is\n"
          f"not shown, and neither are the test results. Say how A relates\n"
          f"to B.\n")

    try:
        for i, pair in enumerate(todo, 1):
            if pair["task_id"] not in tasks:
                tasks[pair["task_id"]] = Task.load(pair["task_id"])
            view = _render(pair, tasks[pair["task_id"]], f"{i}/{len(todo)}")

            neutral = _ask_verdict(view)
            rationale = ("not worth labelling" if neutral == SKIP
                         else _ask_rationale())
            pair["human"] = _label_entry(pair, neutral, rationale)
            _write(path, data)
    except (KeyboardInterrupt, EOFError):
        print("\nstopped")

    done = sum(1 for p in data["pairs"] if p["human"] is not None)
    print(f"\n{done}/{len(data['pairs'])} labelled -> {path}")
    if done == len(data["pairs"]):
        print("\nCommit this file now, before the judge runs. The git "
              "history is what\nproves the labels were not written after "
              "seeing the judge's answers.")
    return 0


def sheet(path: Path, out: Path, include_labelled: bool,
          force: bool = False) -> int:
    """Write every unlabelled pair to one document, for reading away from
    the terminal.

    The same views the prompt shows, built by the same function, so the
    sheet cannot drift from what you are asked about. Each pair carries
    the token the prompt prints, which is how you check that the answer
    you wrote against `a0ef25` is going into the pair the tool is asking
    about — the sheet is read in one order and typed in another often
    enough that it needs to be checkable.

    Nothing is recorded here. Verdicts still go through `label`, because
    the labels file is what carries the timestamps and the rationales, and
    a sheet filled in offline has neither.
    """
    data = _read(path)
    pairs = [p for p in data["pairs"]
             if include_labelled or p["human"] is None]
    if not pairs:
        print("every pair is already labelled", file=sys.stderr)
        return 1

    # Regenerating over a sheet you have been filling in would delete the
    # only copy of that work: the sheet is derived, so it is not in git,
    # and the verdicts in it are not in the labels file until they are
    # imported. An hour of labelling is not something to lose to a command
    # that looks idempotent.
    if out.exists() and not force:
        recorded = {_token(p) for p in data["pairs"] if p["human"]}
        pending = [token for token, _v, _w in _sheet_rows(out.read_text())
                   if token not in recorded]
        if pending:
            print(f"REFUSING to overwrite {out}: it holds "
                  f"{len(pending)} verdict(s) that are not in {path} yet.\n"
                  f"  Import them first:\n"
                  f"    .venv/bin/python scripts/validate_judge.py import "
                  f"{out} --into {path}\n"
                  f"  or pass --force to throw them away.", file=sys.stderr)
            return 1

    tasks: dict[str, Task] = {}
    doc = [
        "# Judge calibration — the pairs to label",
        "",
        f"{len(pairs)} pair(s), generated {_now()} from `{path}`.",
        f"Protocol `{data['protocol']}` @ {data['protocol_git_hash'][:12]}.",
        "",
        "Read these, then record your verdicts either way:",
        "",
        "- **In this file** — fill in the checklist below and import it "
        "(one command, matched on the token).",
        f"- **At the prompt** — `validate_judge.py label {path}`, which "
        "prints the same token in its header, so you can check that the "
        "pair on screen is the one you are reading. If the tokens differ, "
        "stop: a verdict filed against the wrong pair is worse than an "
        "unlabelled one.",
        "",
        "Both go through the same recording path, which derives the "
        "agent-relative label from the randomised slot and stamps the "
        "time. Editing `judge-labels.json` by hand does neither.",
        "",
        "**This file is not in git** — it is 2 MB of diffs already stored "
        "in `runs/`, regenerated on demand. Your verdicts only become "
        "durable when you import them, so import as you go rather than "
        "filling in all of it first. Re-importing a sheet is safe: rows "
        "that match what is already recorded are counted and skipped.",
        "",
        "**What is deliberately not here:** the test results, the outcome, "
        "the progress score, which model produced which patch, and which "
        "of A and B is the maintainer's. Do not go looking for them in "
        "`runs/` — each is an anchor, and `judge-protocol.md` §4 hides "
        "them on purpose. A is not 'the agent': the slot is randomised "
        "per pair.",
        "",
        "## The question, exactly as the judge is asked it",
        "",
        "Both raters answer the same question from the same definitions, "
        "or the agreement between them measures the difference in their "
        "instructions.",
        "",
        "```",
        SYSTEM_PROMPT.strip(),
        "```",
        "",
        "## Checklist",
        "",
        "Fill in the last two columns, then import them with:",
        "",
        "```",
        f"    .venv/bin/python scripts/validate_judge.py import {out} "
        f"--into {path}",
        "```",
        "",
        f"A verdict needs a reason of at least {MIN_RATIONALE} characters "
        f"— `skip` is the one exception. The import matches on the token, "
        f"never on row order, refuses a token it does not recognise, and "
        f"will not overwrite a pair you have already labelled. Rows you "
        f"leave blank are simply not imported, so filling this in over "
        f"several sittings is fine.",
        "",
        "| # | pair | task | verdict | why |",
        "|---|---|---|---|---|",
    ]
    for i, pair in enumerate(pairs, 1):
        doc.append(f"| {i} | `{_token(pair)}` | `{pair['task_id']}` |  |  |")
    doc.append("")

    for i, pair in enumerate(pairs, 1):
        if pair["task_id"] not in tasks:
            tasks[pair["task_id"]] = Task.load(pair["task_id"])
        task = tasks[pair["task_id"]]
        agent_first = pair["agent_position"] == "a"
        reference = task.reference_diff() or ""
        doc += [
            "---",
            "",
            f"## {i}/{len(pairs)} · pair `{_token(pair)}` · "
            f"`{pair['task_id']}` · {task.package} {task.from_version} → "
            f"{task.to_version}",
            "",
            "### PATCH A",
            "",
            "```diff",
            _show_diff(pair["agent_diff"] if agent_first else reference,
                       "PATCH A").strip(),
            "```",
            "",
            "### PATCH B",
            "",
            "```diff",
            _show_diff(reference if agent_first else pair["agent_diff"],
                       "PATCH B").strip(),
            "```",
            "",
            "**verdict:**",
            "",
        ]

    out.write_text("\n".join(doc))
    size = out.stat().st_size
    print(f"wrote {out} — {len(pairs)} pair(s), {size / 1e6:.1f} MB")
    print("Fill in its checklist and import it:\n"
          f"    .venv/bin/python scripts/validate_judge.py import {out} "
          f"--into {path}\n"
          "or answer at the prompt instead:\n"
          f"    .venv/bin/python scripts/validate_judge.py label {path}")
    return 0


def _sheet_rows(text: str) -> list[tuple[str, str, str]]:
    """(token, verdict, why) for every filled-in row of a sheet's checklist.

    Matched on the TOKEN, never on row order. A sheet is read in one order
    and typed in another — that is the whole reason it exists — and a
    verdict filed against the wrong pair is worse than an unlabelled one.

    The `why` column is allowed to contain pipes: everything after the
    fourth cell is rejoined, so a rationale like "uses a|b spelling" is
    kept rather than silently cut in half.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = line.split("|")
        if len(cells) < 6:
            continue
        token = cells[2].strip().strip("`")
        verdict = cells[4].strip().lower()
        why = "|".join(cells[5:-1]).strip()
        if token in ("pair", "") or not verdict:
            continue
        rows.append((token, verdict, why))
    return rows


def import_sheet(sheet_path: Path, labels_path: Path) -> int:
    """Take the verdicts written into a sheet and record them properly.

    Strict on purpose, and it refuses the whole file rather than applying
    part of it. This writes the reference data the judge is measured
    against; a half-applied import would leave a labels file nobody can
    reason about, and "which of these 46 went in?" is not a question worth
    having.
    """
    if not sheet_path.exists():
        print(f"{sheet_path} does not exist", file=sys.stderr)
        return 1
    data = _read(labels_path)
    by_token = {_token(p): p for p in data["pairs"]}

    rows = _sheet_rows(sheet_path.read_text())
    if not rows:
        print(f"no filled-in verdicts in {sheet_path}", file=sys.stderr)
        return 1


    problems, ready, unchanged = [], [], []
    for token, verdict, why in rows:
        pair = by_token.get(token)
        if pair is None:
            problems.append(f"{token}: no such pair in {labels_path}. This "
                            f"sheet was written against a different frame.")
            continue
        if verdict not in NEUTRAL_VERDICTS and verdict != SKIP:
            problems.append(f"{token}: {verdict!r} is not a verdict "
                            f"({', '.join(sorted(NEUTRAL_VERDICTS))}, {SKIP})")
            continue
        if pair["human"] is not None:
            # Re-importing a sheet you have added rows to is the expected
            # way to work, so a row that agrees with what is already
            # recorded is done, not an error. Only a row that CONTRADICTS
            # a recorded label stops the import — that is either a changed
            # mind, which belongs in a deliberate edit, or two sheets out
            # of step, which is worth stopping for.
            if pair["human"]["neutral"] == verdict:
                unchanged.append(token)
                continue
            problems.append(f"{token}: already labelled "
                            f"{pair['human']['neutral']!r} at "
                            f"{pair['human']['at'][:19]}, and this sheet "
                            f"says {verdict!r}. Nothing here overwrites a "
                            f"label.")
            continue
        if verdict != SKIP and len(why) < MIN_RATIONALE:
            problems.append(f"{token}: the reason is {len(why)} characters, "
                            f"and {MIN_RATIONALE} are required. A label you "
                            f"cannot explain in a clause is one you will "
                            f"not be able to adjudicate later.")
            continue
        ready.append((pair, verdict, why))

    if problems:
        print(f"REFUSING to import {sheet_path} — "
              f"{len(problems)} problem(s), and a partly applied import is "
              f"worse than none:\n", file=sys.stderr)
        for problem in problems:
            print(f"  x {problem}", file=sys.stderr)
        if ready:
            print(f"\n  {len(ready)} row(s) were fine. Fix the above and "
                  f"run it again — nothing has been written.", file=sys.stderr)
        return 1

    for pair, verdict, why in ready:
        pair["human"] = _label_entry(
            pair, verdict, why or "not worth labelling", source="sheet")
    if ready:
        _write(labels_path, data)

    done = sum(1 for p in data["pairs"] if p["human"] is not None)
    print(f"imported {len(ready)} verdict(s) from {sheet_path}"
          + (f"; {len(unchanged)} row(s) were already recorded"
             if unchanged else ""))
    print(f"{done}/{len(data['pairs'])} labelled -> {labels_path}")
    if done == len(data["pairs"]):
        print("\nCommit this file now, before the judge runs. The git "
              "history is what\nproves the labels were not written after "
              "seeing the judge's answers.")
    return 0


def relabel(path: Path, n: int, force: bool) -> int:
    """Re-label a subset, days later, to measure your own reliability.

    RoadmapBench's 0.83 is a human-human figure. A single-annotator project
    cannot produce inter-rater agreement honestly, but it can produce
    test-retest reliability, and without some human number a judge-human
    kappa of 0.65 has nothing to be compared against. It also bounds the
    judge from above: agreement with a rater who disagrees with themselves
    cannot exceed that rater's self-agreement by much.

    A different randomisation seed, so the agent's slot moves, and the
    first labels are never shown.
    """
    data = _read(path)
    labelled = [p for p in data["pairs"]
                if p["human"] and p["human"]["neutral"] != SKIP]
    if len(labelled) < n:
        print(f"only {len(labelled)} labelled pair(s); need {n}",
              file=sys.stderr)
        return 1

    # A different stream from the one that fixed the first pass's slots,
    # so a remembered "A was the narrow one" does not survive as a correct
    # answer. Derived from the build seed, so the retest is reproducible.
    selected = [(p, slot) for p, slot in retest_sample(data, n)
                if p["relabel"] is None]
    subset = [p for p, _ in selected]
    if not subset:
        print("every sampled pair has already been re-labelled")
        return 0

    # The delay is a property of EACH pair: how long since that pair was
    # labelled. It used to be measured from the oldest label in the file,
    # so one pair labelled a month ago unlocked the retest of a pair
    # labelled yesterday — recall passing itself off as reliability. The
    # gate is the newest first label among the pairs about to be retested.
    now = datetime.now(timezone.utc)
    labelled_at = {p["id"]: datetime.fromisoformat(p["human"]["at"])
                   for p in subset}
    newest = max(labelled_at.values())
    if now - newest < timedelta(days=RELABEL_DELAY_DAYS) and not force:
        opens = newest + timedelta(days=RELABEL_DELAY_DAYS)
        print(f"REFUSING: {len(subset)} pair(s) are due for the retest, and "
              f"the most recently\nlabelled of them is "
              f"{(now - newest).days} day(s) old. Test-retest measures "
              f"reliability, not recall.\nIt opens at "
              f"{opens:%Y-%m-%d %H:%M} UTC, or pass --force and let the "
              f"report state the\nactual interval for every pair.",
              file=sys.stderr)
        return 1

    tasks: dict[str, Task] = {}
    shortest = min((now - at).days for at in labelled_at.values())
    print(f"re-labelling {len(subset)} pair(s), {shortest}+ days on.\n"
          f"Your earlier labels are not shown, and the patch order has "
          f"changed.\n")
    try:
        for i, (pair, slot) in enumerate(selected, 1):
            if pair["task_id"] not in tasks:
                tasks[pair["task_id"]] = Task.load(pair["task_id"])
            # Re-randomise the slot for this pass, so a remembered "A was
            # the narrow one" does not carry over as a correct answer.
            shown = dict(pair)
            shown["agent_position"] = slot
            # No token on the retest. By now the judge has run, and its
            # progress output named pairs by token: showing the token here
            # would let a remembered judge verdict attach itself to the
            # pair being relabelled, and inflate the test-retest ceiling
            # that §10 compares the judge against.
            view = _render(shown, tasks[pair["task_id"]],
                           f"{i}/{len(subset)}", show_token=False)

            neutral = _ask_verdict(view)
            rationale = ("not worth labelling" if neutral == SKIP
                         else _ask_rationale())
            pair["relabel"] = _label_entry(shown, neutral, rationale)
            # Per pair, from that pair's own first label — the interval
            # the report quotes has to be the interval that pair had.
            pair["relabel"]["days_after"] = (now - labelled_at[pair["id"]]).days
            _write(path, data)
    except (KeyboardInterrupt, EOFError):
        print("\nstopped")

    done = sum(1 for p in data["pairs"] if p["relabel"])
    print(f"\n{done} pair(s) re-labelled -> {path}")
    return 0


# --- status -----------------------------------------------------------------

def status(path: Path) -> int:
    """Progress only: safe to read while the annotator is still blinded."""
    data = _read(path)
    labelled = [p for p in data["pairs"] if p["human"]]
    usable = [p for p in labelled if p["human"]["neutral"] != SKIP]
    print(f"Human labels: {len(labelled)}/{len(data['pairs'])}; "
          f"{len(labelled) - len(usable)} skipped")
    print("Labels committed: " + ("yes" if _committed_labels(path) else "no"))
    print("Protocol committed: " + ("yes" if _committed_clean(PROTOCOL) else "no"))
    for alias in REGISTERED_JUDGES:
        entries = [p["judge"][alias] for p in usable if alias in p["judge"]]
        cost = sum(e.get("cost_usd", 0.0) for e in entries)
        known = all(e.get("cost_known", False) for e in entries)
        print(f"{alias}: {len(entries)}/{len(usable)} pairs judged; "
              f"recorded cost {'$' if known else '>=$'}{cost:.2f}"
              + (" (some usage missing)" if not known else ""))
    sampled = retest_sample(data)
    done = sum(p["relabel"] is not None for p, _ in sampled)
    print(f"Blind retest: {done}/{len(sampled)} pairs recorded")
    opens = retest_opens(data)
    if opens:
        print(f"Remaining retest opens: {opens:%Y-%m-%d %H:%M:%S} UTC")
    issues = retest_issues(data)
    if done < len(sampled) or not sampled:
        print("Report: blocked — " + "; ".join(issues))
    elif not any(p["judge"] for p in usable):
        print("Report: waiting for judge output")
    else:
        print("Report: blind retest complete; report may be opened")
        if issues:
            print("Protocol violations: " + "; ".join(issues))
    return 0


def _allowances(pairs: list[dict], alias: str) -> tuple[set[int], bool]:
    """Output-token limits used for `alias`, and whether they were recorded.

    A judge that ran before `generation_config` existed leaves no allowance
    in the file. The resume guard assumes DEFAULT_MAX_TOKENS for it, and
    that assumption is fine for refusing a mismatched resume — but it is
    not evidence, and a report that prints it as though it were would be
    stating a number the data does not contain. Hence the flag: callers
    must say "assumed" out loud.
    """
    limits, recorded = set(), True
    for p in pairs:
        config = p["judge"][alias].get("generation_config")
        if config is None:
            recorded = False
            limits.add(DEFAULT_MAX_TOKENS)
        else:
            limits.add(int(config["max_tokens"]))
    return limits, recorded


# --- run --------------------------------------------------------------------

def _progress(result: Any) -> str:
    """What a judge run prints per pair: health, never the verdict.

    The verdict, the agreement and the per-order answers are all in the
    labels file for `report`. On screen they would be read by the person
    who relabels these pairs blind a week later — and a judge's answer
    remembered against a pair is an anchor, the same as a test result.
    What is shown is what someone watching the run needs: did the calls
    parse, did ordering flip the answer, was a patch cut, what did it cost.
    """
    bits = [f"{result.n_parsed}/{result.n_calls} calls parsed"]
    if result.position_bias:
        bits.append("POSITION BIAS")
    if result.n_unparseable:
        bits.append(f"{result.n_unparseable} unparseable")
    limited = sum(c.get("finish_reason") == "length" for c in result.raw)
    if limited:
        bits.append(f"{limited} output-limit stops")
    if result.truncated:
        bits.append("TRUNCATED")
    bits.append(f"${result.cost_usd:.3f}" if result.cost_known
                else f">=${result.cost_usd:.3f} (usage not reported)")
    return " · ".join(bits)


#: Default spend ceiling for one `run` invocation, in USD. Verify live
#: endpoint prices before running: the dearest endpoint changed even
#: between registration and preflight. Stops between pairs and resumes;
#: this is not a combined budget across judges or across invocations.
DEFAULT_JUDGE_BUDGET = 20.0


def run_judge(path: Path, alias: str, repeats: int,
              allow_self: bool, require_committed: bool = True,
              max_cost: float = DEFAULT_JUDGE_BUDGET,
              allow_unregistered: bool = False,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              diagnostic_out: Path | None = None,
              diagnostic_pairs: int = 3) -> int:
    data = _read(path)
    if repeats < 1 or max_tokens < 1 or not math.isfinite(max_cost) or max_cost <= 0:
        print("repeats, max-tokens and max-cost must be positive and finite",
              file=sys.stderr)
        return 1
    if data.get("diagnostic"):
        print("REFUSING: diagnostic output cannot be resumed as a calibration",
              file=sys.stderr)
        return 1
    if diagnostic_out is not None and (
            diagnostic_pairs < 1 or diagnostic_out.resolve() == path.resolve()
            or diagnostic_out.exists()):
        print("REFUSING: probe needs a positive pair count and a new output "
              "path separate from the source labels", file=sys.stderr)
        return 1

    unlabelled = [p for p in data["pairs"] if p["human"] is None]
    if unlabelled:
        # The single most important guard in this file.
        print(f"REFUSING: {len(unlabelled)} pair(s) have no human label.\n"
              f"Label them first. Judging now and labelling afterwards "
              f"measures how\npersuasive the judge is, not whether it is "
              f"right.\n"
              f"    .venv/bin/python scripts/validate_judge.py label {path}",
              file=sys.stderr)
        return 1

    if alias not in MODELS:
        print(f"unknown model {alias!r}; choose from "
              f"{', '.join(sorted(MODELS))}", file=sys.stderr)
        return 1

    if alias not in REGISTERED_JUDGES and not allow_unregistered:
        # A judge picked after the fact is a judge picked for its answer.
        print(f"REFUSING: {alias!r} is not a registered judge. "
              f"judge-protocol.md §9 registers\n"
              f"{', '.join(REGISTERED_JUDGES)}. Using another is an "
              f"amendment (§11): write it, commit it,\nthen add the alias "
              f"to agentcheck.calibration.REGISTERED_JUDGES.",
              file=sys.stderr)
        return 1

    judged_ids = {MODELS[alias].id}
    graded = {p["model"] for p in data["pairs"]}
    overlap = judged_ids & graded
    if overlap and not allow_self:
        print(f"REFUSING: {alias} ({', '.join(sorted(overlap))}) also "
              f"produced patches in\nthis sample. A model rating its own "
              f"output scores it higher — self-preference\nis well "
              f"documented and would be indistinguishable here from a "
              f"calibrated\njudge. Use a different judge model, or pass "
              f"--allow-self-judging and expect\nthe report to flag every "
              f"affected pair.", file=sys.stderr)
        return 1

    # judge-protocol.md §10 makes this an invalidation condition, and it
    # was enforced by nothing but memory: "commit the labels before the
    # judge runs" is exactly the step that gets skipped at eleven at night
    # with the API key already exported. Checked last among the refusals
    # and before the model is built, so a refusal never costs a call.
    labels_hash = _committed_labels(path)
    # The protocol too. An amendment only means something if it is in git
    # before the verdicts it governs, and §11 requires every verdict to
    # say which version it ran under.
    protocol_hash = _committed_clean(PROTOCOL)
    if not protocol_hash and require_committed:
        print(f"REFUSING: {PROTOCOL} has changes that are not committed.\n"
              f"A verdict has to name the protocol version it ran under "
              f"(§11), and an\nuncommitted one has no version. Commit it "
              f"first.", file=sys.stderr)
        return 1
    if not labels_hash and require_committed:
        print(f"REFUSING: {path} has human labels or sampling inputs that "
              f"are not committed.\n"
              f"judge-protocol.md §10: the labels are committed BEFORE the "
              f"judge runs, because\ngit history is the only thing that "
              f"proves they were not written after seeing\nits answers. "
              f"Commit it, then run this again:\n"
              f"    git add {path} && git commit -m 'Record the human labels'",
              file=sys.stderr)
        return 1

    generation_config = {"max_tokens": max_tokens, "repeats_per_order": repeats}
    if diagnostic_out is None:
        for pair in data["pairs"]:
            old = pair["judge"].get(alias)
            if old is None:
                continue
            previous = old.get("generation_config", {
                "max_tokens": DEFAULT_MAX_TOKENS,
                "repeats_per_order": old["n_calls"] // 2})
            if previous != generation_config:
                print("REFUSING: saved judge output uses different token "
                      "limits or repeat counts. Preserve it separately; "
                      "changed settings require a fresh, documented run.",
                      file=sys.stderr)
                return 1
    else:
        # Diagnostics use the first n pairs in the existing seeded sample,
        # regardless of past answers. They never overwrite calibration data.
        for pair in data["pairs"]:
            pair["judge"].pop(alias, None)
        data["diagnostic"] = {
            "source": str(path), "labels_git_hash": labels_hash,
            "generation_config": generation_config,
            "pairs_requested": diagnostic_pairs,
            "selection": "first non-skipped pairs in the saved sample order",
        }

    model = build_model(alias, max_tokens=max_tokens)
    resolved = model_id(model) or MODELS[alias].id
    # Anthropic-only syntax, gated exactly as run_agent gates it: a
    # cache_control block sent to an OpenAI-compatible endpoint is a
    # different content shape, which the provider may reject or mangle.
    cache = MODELS[alias].supports_caching
    tasks: dict[str, Task] = {}
    todo = [p for p in data["pairs"]
            if alias not in p["judge"] and p["human"]["neutral"] != SKIP]
    if diagnostic_out is not None:
        todo = todo[:diagnostic_pairs]
    output = diagnostic_out or path
    if diagnostic_out is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'diagnostic: ' if diagnostic_out is not None else ''}"
          f"judging {len(todo)} pair(s) with {alias} ({resolved}), "
          f"{repeats} repeat(s) per order = {len(todo) * repeats * 2} calls, "
          f"max_tokens={max_tokens}, "
          f"prompt caching {'on' if cache else 'off'}, "
          f"stopping at ${max_cost:.2f}")
    spent = 0.0
    for i, pair in enumerate(todo, 1):
        if pair["task_id"] not in tasks:
            tasks[pair["task_id"]] = Task.load(pair["task_id"])
        task = tasks[pair["task_id"]]

        result = judge_patches(
            model, pair["agent_diff"], task.reference_diff() or "",
            context=(f"The dependency {task.package} was upgraded from "
                     f"{task.from_version} to {task.to_version}."),
            repeats=repeats, cache=cache, model_name=resolved)
        entry = result.as_dict()
        entry["model_id"] = resolved
        entry["self_judged"] = pair["model"] == resolved
        entry["at"] = _now()
        # The exact committed version of the labels this verdict was
        # measured against. Empty only if --allow-uncommitted was passed,
        # and then the report can say so rather than implying otherwise.
        entry["labels_git_hash"] = labels_hash
        entry["protocol_git_hash"] = protocol_hash
        entry["registered_judge"] = alias in REGISTERED_JUDGES
        entry["generation_config"] = generation_config
        pair["judge"][alias] = entry
        # The token, not the run name: the run name carries the model and
        # configuration, and in seven days these same pairs are relabelled
        # blind.
        print(f"  [{i}/{len(todo)}] {_progress(result)}")
        _write(output, data)
        spent += result.cost_usd

        # Checked between pairs, after the result is safely written, so a
        # stop loses nothing and a rerun resumes at the next pair.
        if not result.cost_known:
            print(f"\nSTOPPING: {alias} did not report token usage for pair "
                  f"{_token(pair)}, so spend can no longer be counted and "
                  f"the ${max_cost:.2f} ceiling is not enforcing anything. "
                  f"{i} pair(s) are saved; rerun to continue once usage is "
                  f"reported.", file=sys.stderr)
            return 1
        if result.n_calls >= 10 and result.n_unparseable > result.n_calls / 2:
            print("\nSTOPPING: more than half of this pair's replies could "
                  "not be parsed. Saved all completed pairs; diagnose "
                  "the generation settings before resuming.", file=sys.stderr)
            return 1
        if spent >= max_cost:
            print(f"\nSTOPPING at ${spent:.2f} of a ${max_cost:.2f} ceiling "
                  f"after {i} pair(s). Everything judged so far is saved; "
                  f"rerun with a higher --max-cost to continue.",
                  file=sys.stderr)
            return 1

    total = sum(p["judge"][alias].get("cost_usd", 0.0)
                for p in data["pairs"] if alias in p["judge"])
    print(f"\nspent ${spent:.2f} this run, ${total:.2f} on {alias} in total")
    if diagnostic_out is not None:
        entries = [p["judge"][alias] for p in todo]
        calls = [c for e in entries for c in e["raw"]]
        parsed = sum(e["n_parsed"] for e in entries)
        finishes = Counter(c.get("finish_reason") or "unreported" for c in calls)
        reasoning = [c["reasoning_tokens"] for c in calls
                     if c.get("reasoning_tokens") is not None]
        print(f"Diagnostic: {parsed}/{len(calls)} calls parsed")
        print("Finish reasons: " + ", ".join(f"{k}={v}" for k, v in finishes.items()))
        print(f"Reasoning usage reported on {len(reasoning)}/{len(calls)} calls; "
              f"{sum(reasoning)} reasoning tokens")
        print(f"Calls reporting output above requested cap: "
              f"{sum(c['output_tokens'] > max_tokens for c in calls)}")
        print(f"Saved diagnostic to {output}; source labels unchanged.\n"
              "This probe is excluded from calibration. Review operational "
              "health before registering a full run.")
        return 0 if calls and parsed == len(calls) and not finishes.get("length") else 1
    print(f"wrote {path}\n"
          f"    .venv/bin/python scripts/validate_judge.py status {path}\n"
          "Complete the blind retest before opening the report.")
    return 0


# --- report -----------------------------------------------------------------

def _fmt_interval(label: str, value: float, ci) -> str:
    body = f"{label}: {value:+.2f} [{ci.low:+.2f}, {ci.high:+.2f}]"
    if ci.degenerate:
        body += "  (no spread across resamples — read n, not the bounds)"
    return body


def _agreement_block(out: list[str], title: str, human: list[str],
                     judge: list[str], seed: int) -> None:
    out.append(f"### {title}")
    out.append("")
    try:
        a = agree(human, judge, seed=seed)
    except ValueError:
        out.append(f"No comparable verdicts; all {len(human)} pairs excluded.")
        out.append("")
        return
    out.append(f"- pairs compared: **{a.n}** "
               f"({a.n_excluded} excluded as non-verdicts)")
    out.append(f"- raw agreement: {a.raw:.0%}")
    out.append(f"- Cohen's kappa: **{a.kappa:.2f}** ({kappa_band(a.kappa)}), "
               f"95% CI [{a.kappa_interval.low:.2f}, "
               f"{a.kappa_interval.high:.2f}]"
               + ("  — degenerate, see note" if a.kappa_interval.degenerate
                  else ""))
    out.append(f"- Gwet's AC1: {a.ac1:.2f}   PABAK: {a.pabak:.2f}")
    out.append(f"- modal class prevalence (human): {a.prevalence:.0%}")
    out.append("")
    if a.prevalence > 0.75:
        out.append(f"> At {a.prevalence:.0%} prevalence kappa is substantially "
                   f"deflated and AC1 substantially inflated. Read the "
                   f"confusion matrix; neither coefficient alone describes "
                   f"this sample.")
        out.append("")


def _matrix_block(out: list[str], human: list[str], judge: list[str]) -> None:
    labels = sorted(set(human) | set(judge))
    counts = Counter(zip(human, judge, strict=True))
    out.append("| human \\ judge | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for h in labels:
        row = [f"**{h}**"] + [str(counts.get((h, j), 0)) for j in labels]
        out.append("| " + " | ".join(row) + " |")
    out.append("")


def report(path: Path, alias: str | None, seed: int,
           write_to: Path | None) -> int:
    data = _read(path)
    if data.get("diagnostic"):
        print("REFUSING: a diagnostic probe is not a calibration report",
              file=sys.stderr)
        return 1

    aliases = sorted({a for p in data["pairs"] for a in p["judge"]})
    if not aliases:
        print("no judge has run on this file yet", file=sys.stderr)
        return 1
    if alias and alias not in aliases:
        print(f"no verdicts from {alias!r}; have {', '.join(aliases)}",
              file=sys.stderr)
        return 1

    sampled = retest_sample(data)
    remaining = sum(p["relabel"] is None for p, _ in sampled)
    if remaining or not sampled:
        print(f"REFUSING: {remaining}/{len(sampled)} blind retest pairs remain.\n"
              "judge-protocol.md §7 requires a blind retest after seven "
              "days. Opening judge verdicts now would anchor that pass.\n"
              f"Use `status {path}` for progress without verdicts.",
              file=sys.stderr)
        return 1

    out: list[str] = [
        "# Judge calibration",
        "",
        f"Generated by `scripts/validate_judge.py report` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d}. Do not edit by hand.",
        "",
        f"Protocol: [`{data['protocol']}`]({Path(data['protocol']).name}) "
        f"(recorded blob `{data['protocol_git_hash'][:12]}`).",
        "",
        "Blob identifiers alone do not prove event ordering. Verify "
        "chronology separately before treating the study as preregistered; "
        "see the protocol's release-provenance note.",
        "",
    ]
    # §11: a calibration run under an amended protocol says which version
    # it followed. Verdicts carry the hash they ran under; if that is not
    # the version the labels were written under, both are named.
    ran_under = sorted({e.get("protocol_git_hash") or "unrecorded"
                        for p in data["pairs"] for e in p["judge"].values()})
    amended = [h for h in ran_under if h != data["protocol_git_hash"]]
    if amended:
        out += [
            f"**Judged under an amended protocol:** "
            f"{', '.join(f'`{h[:12]}`' for h in amended)}. These identifiers "
            f"differ from the labels' recorded protocol; see §11 for the "
            f"amendment and provenance limits.",
            "",
        ]
    unregistered = sorted({alias for p in data["pairs"]
                           for alias, e in p["judge"].items()
                           if e.get("registered_judge") is False})
    if unregistered:
        out += [
            f"**Unregistered judge(s):** {', '.join(unregistered)}. Not "
            f"named in §9; their verdicts are reported but are not the "
            f"specified calibration.",
            "",
        ]
    out += [
        "## Sample",
        "",
        f"- pairs labelled: **{sum(1 for p in data['pairs'] if p['human'])}** "
        f"of {len(data['pairs'])} drawn",
        f"- sampling frame: {data['frame']['frame_size']} candidate pairs",
        f"- seed: {data['frame']['seed']}, floor "
        f"{data['frame']['floor']} per stratum",
        f"- built: {data['created']}",
        "",
        "| stratum (task, outcome class, cheated) | available | drawn |",
        "|---|---|---|",
    ]
    for key, counts in data["frame"]["strata"].items():
        out.append(f"| `{key}` | {counts['available']} | {counts['drawn']} |")
    out.append("")

    labelled = [p for p in data["pairs"]
                if p["human"] and p["human"]["neutral"] != SKIP]
    if len(labelled) < MIN_PAIRS:
        out.append(f"> **PILOT, not a calibration.** {len(labelled)} "
                   f"labelled pairs against the {MIN_PAIRS} the protocol "
                   f"requires (clause 10). Every coefficient below has an "
                   f"interval wide enough to span two agreement bands, and "
                   f"none of it licenses quoting judge verdicts. Grow the "
                   f"archive and rebuild.")
        out.append("")
    skipped = sum(1 for p in data["pairs"]
                  if p["human"] and p["human"]["neutral"] == SKIP)
    if skipped:
        out.append(f"{skipped} pair(s) marked `skip` and excluded: no patch "
                   f"worth judging.")
        out.append("")

    # --- the human's own reliability, before the judge is discussed ---------
    out.append("## The annotator")
    out.append("")
    positions = [p["agent_position"] for p in labelled]
    human_rel = [p["human"]["agent_relative"] for p in labelled]
    try:
        pb = position_bias(positions, human_rel)
        out.append(
            f"**Position bias.** The agent's patch was shown first in "
            f"{pb.n_first} pairs and second in {pb.n_second}. Deficient "
            f"verdicts: {pb.deficient_first}/{pb.n_first} vs "
            f"{pb.deficient_second}/{pb.n_second}; difference "
            f"{pb.difference.point:+.0%} [{pb.difference.low:+.0%}, "
            f"{pb.difference.high:+.0%}] 95% Newcombe — "
            + ("**bias detected**, and every number below inherits it."
               if pb.detected else
               "the interval contains 0, so no position effect is "
               "established at this n. That is not the same as none."))
        out.append("")
    except ValueError as e:
        out.append(f"Position bias could not be estimated: {e}")
        out.append("")

    retest = [p for p, _ in sampled if p["relabel"]
              and p["relabel"]["neutral"] != SKIP]
    human_ceiling = None
    if retest:
        first = [p["human"]["agent_relative"] for p in retest]
        again = [p["relabel"]["agent_relative"] for p in retest]
        gaps = [(datetime.fromisoformat(p["relabel"]["at"])
                 - datetime.fromisoformat(p["human"]["at"])).days
                for p in retest]
        days = (str(min(gaps)) if min(gaps) == max(gaps)
                else f"{min(gaps)}–{max(gaps)}")
        _agreement_block(out, f"Test-retest, n={len(retest)}, {days} days "
                              f"apart (7-class)", first, again, seed)
        _agreement_block(out, "Test-retest, collapsed binary",
                         collapsed(first), collapsed(again), seed)
        try:
            human_ceiling = agree(collapsed(first), collapsed(again),
                                  seed=seed).kappa
        except ValueError:
            human_ceiling = None  # All retest answers were non-verdicts.
        out.append(
            "> This is **intra-rater** agreement from a single annotator, "
            "not the inter-annotator figure RoadmapBench reports as 0.83. "
            "It is weaker in two specific ways: one person's blind spots "
            "are shared across both passes, so a consistent "
            "misunderstanding raises it rather than lowering it; and a "
            f"{days}-day gap is short enough that some recall survives. "
            "It bounds the judge from above and nothing more.")
        out.append("")
    else:
        out.append("No test-retest pass yet — run `relabel` after "
                   f"{RELABEL_DELAY_DAYS} days. Without it the judge's "
                   "agreement has no human ceiling to be read against.")
        out.append("")

    # --- the judge ----------------------------------------------------------
    for a in ([alias] if alias else aliases):
        pairs = [p for p in labelled if a in p["judge"]]
        if not pairs:
            continue
        model_ids = {p["judge"][a]["model_id"] for p in pairs}
        self_judged = sum(1 for p in pairs if p["judge"][a]["self_judged"])

        out.append(f"## Judge: `{a}` ({', '.join(sorted(model_ids))})")
        out.append("")
        limits, recorded = _allowances(pairs, a)
        out.append("Requested output token limit: "
                   + ", ".join(str(v) for v in sorted(limits)) + ".")
        if not recorded:
            out.append("")
            out.append(f"> This judge ran before the generation config was "
                       f"stored, so its allowance is **not recorded in the "
                       f"data**: {DEFAULT_MAX_TOKENS} is what the resume "
                       f"guard assumes, not what the file proves. Its calls "
                       f"also carry no finish reason, so a truncated reply "
                       f"cannot be distinguished from a short one.")
        out.append("")
        if self_judged:
            out.append(f"> **{self_judged} of {len(pairs)} pairs were "
                       f"self-judged** — this model produced the patch it "
                       f"is grading. Self-preference inflates those "
                       f"verdicts and they are not separable here.")
            out.append("")

        human = [p["human"]["agent_relative"] for p in pairs]
        verdicts = [p["judge"][a]["verdict"] for p in pairs]

        _agreement_block(out, "Collapsed binary (specified headline)",
                         collapsed(human), collapsed(verdicts), seed)
        try:
            headline = agree(collapsed(human), collapsed(verdicts), seed=seed)
        except ValueError:
            out.append("**Calibration inconclusive:** no comparable binary "
                       "verdicts. No judge findings are licensed.")
            out.append("")
            continue
        _agreement_block(out, "Seven-class (reported beside it)",
                         human, verdicts, seed)

        out.append("### Confusion, human -> judge")
        out.append("")
        _matrix_block(out, human, verdicts)

        stable = sum(1 for p in pairs if p["judge"][a].get("is_stable"))
        biased = sum(1 for p in pairs if p["judge"][a].get("position_bias"))
        unread = sum(1 for p in pairs
                     if p["judge"][a].get("n_unparseable", 0))
        n = len(pairs)
        out.append("### Judge self-consistency")
        out.append("")
        for label_, k in (("unanimous and order-independent", stable),
                          ("position bias detected", biased),
                          ("any unparseable reply", unread)):
            ci = rate(k, n)
            out.append(f"- {label_}: {k}/{n} = {ci.point:.0%} "
                       f"[{ci.low:.0%}, {ci.high:.0%}] 95% Wilson")
        out.append("")

        name, licence = threshold_verdict(headline.kappa)
        out.append("### Verdict against the specified threshold")
        out.append("")
        out.append(f"Headline kappa **{headline.kappa:.2f}** "
                   f"[{headline.kappa_interval.low:.2f}, "
                   f"{headline.kappa_interval.high:.2f}] -> numerical band "
                   f"**{name}**.")
        out.append("")
        invalid = retest_issues(data)
        if len(labelled) < MIN_PAIRS:
            invalid.append(f"fewer than {MIN_PAIRS} non-skipped human labels")
        if len(pairs) != len(labelled):
            invalid.append(f"judge coverage is incomplete ({len(pairs)}/{len(labelled)})")
        if not any(c["stratum"][2] for c in data["frame"]["candidates"]):
            invalid.append("no cheating stratum in the recorded sampling frame")
        if len(retest) < RELABEL_N:
            invalid.append(f"fewer than {RELABEL_N} non-skipped retest labels")
        if human_ceiling is None:
            invalid.append("human test-retest kappa is unavailable")
        elif human_ceiling < headline.kappa:
            invalid.append(f"human test-retest kappa ({human_ceiling:.2f}) is "
                           "below the judge's kappa (§10)")
        if any(not p["judge"][a].get("labels_git_hash") or
               not p["judge"][a].get("protocol_git_hash") for p in pairs):
            invalid.append("committed labels/protocol provenance is missing")
        if a not in REGISTERED_JUDGES or any(
                p["judge"][a].get("registered_judge") is False for p in pairs):
            invalid.append("judge is not registered under the protocol")
        if any(p["judge"][a]["n_calls"] != 10 for p in pairs):
            invalid.append("judge calls do not match five repeats in both orders")
        if invalid:
            out.append("**Protocol requirements unmet; no judge findings are "
                       "licensed.** " + "; ".join(invalid) + ".")
        else:
            out.append(f"{licence}.")
        out.append("")

        disagreements = [p for p in pairs
                         if collapsed([p["human"]["agent_relative"]])[0]
                         != collapsed([p["judge"][a]["verdict"]])[0]]
        out.append(f"### Disagreements ({len(disagreements)})")
        out.append("")
        out.append("Read these to understand disagreements. Preserve the "
                   "committed human labels; any later adjudication belongs "
                   "in a separate analysis.")
        out.append("")
        for p in disagreements:
            j = p["judge"][a]
            out.append(f"**`{p['run']}`** — agent patch shown as "
                       f"patch {p['agent_position'].upper()}")
            out.append("")
            out.append(f"- human: `{p['human']['agent_relative']}` — "
                       f"{p['human']['rationale']}")
            out.append(f"- judge: `{j['verdict']}` "
                       f"(agreement {j['agreement']:.0%} over "
                       f"{j['n_calls']} calls)")
            reasoning = (j.get("raw") or [{}])[0].get("reasoning", "")
            if reasoning:
                out.append(f"- judge said: {reasoning[:400]}")
            out.append("")

    if alias is None and all(a in aliases for a in REGISTERED_JUDGES):
        left, right = REGISTERED_JUDGES
        common = [p for p in labelled if left in p["judge"] and right in p["judge"]]
        lim_l, rec_l = _allowances(common, left)
        lim_r, rec_r = _allowances(common, right)
        out.extend(["## Judge-model sensitivity", "",
                    f"Both judges evaluated {len(common)}/{len(labelled)} "
                    "non-skipped pairs. This comparison mixes vendor and "
                    "model scale (see §11, A1); it does not isolate "
                    "either.", ""])
        if lim_l != lim_r:
            out.extend([
                f"> **The two judges also ran under different output "
                f"allowances** — `{left}` at "
                f"{', '.join(str(v) for v in sorted(lim_l))}"
                f"{'' if rec_l else ' (assumed, not recorded)'} and "
                f"`{right}` at "
                f"{', '.join(str(v) for v in sorted(lim_r))}"
                f"{'' if rec_r else ' (assumed, not recorded)'}. "
                f"Any disagreement below therefore confounds vendor, scale "
                f"and allowance together, and cannot be read as a property "
                f"of either model.", ""])
        _agreement_block(out, f"{left} vs {right}, collapsed binary",
                         collapsed([p["judge"][left]["verdict"] for p in common]),
                         collapsed([p["judge"][right]["verdict"] for p in common]),
                         seed)

    text = "\n".join(out)
    print(text)
    target = write_to or CALIBRATION_REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"\n(written to {target})", file=sys.stderr)
    return 0


# --- cli --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="phase", required=True)

    s = sub.add_parser("status", help="progress without revealing verdicts")
    s.add_argument("path", type=Path)

    b = sub.add_parser("build", help="stratified sample of pairs to label")
    b.add_argument("paths", nargs="+")
    b.add_argument("--out", type=Path, default=Path("judge-labels.json"))
    b.add_argument("-n", type=int, default=48,
                   help="pairs to draw (default 48: above the 40 the "
                        "objectives call for, and affordable to label)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--floor", type=int, default=2,
                   help="minimum draws per non-empty stratum")

    l = sub.add_parser("label", help="label them yourself, first")
    l.add_argument("path", type=Path)
    l.add_argument("--dry-run", action="store_true",
                   help="render one pair's view and exit, recording nothing")

    rl = sub.add_parser("relabel", help="re-label a subset, days later")
    rl.add_argument("path", type=Path)
    rl.add_argument("-n", type=int, default=RELABEL_N)
    rl.add_argument("--force", action="store_true",
                    help="re-label before the delay has elapsed, and report "
                         "the actual interval")

    sh = sub.add_parser("sheet", help="write the pairs to one document to "
                                      "read away from the terminal")
    sh.add_argument("path", type=Path)
    sh.add_argument("--out", type=Path, default=Path("judge-pairs.md"))
    sh.add_argument("--all", action="store_true", dest="include_labelled",
                    help="include pairs already labelled")
    sh.add_argument("--force", action="store_true",
                    help="overwrite a sheet holding verdicts that have not "
                         "been imported")

    im = sub.add_parser("import", help="record the verdicts written into a "
                                       "sheet")
    im.add_argument("sheet", type=Path)
    im.add_argument("--into", type=Path, default=Path("judge-labels.json"))

    r = sub.add_parser("run", help="have the judge label the same pairs")
    r.add_argument("path", type=Path)
    r.add_argument("--model", required=True,
                   choices=sorted(MODELS),
                   help="a registered judge: "
                        + ", ".join(REGISTERED_JUDGES))
    r.add_argument("--repeats", type=int, default=5,
                   help="per order; every repeat is position-swapped, so "
                        "this is 2x calls per pair")
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="requested output token limit; recorded with every pair")
    r.add_argument("--allow-self-judging", action="store_true")
    r.add_argument("--max-cost", type=float, default=DEFAULT_JUDGE_BUDGET,
                   help="stop between pairs once this invocation has spent "
                        "this many USD (default %(default).2f)")
    r.add_argument("--allow-unregistered-judge", action="store_true",
                   help="run a judge the protocol does not register; every "
                        "verdict is marked, and the report flags it")
    r.add_argument("--allow-uncommitted", action="store_true",
                   help="judge a labels file with uncommitted changes; every "
                        "verdict records an empty labels hash, and §10 "
                        "treats the calibration as unverifiable")

    probe = sub.add_parser("probe", help="small diagnostic without changing labels")
    probe.add_argument("path", type=Path)
    probe.add_argument("--model", required=True, choices=sorted(REGISTERED_JUDGES))
    probe.add_argument("--max-tokens", required=True, type=int)
    probe.add_argument("--pairs", type=int, default=3)
    probe.add_argument("--repeats", type=int, default=1)
    probe.add_argument("--max-cost", type=float, default=2.0)
    probe.add_argument("--out", type=Path, required=True)

    o = sub.add_parser("report", help="agreement, kappa, disagreements")
    o.add_argument("path", type=Path)
    o.add_argument("--model", default=None,
                   help="report one judge model; default is all of them")
    o.add_argument("--seed", type=int, default=0,
                   help="bootstrap seed for the kappa intervals")
    o.add_argument("--out", type=Path, default=None)

    args = p.parse_args()
    if args.phase == "status":
        return status(args.path)
    if args.phase == "build":
        return build(args.paths, args.out, args.n, args.seed, args.floor)
    if args.phase == "label":
        return label(args.path, args.dry_run)
    if args.phase == "import":
        return import_sheet(args.sheet, args.into)
    if args.phase == "sheet":
        return sheet(args.path, args.out, args.include_labelled,
                     args.force)
    if args.phase == "relabel":
        return relabel(args.path, args.n, args.force)
    if args.phase == "run":
        return run_judge(args.path, args.model, args.repeats,
                         args.allow_self_judging,
                         require_committed=not args.allow_uncommitted,
                         max_cost=args.max_cost,
                         max_tokens=args.max_tokens,
                         allow_unregistered=args.allow_unregistered_judge)
    if args.phase == "probe":
        return run_judge(args.path, args.model, args.repeats, allow_self=False,
                         max_cost=args.max_cost, max_tokens=args.max_tokens,
                         diagnostic_out=args.out, diagnostic_pairs=args.pairs)
    return report(args.path, args.model, args.seed, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
