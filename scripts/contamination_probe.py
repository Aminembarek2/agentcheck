#!/usr/bin/env python3
"""Ask a model to describe a migration it has never been shown.

    .venv/bin/python scripts/contamination_probe.py --models ds-flash,haiku
    .venv/bin/python scripts/contamination_probe.py --task 002-databases

Every task in this repository is built from an old, public, merged pull
request. A model that reproduces the maintainer's patch may be recalling it
rather than deriving it, and the harness cannot tell those apart from
inside the container: both look like a correct fix.

`task.yaml` declares a contamination risk per task, but that declaration is
an argument — the migration is old, the guide is famous, the repository is
popular. This script is the falsifiable version. It asks each model, with
no repository access and no tools, to describe the upgrade and to NAME THE
FILES the maintainer's PR touched. Naming `databases/backends/sqlite.py`
unprompted is not inference; it is recall, and it is checkable against
`reference_files` in the task definition.

Two things this cannot do, stated so the output is not overread:

  * It cannot prove absence. A model that names no files may still have
    seen the PR and be unable to retrieve it under this prompt.
  * A correct general description is weak evidence. The pydantic 1 -> 2
    renames are in every tutorial; knowing them says little. Specific FILE
    PATHS in a specific repository are the strong signal, which is why the
    score below is file overlap and not prose quality.

Output is verbatim. The reply text goes into `docs/contamination.md` in
full, because a summary of a probe is a probe nobody can check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck.models import MODELS, build_model, model_id
from agentcheck.task import Task

REPORT = Path("docs/contamination.md")

#: Deliberately gives the repository and the version pair and nothing else.
#: Naming the files would be the answer; describing the symptom would let a
#: model derive them. The question is what it already knows.
PROMPT = """\
You are being asked what you already know, with no access to any repository.

Repository: {repo}
Dependency upgrade: {package} {from_version} -> {to_version}

Answer in JSON with exactly these keys:

  "familiar": true or false — have you seen this specific repository's
              migration for this dependency?
  "description": what the migration required, in two or three sentences.
  "files": a list of source file PATHS in that repository that the
           migration changed. Give real paths as you remember them, or an
           empty list if you do not remember any. Do not guess plausible
           paths; an empty list is a better answer than an invented one.
  "confidence": "high", "medium" or "low", for the file list only.

Return only the JSON object."""


def parse_reply(text: str) -> dict:
    """Pull the JSON object out of a reply, or record that we could not.

    An unparseable reply is never coerced into a result. A probe that
    silently reads "no files" out of prose it failed to parse would report
    a clean contamination check for a model that may have listed every
    file.
    """
    fenced = text.strip()
    if "```" in fenced:
        parts = fenced.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                fenced = candidate
                break
    start, end = fenced.find("{"), fenced.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(fenced[start:end + 1])


def overlap(named: list[str], reference: tuple[str, ...]) -> list[str]:
    """Which of the maintainer's files the model named, by basename.

    Basename rather than full path because a model recalling
    `backends/sqlite.py` for `databases/backends/sqlite.py` has recalled
    it; requiring the exact prefix would understate recall and flatter the
    task. The full strings are printed alongside so a reader can judge.
    """
    wanted = {Path(f).name: f for f in reference}
    return sorted({wanted[Path(n).name] for n in named
                   if Path(n).name in wanted})


def probe(task: Task, alias: str) -> dict:
    model = build_model(alias)
    reply = model.invoke(PROMPT.format(
        repo=task.repo, package=task.package,
        from_version=task.from_version, to_version=task.to_version))
    text = reply.content if isinstance(reply.content, str) else str(reply.content)

    entry: dict[str, Any] = {
        "task": task.id, "model": alias, "model_id": model_id(model),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw": text,
        "reference_files": list(task.reference_files),
        "declared_risk": task.contamination_risk,
    }
    try:
        parsed = parse_reply(text)
    except (ValueError, json.JSONDecodeError) as e:
        entry["parsed"] = None
        entry["parse_error"] = str(e)
        return entry

    named = [str(f) for f in (parsed.get("files") or [])]
    entry["parsed"] = parsed
    entry["named_files"] = named
    entry["recalled"] = overlap(named, task.reference_files)
    return entry


def render(entries: list[dict]) -> str:
    out = [
        "# Contamination probes",
        "",
        f"Generated by `scripts/contamination_probe.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d}. Do not edit by hand.",
        "",
        "Each model was asked, with no repository access and no tools, to "
        "describe a migration and name the files the maintainer's PR "
        "touched. Naming them is recall, not inference.",
        "",
        "Read this as evidence in one direction only: files recalled are "
        "evidence of contamination, files not recalled are **not** evidence "
        "of its absence. A model may have seen the PR and simply not "
        "retrieved it under this prompt.",
        "",
        "| task | model | declared risk | files recalled | of |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        recalled = e.get("recalled")
        cell = ("unparseable" if e.get("parsed") is None
                else f"**{len(recalled)}**" if recalled else "0")
        out.append(f"| `{e['task']}` | {e['model']} | {e['declared_risk']} | "
                   f"{cell} | {len(e['reference_files'])} |")
    out.append("")

    for e in entries:
        out.append(f"## `{e['task']}` — {e['model']} (`{e['model_id']}`)")
        out.append("")
        if e.get("recalled"):
            out.append(f"**Recalled {len(e['recalled'])} of "
                       f"{len(e['reference_files'])} files the PR touched:**")
            out.append("")
            for f in e["recalled"]:
                out.append(f"- `{f}`")
            out.append("")
        elif e.get("parsed") is None:
            out.append(f"Reply could not be parsed: {e['parse_error']}")
            out.append("")
        else:
            out.append("Named no file that the PR touched.")
            out.append("")
        out.append("<details><summary>verbatim reply</summary>")
        out.append("")
        out.append("```")
        out.append(e["raw"].strip())
        out.append("```")
        out.append("")
        out.append("</details>")
        out.append("")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default="ds-flash",
                   help="comma-separated model aliases")
    p.add_argument("--task", default=None,
                   help="one task id; default is every task")
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args()

    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    unknown = [a for a in aliases if a not in MODELS]
    if unknown:
        print(f"unknown model(s) {unknown}", file=sys.stderr)
        return 1

    task_ids = [args.task] if args.task else Task.available()
    entries = []
    for task_id in task_ids:
        task = Task.load(task_id)
        if not task.reference_files:
            print(f"skipping {task_id}: no reference_files recorded, so "
                  f"there is nothing to check recall against", file=sys.stderr)
            continue
        for alias in aliases:
            print(f"probing {task_id} with {alias}...", file=sys.stderr)
            entry = probe(task, alias)
            entries.append(entry)
            n = len(entry.get("recalled") or [])
            print(f"  recalled {n}/{len(task.reference_files)} file(s)",
                  file=sys.stderr)

    if not entries:
        print("nothing probed", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(entries))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
