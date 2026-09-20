#!/usr/bin/env python3
"""Move records that no longer load into `runs/archive/`, with a manifest.

    python3 scripts/archive_runs.py runs/*.json          # dry run
    python3 scripts/archive_runs.py runs/*.json --apply

28 of the first 44 records this project produced were written under schema
v2 or earlier and cannot be loaded by the current harness. `record.load`
rejects them, correctly and loudly, and that behaviour does not change
here. But a rejection block longer than the results is a rejection block
nobody reads, and these files are evidence rather than mess: they are the
clearest demonstration in the repo that the schema discipline is
load-bearing.

So they move out of the default glob and into `runs/archive/`, and a
generated manifest records what each one was missing and which commit
wrote it.

They are NOT migrated. The absent fields are `before_failed_ids` (the
failing set measured in the container before the agent started) and
`verdict_status` (whether the final test run produced a verdict at all).
Neither can be recovered from what was saved, and inventing them is
exactly the fabrication the schema exists to prevent: a run with no
recorded before-state, given an empty one, scores 100%. That happened. It
is bug four on the list in docs/findings.md §1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck.record import SCHEMA_VERSION, IncompatibleRecord, load

ARCHIVE = Path("runs/archive")
MANIFEST = ARCHIVE / "MANIFEST.md"


def _adding_commit(path: Path) -> tuple[str, str]:
    """(short sha, date) of the commit that added this file, or ("", "").

    `--diff-filter=A` finds the addition rather than the last touch. A
    record's provenance is the commit it arrived in; later commits that
    reformatted the tree say nothing about what wrote it.
    """
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%h\t%ad",
             "--date=short", "-1", "--", str(path)],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "", ""
    line = out.stdout.strip()
    if not line or "\t" not in line:
        return "", ""
    sha, date = line.split("\t", 1)
    return sha.strip(), date.strip()


def superseded(paths: list[Path]) -> dict[Path, str]:
    """Records made under a configuration that a later run replaced.

    A `config_version` covers the model, prompt, tools, caps, test command,
    image digest and route. Change any of them and the old records describe
    a different experiment — still true, still worth keeping, but not
    poolable with the new ones and not a substitute for them.

    That is not hypothetical. Removing a provider pin changed the route and
    therefore the stamp; the sweep, which resumes on "does this cell hold a
    measurement", skipped 42 cells from the previous experiment and ran
    only the 22 that were missing. The result was a ladder of sixteen
    configurations with at most seven runs each, where eight rungs at n=8
    were intended. Nothing failed — the numbers were simply spread across
    twice as many columns as anyone wanted.

    Superseded is decided per (task, iteration cap): whichever config wrote
    the most recent record for that cell is current, and the rest are not.
    Comparing globally would be wrong, because a task built later has a
    later timestamp without anything having been superseded.
    """
    newest: dict[tuple[str, int], tuple[float, str]] = {}
    loaded: list[tuple[Path, tuple[str, int], str]] = []
    for path in paths:
        try:
            record = load(path)
        except IncompatibleRecord:
            continue
        key = (record.task_id, record.max_iterations)
        stamp = path.stat().st_mtime
        loaded.append((path, key, record.config_version))
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, record.config_version)

    return {path: config for path, key, config in loaded
            if config != newest[key][1]}


def _rejection(path: Path) -> str | None:
    """The loader's own reason, or None if the record loads fine."""
    try:
        load(path)
    except IncompatibleRecord as e:
        # Strip the leading "<filename>: " the loader prefixes, since the
        # manifest already has a filename column.
        reason = str(e)
        prefix = f"{path.name}: "
        return reason[len(prefix):] if reason.startswith(prefix) else reason
    return None


def _schema_of(reason: str) -> str:
    """The schema version named in the loader's message, for grouping."""
    marker = "schema v"
    if marker not in reason:
        return "unknown"
    tail = reason.split(marker, 1)[1]
    return tail.split(" ", 1)[0].strip()


def _missing_from(reason: str) -> str:
    """The missing-field list the loader reported, verbatim."""
    if "missing " not in reason:
        return "—"
    tail = reason.split("missing ", 1)[1]
    return tail.split(".", 1)[0].strip().strip("[]").replace("'", "")


def render_manifest(rows: list[dict[str, str]]) -> str:
    """The manifest text. Generated, so it cannot drift from the files."""
    by_schema = Counter(r["schema"] for r in rows)
    spread = ", ".join(f"v{k}: {n}" for k, n in sorted(by_schema.items()))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "# Archived run records",
        "",
        f"Generated by `scripts/archive_runs.py` on {stamp}. "
        f"Do not edit by hand.",
        "",
        f"{len(rows)} records that do not load under the current schema "
        f"(v{SCHEMA_VERSION}). Breakdown: {spread}.",
        "",
        "## Why these are not migrated",
        "",
        "Every one of them is missing `before_failed_ids`, the set of tests",
        "failing in the container *before* the agent ran, and most are also",
        "missing `verdict_status`, which says whether the final test run",
        "produced a verdict at all.",
        "",
        "These fields cannot be reconstructed. Progress is",
        "`|before_failed − after_failed| / |before_failed|`, so supplying an",
        "empty before-state does not produce a slightly wrong number — it",
        "produces 100%, for every run, silently. That is bug four in",
        "`docs/findings.md` §1: a re-scoring script passed an empty",
        "before-state and stamped seven runs with a fabricated result.",
        "",
        "They are kept because they are the evidence that the schema check",
        "does something. A loader that had tolerated the missing fields",
        "would have produced 28 plausible numbers instead of 28 rejections.",
        "",
        "## Records",
        "",
        "| record | schema | missing | added in | date |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["name"]):
        lines.append(
            f"| `{r['name']}` | {r['schema']} | `{r['missing']}` | "
            f"{r['sha'] or '—'} | {r['date'] or '—'} |")
    lines.append("")
    return "\n".join(lines)


def archive_superseded(raw_paths: list[str], apply: bool) -> int:
    paths = [Path(p) for p in raw_paths
             if Path(p).is_file() and ARCHIVE not in Path(p).parents]
    stale = superseded(paths)
    if not stale:
        print("nothing superseded — every cell's records share one "
              "configuration")
        return 0

    by_config: dict[str, int] = {}
    for config in stale.values():
        by_config[config] = by_config.get(config, 0) + 1

    print(f"{len(stale)} record(s) belong to a superseded configuration:")
    for config, count in sorted(by_config.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3} under cfg {config}")

    if not apply:
        print(f"\ndry run; pass --apply to move them into {ARCHIVE}/")
        return 0

    target = ARCHIVE / "superseded"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(stale):
        path.rename(target / path.name)
    print(f"\nmoved {len(stale)} record(s) into {target}/")
    print("They are kept, not deleted: a superseded run is a true record "
          "of a\ndifferent experiment, and the harness-failure rate it "
          "carries is a result.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+")
    p.add_argument("--apply", action="store_true",
                   help="actually move the files (default is a dry run)")
    p.add_argument("--superseded", action="store_true",
                   help="archive records whose configuration a later run "
                        "replaced, rather than records that fail to load")
    args = p.parse_args()

    if args.superseded:
        return archive_superseded(args.paths, args.apply)

    rows: list[dict[str, str]] = []
    already_archived = 0
    for raw in args.paths:
        path = Path(raw)
        if not path.is_file():
            continue
        if ARCHIVE in path.parents:
            already_archived += 1
            continue
        reason = _rejection(path)
        if reason is None:
            continue
        sha, date = _adding_commit(path)
        rows.append({"name": path.name, "schema": _schema_of(reason),
                     "missing": _missing_from(reason), "sha": sha,
                     "date": date, "path": str(path)})

    if not rows:
        if already_archived:
            print(f"nothing to do — all {already_archived} record(s) given "
                  f"are already in {ARCHIVE}/")
        else:
            print("nothing to archive — every record given loads")
        return 0

    print(f"{len(rows)} record(s) do not load under schema v{SCHEMA_VERSION}:")
    for r in rows:
        print(f"  {r['name']}  (schema {r['schema']})")

    if not args.apply:
        print("\ndry run; pass --apply to move them into runs/archive/")
        return 0

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for r in rows:
        Path(r["path"]).rename(ARCHIVE / r["name"])
    MANIFEST.write_text(render_manifest(rows))
    print(f"\nmoved {len(rows)} record(s) into {ARCHIVE}/")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
