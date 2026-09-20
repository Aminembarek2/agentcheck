#!/usr/bin/env python3
"""Record the two artifacts that make a task scoreable.

    .venv/bin/python scripts/prepare_task.py 002-databases --build
    .venv/bin/python scripts/prepare_task.py 002-databases --broken
    .venv/bin/python scripts/prepare_task.py 002-databases --reference
    .venv/bin/python scripts/prepare_task.py 002-databases --coverage
    .venv/bin/python scripts/prepare_task.py 002-databases --all

The last two are optional inputs to scoring, and when either is missing the
scorer says so in its findings rather than assuming the permissive answer.
This script is how you stop it having to.

It also builds the image and records the broken state, which used to live
in a shell script with its own grep/cut YAML parser — a third description
of the same test command and env, maintained by hand alongside task.yaml
and a TASKS dict in run_agent.py. Recording the broken state through the
same Sandbox and Tools a live run uses means the recorded baseline and the
measured one cannot drift apart through two different readers of the same
report format.

reference.diff — the maintainer's own change, fetched from the PR named in
task.yaml. Without it, "the agent edited a test file" is unjudgeable: task
002 upgrades SQLAlchemy to 2.0, which removed the list form of select(),
and six of its 55 failures are raised INSIDE test bodies by
`select([notes.c.text])`. They cannot be fixed from source at all. A run
that made exactly the five-line change the maintainer made was scored
credible_progress 0.0 because a blanket rule called every test edit a
cheat. The maintainer's diff is the standard that rule was missing.

coverage.json — which lines the test command actually executes, recorded
from the GREEN baseline. It replaces a hand-maintained list of unreachable
path prefixes that could be wrong in both directions and was blind to dead
branches inside reachable files. Every observed run edits
databases/core.py, and a path-level list called all of that verified work.

The coverage container runs on the DEFAULT bridge network, not the
isolated one, because it has to pip-install the baseline pins. That is
safe here and nowhere else: no agent is involved in this step. Agent runs
always use the internal network — see sandbox.ensure_network.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from agentcheck.task import Task, TaskError

_PR_URL = re.compile(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<n>\d+)")


# --- build ------------------------------------------------------------------

def build_image(task: Task, force: bool = False) -> int:
    """Build the task container from task.yaml."""
    cmd = ["docker", "build",
           "--build-arg", f"PY_VERSION={task.python}",
           "--build-arg", f"REPO_URL={task.repo}",
           "--build-arg", f"BASE_SHA={task.base_sha}",
           "-t", task.image,
           "-f", str(Path(__file__).resolve().parent.parent / "Dockerfile"),
           str(task.root)]
    if force:
        cmd.insert(2, "--no-cache")

    print(f"building {task.image} (python {task.python}, "
          f"{task.base_sha[:12]})")
    if subprocess.run(cmd).returncode != 0:
        return 1
    print(f"built {task.image} -> {task.image_id()[:19]}")
    return 0


# --- broken state -----------------------------------------------------------

def record_broken(task: Task, force: bool = False) -> int:
    """Run the suite in a fresh container and record what fails.

    This is the reference the agent's work is measured against, so it is
    produced by exactly the code a live run uses — same Sandbox, same
    Tools, same report parser. When a shell script produced it instead,
    the recorded baseline and the measured one were two readings of the
    same format by two different parsers, free to disagree.
    """
    from agentcheck.sandbox import Sandbox
    from agentcheck.tools import Tools

    out = task.broken_report_path
    if out.exists() and not force:
        print(f"{out} already exists (use --force to re-record)")

    try:
        print(f"image: {task.image_id()[:19]}")
    except TaskError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    with Sandbox(task.image, env=task.env) as box:
        tools = Tools(box, task.test_command)
        print(f"running: {task.test_command}")
        result = tools.run_tests()

        if result.is_error:
            print(f"error: the suite produced no verdict: {result.reason}",
                  file=sys.stderr)
            tail = box.exec(["tail", "-30", "/tmp/agentcheck/out.txt"])
            print(tail.stdout, file=sys.stderr)
            return 1

        # The document the parser actually read, not a second read of the
        # file by a different code path.
        report = json.dumps(tools.last_report, indent=1)

    print(f"got:      {result.passed} passed, {result.failed} failed, "
          f"{result.errored} errored, {result.skipped} skipped")

    expected = task.expected_broken
    if expected:
        want = (expected.get("passed"), expected.get("failed"))
        got = (result.passed, result.failed)
        print(f"expected: {want[0]} passed, {want[1]} failed  (task.yaml)")
        if None not in want and want != got:
            print("MISMATCH — the image or the pins have drifted. Fix that "
                  "before recording,\nor every progress number measured "
                  "against this file is measured against\nthe wrong "
                  "baseline.", file=sys.stderr)
            return 1

    if not result.failed_ids:
        print("error: nothing fails in this container, so the task measures "
              "nothing", file=sys.stderr)
        return 1

    if out.exists() and not force:
        previous = task.recorded_broken_state()
        if previous.failed_ids != result.failed_ids:
            only_old = sorted(previous.failed_ids - result.failed_ids)
            only_new = sorted(result.failed_ids - previous.failed_ids)
            print("MISMATCH against the recorded failing set:", file=sys.stderr)
            for t in only_old[:5]:
                print(f"  only in the recorded file: {t}", file=sys.stderr)
            for t in only_new[:5]:
                print(f"  only in this run:          {t}", file=sys.stderr)
            print("Re-record with --force only if you intend to change the "
                  "baseline.", file=sys.stderr)
            return 1
        print("MATCH — the recorded failing set is reproduced exactly")
        return 0

    out.write_text(report)
    clusters = len({m for m in result.errors.values()})
    print(f"wrote {out} — {len(result.failed_ids)} failing tests from "
          f"{clusters} root cause(s)")
    print(f"  {clusters} root cause(s) is the honest denominator for "
          f"progress; {len(result.failed_ids)} tests is not")
    return 0


# --- reference diff ---------------------------------------------------------

def _fetch(url: str, timeout: int = 60) -> tuple[str | None, str]:
    """Fetch a URL, preferring curl.

    A framework Python on macOS routinely ships without a usable CA
    bundle, which makes urllib fail on every https URL with a certificate
    error that has nothing to do with this project. curl uses the system
    trust store and is always present here; urllib is the fallback for
    environments where it is not.
    """
    curl = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(timeout), url],
        capture_output=True, text=True)
    if curl.returncode == 0 and curl.stdout:
        return curl.stdout, ""
    reason = curl.stderr.strip() or f"curl exited {curl.returncode}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace"), ""
    except (urllib.error.URLError, OSError) as e:
        return None, f"{reason}; urllib: {e}"


def fetch_reference(task: Task, force: bool = False) -> int:
    out = task.reference_diff_path
    if out.exists() and not force:
        print(f"{out} already exists (use --force to refetch)")
        return 0

    m = _PR_URL.search(task.reference_pr or "")
    if not m:
        print(f"error: task.yaml has no usable reference_pr "
              f"(got {task.reference_pr!r})", file=sys.stderr)
        return 1

    url = (f"https://patch-diff.githubusercontent.com/raw/"
           f"{m['owner']}/{m['repo']}/pull/{m['n']}.diff")
    print(f"fetching {url}")
    diff, why = _fetch(url)
    if diff is None:
        print(f"error: could not fetch the reference diff: {why}\n"
              f"Save it by hand if the network is not available:\n"
              f"    curl -sL {url} > {out}", file=sys.stderr)
        return 1

    if "diff --git" not in diff:
        print("error: the response is not a diff", file=sys.stderr)
        return 1

    out.write_text(diff)
    files = sorted(set(re.findall(r"^diff --git a/(.+?) b/", diff, re.M)))
    tests = [f for f in files if "test" in f]
    print(f"wrote {out} — {len(files)} file(s) changed by the maintainer")
    if tests:
        print(f"  including {len(tests)} test file(s): {', '.join(tests)}")
        print("  Edits matching these are now reported as required "
              "migrations, not cheats.")
    else:
        print("  the maintainer changed no test files, so ANY test edit by "
              "an agent is a cheat")
    return 0


# --- coverage ---------------------------------------------------------------

def _coverage_command(test_command: str) -> str:
    """Turn the task's pytest invocation into a coverage run."""
    parts = shlex.split(test_command)
    if not parts or parts[0] != "pytest":
        raise TaskError(
            f"cannot derive a coverage command from {test_command!r} — "
            f"expected it to start with 'pytest'")
    rest = " ".join(shlex.quote(p) for p in parts[1:])
    return f"coverage run --source=. --branch -m pytest {rest}"


def record_coverage(task: Task, force: bool = False,
                    timeout: int = 900) -> int:
    out = task.coverage_path
    if out.exists() and not force:
        print(f"{out} already exists (use --force to re-record)")
        return 0

    baseline = task.root / "baseline-requirements.txt"
    if not baseline.exists():
        print(f"error: missing {baseline} — the green baseline cannot be "
              f"reconstructed, and coverage from the BROKEN state would "
              f"stop at the first failure and under-report every file",
              file=sys.stderr)
        return 1

    try:
        image = task.image_id()
    except TaskError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"image: {image[:19]}")

    name = f"agentcheck-coverage-{task.id}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    env_args = []
    for k, v in task.env.items():
        env_args += ["-e", f"{k}={v}"]

    started = subprocess.run(
        ["docker", "run", "-d", "--init", "--name", name, *env_args,
         task.image, "sleep", "infinity"],
        capture_output=True, text=True)
    if started.returncode != 0:
        print(f"error: {started.stderr.strip()}", file=sys.stderr)
        return 1

    try:
        subprocess.run(["docker", "cp", str(baseline),
                        f"{name}:/work/baseline-requirements.txt"], check=True)

        script = "\n".join([
            "set -e",
            "pip install --no-cache-dir --no-deps -r "
            "/work/baseline-requirements.txt",
            "pip install --no-cache-dir coverage",
            "cd /work/repo",
            _coverage_command(task.test_command),
            "coverage json -o /work/coverage.json --pretty-print",
        ])

        print("installing the baseline pins and running the suite under "
              "coverage (a few minutes)...")
        run = subprocess.run(
            ["docker", "exec", "-w", "/work/repo", name, "sh", "-c", script],
            capture_output=True, text=True, timeout=timeout)

        # `coverage run` exits non-zero when tests fail. On the baseline
        # they should not — and if they do, the recorded reachability would
        # describe a partial run, so it is refused rather than saved.
        tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
        if run.returncode != 0:
            print(f"error: the baseline suite did not pass cleanly, so this "
                  f"coverage would describe a partial run:\n{tail}",
                  file=sys.stderr)
            return 1

        got = subprocess.run(
            ["docker", "cp", f"{name}:/work/coverage.json", str(out)],
            capture_output=True, text=True)
        if got.returncode != 0:
            print(f"error: {got.stderr.strip()}", file=sys.stderr)
            return 1
    except subprocess.TimeoutExpired:
        print(f"error: timed out after {timeout}s", file=sys.stderr)
        return 1
    except (subprocess.CalledProcessError, TaskError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    data = json.loads(out.read_text())
    files = data.get("files") or {}
    executed = sum(len(f.get("executed_lines") or ()) for f in files.values())
    print(f"wrote {out} — {len(files)} file(s), {executed} executed lines")

    dead = sorted(p for p, f in files.items()
                  if not (f.get("executed_lines") or ()))
    if dead:
        print(f"  never executed at all: {', '.join(dead[:8])}")
    if task.unreachable:
        print(f"  the hand-listed prefixes in task.yaml are now superseded: "
              f"{list(task.unreachable)}")
    return 0


# --- cli --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=Task.available())
    p.add_argument("--build", action="store_true",
                   help="build the task image from task.yaml")
    p.add_argument("--broken", action="store_true",
                   help="record (or verify) the failing set the agent starts from")
    p.add_argument("--reference", action="store_true",
                   help="fetch the maintainer's PR diff")
    p.add_argument("--coverage", action="store_true",
                   help="record which lines the test command executes")
    p.add_argument("--all", action="store_true", help="both")
    p.add_argument("--force", action="store_true", help="overwrite existing")
    args = p.parse_args()

    if not (args.build or args.broken or args.reference or args.coverage
            or args.all):
        p.error("choose --build, --broken, --reference, --coverage or --all")

    task = Task.load(args.task)
    status = 0
    if args.build or args.all:
        status |= build_image(task, force=args.force)
        if status:
            return status
    if args.broken or args.all:
        status |= record_broken(task, force=args.force)
        if status:
            return status
    if args.reference or args.all:
        status |= fetch_reference(task, force=args.force)
    if args.coverage or args.all:
        status |= record_coverage(task, force=args.force)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
