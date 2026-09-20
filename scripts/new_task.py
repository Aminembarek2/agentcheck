#!/usr/bin/env python3
"""Screen and scaffold a new task from an upstream migration PR.

    .venv/bin/python scripts/new_task.py screen https://github.com/encode/databases/pull/540
    .venv/bin/python scripts/new_task.py screen --file candidates.txt
    .venv/bin/python scripts/new_task.py scaffold https://github.com/... --id 003-foo

Task authoring is the expensive part of this benchmark and most candidates
do not survive contact. `screen` applies the cheap disqualifying checks
before you spend an afternoon on dependency archaeology, and prints the
evidence for its verdict so you can disagree with it.

The four checks, in the order they save the most time:

  1. DOES THE PR TOUCH SOURCE? A migration PR that changes only tests and
     a version pin means the library needed no adaptation. There is no
     task there — the agent's entire job would be editing tests, which is
     the behaviour this benchmark treats as cheating.

  2. DID CI ALREADY TEST BOTH VERSIONS? This is the check worth having.
     If the config at the base commit runs the suite against the old AND
     the new dependency, the code already had a compatibility shim and the
     PR removed dead code rather than fixing breakage. Bumping the pin
     then breaks nothing and the task has no failing state to start from.
     Three candidates have been rejected on this alone, including
     marshmallow-jsonapi#100, whose Travis matrix ran
     MARSHMALLOW_VERSION="==2.8.0" alongside an unpinned install.

  3. IS THE ERA REACHABLE? A 2018 PR wants Python 3.6 and pins that
     predate wheels for anything newer. Buildable, but at a cost worth
     knowing before you start.

  4. IS IT THE SAME PACKAGE ON BOTH SIDES? A PR that swaps one library
     for another — aioredis for redis.asyncio, say — is a REPLACEMENT, not
     a major-version upgrade. The premise this benchmark is built on stops
     holding: there is no "the declared version is stale, ask
     package_version for the truth", because the new package was never
     declared at all. aiocache#546 passes every other check and fails
     this one.

  5. DOES IT NEED A SERVER? A suite that requires MongoDB or Redis cannot
     run on an internal network with no route out. Sometimes fixable by
     deselecting a module, sometimes fatal.

What screening CANNOT tell you is whether the failure is partial rather
than total — an import-time break gives the agent no incremental foothold
and makes a useless task. That needs a real build, which is what
`scaffold` then `prepare_task.py --build --broken` is for.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PR_URL = re.compile(
    r"github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/(?P<number>\d+)")

#: Files that are not the library. A PR touching only these did not adapt
#: any code.
NON_SOURCE = re.compile(
    r"(^|/)(tests?|testing|docs?|examples?)/"
    r"|\.(rst|md|txt|cfg|toml|ini|yml|yaml|lock)$"
    r"|(^|/)(CHANGELOG|CHANGES|HISTORY|AUTHORS|MANIFEST)")

#: Where a project declares what CI runs against.
CI_PATHS = (".github/workflows", ".travis.yml", "tox.ini", "azure-pipelines.yml",
            ".circleci/config.yml", "noxfile.py")

#: Suites that need something listening on a port.
SERVER_HINTS = ("mongodb", "postgres", "postgresql", "mysql", "mariadb",
                "redis", "memcached", "elasticsearch", "rabbitmq", "kafka",
                "services:", "docker-compose")


def gh(path: str) -> Any:
    """One GitHub API call, via curl so the system trust store is used."""
    proc = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "60",
         "-H", "Accept: application/vnd.github+json",
         f"https://api.github.com/{path.lstrip('/')}"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GitHub API {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def raw(owner: str, repo: str, sha: str, path: str) -> str | None:
    proc = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "60",
         f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"],
        capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


@dataclass
class Screening:
    url: str
    owner: str = ""
    repo: str = ""
    number: str = ""
    title: str = ""
    base_sha: str = ""
    merged: str = ""
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    source_lines: int = 0
    test_lines: int = 0
    ci_files: list[str] = field(default_factory=list)
    dual_version_evidence: list[str] = field(default_factory=list)
    python_versions: list[str] = field(default_factory=list)
    server_hints: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def viable(self) -> bool:
        return not self.blockers

    def report(self) -> str:
        head = "VIABLE" if self.viable else "REJECT"
        lines = [
            f"[{head}] {self.owner}/{self.repo}#{self.number} — {self.title[:60]}",
            f"    base {self.base_sha[:12]}  merged {self.merged}",
            f"    source: {len(self.source_files)} file(s) "
            f"({self.source_lines} lines)   "
            f"tests: {len(self.test_files)} file(s) ({self.test_lines} lines)",
        ]
        if self.python_versions:
            lines.append(f"    CI python: {', '.join(self.python_versions)}")
        for blocker in self.blockers:
            lines.append(f"    ! {blocker}")
        for note in self.notes:
            lines.append(f"    ? {note}")
        return "\n".join(lines)


def screen(url: str) -> Screening:
    m = PR_URL.search(url)
    if not m:
        raise ValueError(f"not a GitHub PR url: {url!r}")
    s = Screening(url=url, owner=m["owner"], repo=m["repo"],
                  number=m["number"])

    pr = gh(f"repos/{s.owner}/{s.repo}/pulls/{s.number}")
    s.title = pr.get("title", "")
    s.base_sha = pr.get("base", {}).get("sha", "")
    s.merged = (pr.get("merged_at") or "not merged")[:10]

    if not pr.get("merged_at"):
        s.blockers.append("the PR was never merged — it is not a reference "
                          "solution, only a proposal")

    # --- 1. does it touch source? -----------------------------------------
    files = gh(f"repos/{s.owner}/{s.repo}/pulls/{s.number}/files?per_page=100")
    for f in files:
        name = f["filename"]
        changed = f.get("additions", 0) + f.get("deletions", 0)
        if NON_SOURCE.search(name):
            if re.search(r"(^|/)tests?/|(^|/)test_", name):
                s.test_files.append(name)
                s.test_lines += changed
        else:
            s.source_files.append(name)
            s.source_lines += changed

    if not s.source_files:
        s.blockers.append(
            "the PR changes no library source — the code needed no "
            "adaptation, so the agent's entire job would be editing tests")
    elif s.source_lines < 10:
        s.notes.append(
            f"only {s.source_lines} source lines changed; the task may be "
            f"too small to separate models")

    # --- 2. did CI already test both versions? ----------------------------
    tree = gh(f"repos/{s.owner}/{s.repo}/git/trees/{s.base_sha}?recursive=1")
    paths = [p["path"] for p in tree.get("tree", []) if p["type"] == "blob"]
    ci_paths = [p for p in paths
                if any(p == c or p.startswith(c + "/") for c in CI_PATHS)]

    package = _guess_package(s, files)
    for path in ci_paths[:12]:
        body = raw(s.owner, s.repo, s.base_sha, path)
        if not body:
            continue
        s.ci_files.append(path)
        s.python_versions += re.findall(
            r"['\"]?(\d\.\d{1,2})['\"]?", _python_section(body))
        s.server_hints += [h for h in SERVER_HINTS if h in body.lower()]

        for line in _dual_version_lines(body, package):
            s.dual_version_evidence.append(f"{path}: {line.strip()[:90]}")

    s.python_versions = sorted(set(s.python_versions))
    s.server_hints = sorted(set(s.server_hints))

    if s.dual_version_evidence:
        s.blockers.append(
            "CI at the base commit already tested BOTH dependency versions, "
            "so the code had a compatibility shim and this PR removed dead "
            "code rather than fixing breakage. Bumping the pin will break "
            "nothing:\n        "
            + "\n        ".join(s.dual_version_evidence[:3]))

    # --- 3. same package on both sides? -----------------------------------
    removed, added = _pin_changes(files)
    swapped = removed - added
    gained = added - removed
    if package and swapped and package in swapped and gained:
        s.blockers.append(
            f"the PR REPLACES {', '.join(sorted(swapped))} with "
            f"{', '.join(sorted(gained))} rather than upgrading it. A swap "
            f"is not a major-version upgrade, and the premise that the "
            f"declared version is merely stale does not hold — the new "
            f"package was never declared at all")

    # --- 4. era ------------------------------------------------------------
    old = [v for v in s.python_versions if v.startswith(("2.", "3.0", "3.1 ",
                                                         "3.2", "3.3", "3.4",
                                                         "3.5", "3.6"))]
    if old and not [v for v in s.python_versions
                    if v in ("3.9", "3.10", "3.11", "3.12", "3.13")]:
        s.notes.append(
            f"era-correct Python is {', '.join(s.python_versions)} — pins "
            f"from this period predate wheels for newer interpreters and "
            f"will compile from source")

    # --- 5. servers --------------------------------------------------------
    if s.server_hints:
        s.notes.append(
            f"the suite may need a service ({', '.join(s.server_hints)}); "
            f"the sandbox has no route out, so those modules must be "
            f"deselected and the deselection explained in task.yaml notes")

    return s


def _pin_changes(files: list) -> tuple[set[str], set[str]]:
    """(packages removed from the pins, packages added to them)."""
    removed: set[str] = set()
    added: set[str] = set()
    for f in files:
        if not re.search(r"(setup\.py|pyproject\.toml|setup\.cfg|"
                         r"requirements.*\.txt|tox\.ini)$", f["filename"]):
            continue
        for line in (f.get("patch") or "").splitlines():
            if line.startswith(("+++", "---")) or line[:1] not in "+-":
                continue
            m = re.search(r"['\"]?([A-Za-z][\w.-]{2,})\s*(?:[><=~!]=|['\"]?\s*$)",
                          line[1:].strip())
            if not m:
                continue
            (added if line[0] == "+" else removed).add(m.group(1).lower())
    return removed, added


def _guess_package(s: Screening, files: list) -> str:
    """The dependency being migrated, from the PR title or the pin diff."""
    for f in files:
        if not re.search(r"(setup\.py|pyproject\.toml|setup\.cfg|"
                         r"requirements.*\.txt)$", f["filename"]):
            continue
        for line in (f.get("patch") or "").splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                m = re.search(r"['\"]?([A-Za-z][\w.-]{2,})\s*[><=~!]=", line)
                if m:
                    return m.group(1).lower()
    m = re.match(r"^\W*([A-Za-z][\w.-]{2,})\b", s.title)
    return m.group(1).lower() if m else ""


def _python_section(body: str) -> str:
    m = re.search(r"python[-_ ]?versions?\s*:(.*?)(\n\S|\Z)", body,
                  re.S | re.I)
    return m.group(1) if m else ""


def _dual_version_lines(body: str, package: str) -> list[str]:
    """Lines showing CI running against more than one version of `package`.

    Two shapes cover almost everything: an explicit version matrix entry
    naming the package, and a tox/nox envlist with per-version factors.
    """
    if not package:
        return []
    hits = []
    token = re.escape(package)
    pinned = re.compile(rf"{token}[-_ ]?version\s*[:=]", re.I)
    factor = re.compile(rf"{token}\d+", re.I)

    lines = body.splitlines()
    for i, line in enumerate(lines):
        if pinned.search(line):
            # A matrix key naming the package. Two or more values under it
            # means two or more versions were tested.
            block = "\n".join(lines[i:i + 8])
            values = re.findall(r"^\s*-\s*(.+)$", block, re.M)
            if len(values) >= 2:
                hits.append(f"{line.strip()} -> {values[:3]}")
        elif factor.search(line) and re.search(r"envlist|matrix|strategy",
                                               body[:i * 40 + 400], re.I):
            hits.append(line)
    return hits


# --- scaffold ---------------------------------------------------------------

TASK_TEMPLATE = """\
id: {task_id}
repo: {repo_url}
base_sha: {base_sha}
package: {package}
from_version: "FILL IN"
to_version: "FILL IN"
python: "{python}"

reference_pr: {url}
reference_files:
{reference_files}

test_command: pytest tests/
env: {{}}

# Recorded by: .venv/bin/python scripts/prepare_task.py {task_id} --build --broken
baseline: {{}}
broken:   {{}}

notes: |
  FILL IN: why each deselected module is deselected, and anything about
  the pins a future reader would otherwise have to rediscover.
"""


def scaffold(url: str, task_id: str, tasks_dir: Path) -> int:
    s = screen(url)
    print(s.report())
    if not s.viable:
        print("\nrefusing to scaffold a rejected candidate. Override by "
              "writing task.yaml by hand if you disagree with the verdict.",
              file=sys.stderr)
        return 1

    root = tasks_dir / task_id
    if root.exists():
        print(f"{root} already exists", file=sys.stderr)
        return 1
    root.mkdir(parents=True)

    package = _guess_package(
        s, gh(f"repos/{s.owner}/{s.repo}/pulls/{s.number}/files?per_page=100"))
    python = next((v for v in reversed(s.python_versions)
                   if v in ("3.10", "3.11", "3.12")), "3.11")

    (root / "task.yaml").write_text(TASK_TEMPLATE.format(
        task_id=task_id,
        repo_url=f"https://github.com/{s.owner}/{s.repo}",
        base_sha=s.base_sha, package=package or "FILL IN", python=python,
        url=url,
        reference_files="\n".join(f"  - {f}" for f in s.source_files) or "  []",
    ))

    diff = subprocess.run(
        ["curl", "-sSL", "--fail",
         f"https://patch-diff.githubusercontent.com/raw/{s.owner}/{s.repo}/"
         f"pull/{s.number}.diff"], capture_output=True, text=True)
    if diff.returncode == 0:
        (root / "reference.patch").write_text(diff.stdout)

    print(f"\nscaffolded {root}. Remaining steps, in order:")
    print("  1. write baseline-requirements.txt — the era-correct GREEN pins")
    print(f"     (read {', '.join(s.ci_files[:2]) or '.github/workflows/'} at "
          f"{s.base_sha[:12]}; this step is the whole cost of task authoring)")
    print(f"  2. copy it to broken-requirements.txt and bump ONLY {package}")
    print(f"  3. .venv/bin/python scripts/prepare_task.py {task_id} --build --broken")
    print("  4. CHECK THE FAILURE IS PARTIAL. A total or import-time break "
          "gives the\n     agent no foothold and makes a useless task.")
    print(f"  5. .venv/bin/python scripts/prepare_task.py {task_id} --coverage")
    return 0


# --- cli --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="cheap disqualifying checks")
    s.add_argument("urls", nargs="*")
    s.add_argument("--file", type=Path, help="one PR url per line")

    c = sub.add_parser("scaffold", help="create tasks/<id>/ from a PR")
    c.add_argument("url")
    c.add_argument("--id", required=True)
    c.add_argument("--tasks-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "tasks")

    args = p.parse_args()
    if args.cmd == "scaffold":
        return scaffold(args.url, args.id, args.tasks_dir)

    urls = list(args.urls)
    if args.file:
        urls += [l.strip() for l in args.file.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    if not urls:
        p.error("give at least one PR url, or --file")

    viable = 0
    for url in urls:
        try:
            result = screen(url)
        except Exception as e:
            print(f"[ERROR ] {url}: {e}")
            continue
        print(result.report())
        print()
        viable += result.viable

    print(f"{viable}/{len(urls)} candidate(s) survived screening.")
    print("Screening cannot tell you whether the failure is PARTIAL. Build "
          "the survivors and\ncheck — an import-time break gives the agent "
          "no foothold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
