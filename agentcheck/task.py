"""Task definitions for agentcheck — one source of truth.

There used to be three: a TASKS dict in run_agent.py, the task.yaml files,
and a grep/cut parser in run_task.sh. All three described the same test
command and env, all three were maintained by hand, and rescore.py needed a
defensive `TASKS.get(task, {})` because it imported one of them and could
not be sure the task was in it — which silently produced an empty
unreachable list and made every unverifiable edit disappear.

A Task also carries the two artifacts the scorer needs and previously did
not have:

  * reference.diff  — the maintainer's own change, so a test edit that
                      matches a required API migration can be told apart
                      from one that deletes the failing assertion.
  * coverage.json   — which lines the test command actually executes,
                      recorded once from the green baseline, replacing a
                      hand-maintained list of unreachable path prefixes.

Both are optional. When absent the scorer says so in its findings instead
of quietly assuming the permissive answer.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentcheck.scorer import Reachability, SuiteState

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


#: The contamination assessments a task may declare. "" — nobody has
#: looked — is deliberately not in this set and is never read as "low":
#: an unassessed task is unassessed, and reporting it as clean would be
#: the absent-is-never-zero failure applied to validity rather than to a
#: score.
CONTAMINATION_RISKS = ("high", "medium", "low")

#: Below this many distinct root causes, a task cannot carry a measurement
#: on its own: one correct edit moves every failing test at once, so
#: progress is near-binary and a per-test interval is meaningless. Three is
#: the floor phase 2 sets for a NEW task. Existing tasks below it are not
#: removed — they are required to say why, in `low_power_reason`, and
#: RESULTS.md repeats that wherever their numbers appear.
MIN_ROOT_CAUSES = 3


class TaskError(Exception):
    """The task is not usable, and guessing would produce a wrong number."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:      # pragma: no cover - environment
        raise TaskError(
            "PyYAML is required to read task definitions: pip install pyyaml"
        ) from e
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TaskError(f"{path}: expected a mapping at the top level")
    return data


@dataclass(frozen=True)
class Task:
    id: str
    root: Path
    repo: str
    base_sha: str
    python: str
    package: str
    from_version: str
    to_version: str
    test_command: str
    env: dict[str, str] = field(default_factory=dict)
    reference_pr: str = ""
    reference_sha: str = ""
    reference_files: tuple[str, ...] = ()
    #: Fallback for tasks with no coverage recorded. Path prefixes the test
    #: command is known not to execute.
    unreachable: tuple[str, ...] = ()
    expected_broken: dict[str, int] = field(default_factory=dict)
    expected_baseline: dict[str, int] = field(default_factory=dict)

    #: The date of the BASE commit, as YYYY-MM-DD, read from a clone with
    #: `git log -1` rather than estimated. It is the strongest available
    #: predictor of whether a model has memorised the migration rather
    #: than worked it out: a base commit years before every current
    #: training cutoff means the merged fix is almost certainly in the
    #: corpus.
    #:
    #: Deliberately NOT the PR merge date, which cannot be read offline.
    #: The base date is a verifiable lower bound; a merge date typed from
    #: memory would be a number nobody could check, which is the shape of
    #: error this project exists to avoid.
    base_sha_date: str = ""

    #: How likely it is that the model has seen this migration before, and
    #: why. All three tasks currently in the repo are old, public and
    #: heavily documented, which means a model reproducing the
    #: maintainer's patch may be recalling it. The harness cannot
    #: distinguish that from capability, so the risk is declared per task
    #: and RESULTS.md never averages across risk levels silently.
    #:
    #: One of "high", "medium", "low", or "" when nobody has judged yet —
    #: and "" is NOT read as low. An unassessed task is unassessed.
    contamination_risk: str = ""
    contamination_reason: str = ""

    #: The number of distinct root causes in the broken state, measured
    #: from clustered failures at task-build time. Task 002's 55 failing
    #: tests come from two causes, so a per-test denominator overstates
    #: its statistical power by roughly 27x. A task with fewer than three
    #: causes cannot carry a measurement on its own.
    root_causes: tuple[str, ...] = ()

    #: Required when there are fewer than MIN_ROOT_CAUSES. Below three
    #: causes, progress moves in jumps too large to measure — task 002 has
    #: two, one of which accounts for 53 of its 55 failing tests, so its
    #: progress is effectively three-valued. That is a real limitation of
    #: the task and it is declared here rather than discovered by a reader
    #: of the results table.
    low_power_reason: str = ""

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(cls, task_id: str, tasks_dir: Path | None = None) -> Task:
        root = (tasks_dir or TASKS_DIR) / task_id
        yaml_path = root / "task.yaml"
        if not yaml_path.exists():
            raise TaskError(f"no task definition at {yaml_path}")

        d = _load_yaml(yaml_path)
        # `package` is required because scorer.find_runtime_patching keys
        # off it: without it, an agent that monkeypatches the library back
        # to its old behaviour is undetectable, and the task would score
        # that as a clean fix.
        missing = [k for k in ("id", "repo", "base_sha", "test_command",
                               "package", "from_version", "to_version")
                   if not d.get(k)]
        if missing:
            raise TaskError(f"{yaml_path}: missing {missing}")
        if d["id"] != task_id:
            raise TaskError(
                f"{yaml_path}: declares id {d['id']!r} but lives in "
                f"{task_id!r} — one of them is wrong, and a mislabelled task "
                f"merges two experiments")

        contamination = {k: str(v).strip()
                         for k, v in (d.get("contamination") or {}).items()}
        risk = contamination.get("risk", "")
        if risk and risk not in CONTAMINATION_RISKS:
            raise TaskError(
                f"{yaml_path}: contamination.risk is {risk!r}; must be one "
                f"of {', '.join(CONTAMINATION_RISKS)}. An unrecognised "
                f"value would be reported as an assessment nobody made.")
        if risk and not contamination.get("reason"):
            raise TaskError(
                f"{yaml_path}: contamination.risk is set but no reason is "
                f"given. The bare label is not checkable, and a reader "
                f"cannot weigh a judgement whose grounds are absent.")

        causes = tuple(d.get("root_causes") or ())
        low_power = str(d.get("low_power_reason", "")).strip()
        if causes and len(causes) < MIN_ROOT_CAUSES and not low_power:
            raise TaskError(
                f"{yaml_path}: {len(causes)} root cause(s), below the "
                f"{MIN_ROOT_CAUSES} a task needs to carry a measurement, "
                f"and no low_power_reason given. Say what the task can and "
                f"cannot show — an undeclared limitation becomes an "
                f"overstated result.")

        return cls(
            id=d["id"],
            root=root,
            repo=d["repo"],
            base_sha=str(d["base_sha"]),
            python=str(d.get("python", "3.11")),
            package=str(d["package"]),
            from_version=str(d["from_version"]),
            to_version=str(d["to_version"]),
            test_command=d["test_command"],
            env={k: str(v) for k, v in (d.get("env") or {}).items()},
            reference_pr=str(d.get("reference_pr", "")),
            reference_sha=str(d.get("reference_sha", "")),
            reference_files=tuple(d.get("reference_files") or ()),
            unreachable=tuple(d.get("unreachable") or ()),
            expected_broken=dict(d.get("broken") or {}),
            expected_baseline=dict(d.get("baseline") or {}),
            base_sha_date=str(d.get("base_sha_date", "")),
            contamination_risk=contamination.get("risk", ""),
            contamination_reason=contamination.get("reason", ""),
            root_causes=causes,
            low_power_reason=low_power,
        )

    @classmethod
    def available(cls, tasks_dir: Path | None = None) -> list[str]:
        base = tasks_dir or TASKS_DIR
        if not base.is_dir():
            return []
        return sorted(p.parent.name for p in base.glob("*/task.yaml"))

    # --- artifacts ----------------------------------------------------------

    @property
    def image(self) -> str:
        return f"agentcheck/{self.id}"

    @property
    def broken_report_path(self) -> Path:
        return self.root / "broken-report.json"

    #: `.patch` is the canonical name; `.diff` is accepted because earlier
    #: fetches used it, and silently ignoring a file that is plainly there
    #: would make every test edit read as unjudged.
    REFERENCE_NAMES = ("reference.patch", "reference.diff")

    @property
    def reference_diff_path(self) -> Path:
        for name in self.REFERENCE_NAMES:
            candidate = self.root / name
            if candidate.exists():
                return candidate
        return self.root / self.REFERENCE_NAMES[0]

    @property
    def coverage_path(self) -> Path:
        return self.root / "coverage.json"

    def image_id(self) -> str:
        """The resolved image digest.

        The tag is not the experiment; the image is. A rebuild changes every
        installed dependency version with no change to any tracked file, so
        the digest belongs in the configuration fingerprint.
        """
        probe = subprocess.run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", self.image],
            capture_output=True, text=True)
        if probe.returncode != 0:
            raise TaskError(
                f"image {self.image} is not built: {probe.stderr.strip()}")
        return probe.stdout.strip()

    def recorded_broken_state(self) -> SuiteState:
        """The failing set the container is supposed to start from.

        Parsed with the same code that parses a live run's report, so the
        recorded baseline and the measured one can never drift apart
        through two different readers of the same format.
        """
        from agentcheck.tools import (
            TestResult,  # local: keeps scorer import-free
        )

        path = self.broken_report_path
        if not path.exists():
            raise TaskError(
                f"missing {path} — the before-state is unknown, and every "
                f"progress number computed without it is fabricated")
        result = TestResult.from_report(json.loads(path.read_text()))
        if result.is_error:
            raise TaskError(f"{path}: {result.reason}")
        return SuiteState.from_result(result)

    def reference_diff(self) -> str | None:
        """The maintainer's own change, if it has been recorded.

        None — not an empty string — when absent. An empty reference would
        mean "the maintainer changed nothing", under which every test edit
        is a cheat; absent means "we cannot tell", which is a different
        finding.
        """
        path = self.reference_diff_path
        if not path.exists():
            return None
        return path.read_text()

    def reachability(self) -> Reachability:
        path = self.coverage_path
        if path.exists():
            return Reachability.from_coverage(json.loads(path.read_text()))
        return Reachability(prefixes=self.unreachable)

    def describe_artifacts(self) -> list[str]:
        """Human-readable notes about what scoring will and will not know."""
        notes = []
        if self.reference_diff() is None:
            notes.append(
                f"no {self.reference_diff_path.name}: test edits cannot be "
                f"checked against the maintainer's PR, so they are reported "
                f"as unjudged rather than as cheats "
                f"(run: .venv/bin/python scripts/prepare_task.py {self.id} --reference)")
        if not self.coverage_path.exists():
            source = (f"the hand-listed prefixes {list(self.unreachable)}"
                      if self.unreachable else "nothing")
            notes.append(
                f"no {self.coverage_path.name}: reachability falls back to "
                f"{source} "
                f"(run: .venv/bin/python scripts/prepare_task.py {self.id} --coverage)")
        return notes
