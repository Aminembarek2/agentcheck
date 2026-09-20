"""Agent tools for agentcheck.

Thin wrappers over Sandbox. Two rules govern everything here:

1. Output is sized for a context window. A 2,000-line file or 55 raw
   tracebacks will bury the signal and burn the budget. Every tool
   truncates deliberately and says so.

2. Output is structured. run_tests() is the scoring primitive as well as
   a tool, so the same code serves the agent loop and the eval layer.

Still no LLM in this file. It stays independently testable.

The one rule that is not about ergonomics: THE VERDICT COMES FROM THE
PER-TEST OUTCOMES, NEVER FROM THE SUMMARY BLOCK. pytest-json-report omits
summary keys whose count is zero, and `error` and `xfailed` are separate
keys from `failed`. Reading `summary["failed"]` with a default of 0 makes a
suite where every test errors in a fixture — or where the agent added xfail
markers — indistinguishable from a suite that passes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from agentcheck.sandbox import Sandbox, SandboxError

MAX_FILE_LINES = 400
MAX_SEARCH_HITS = 40
MAX_LISTING = 400

SCRATCH = "/tmp/agentcheck"
REPORT_PATH = f"{SCRATCH}/report.json"
STDOUT_PATH = f"{SCRATCH}/out.txt"

#: pytest exit codes meaning "no verdict was reached": 2 interrupted
#: (collection error), 3 internal error, 4 usage error, 5 nothing collected.
#: A collection error reports 0 passed and 0 failed. Reading that as success
#: would score "the agent destroyed the codebase" as a clean run — the exact
#: failure mode this project exists to catch.
PYTEST_NO_VERDICT = frozenset({2, 3, 4, 5})

#: Per-test outcomes that mean the test did not demonstrate correct
#: behaviour. `error` belongs here: a test whose fixture raises has not
#: passed, and it is not reported under `failed` anywhere in the report.
BAD_OUTCOMES = frozenset({"failed", "error"})

#: Outcomes that suppress a failure rather than resolve it. Counted
#: separately so an agent that converts failures into expected failures
#: shows up in the numbers as well as in the diff.
SUPPRESSED_OUTCOMES = frozenset({"xfailed", "xpassed"})

#: Every outcome this harness knows how to count. An outcome OUTSIDE this
#: set is not "not failing" — it is a state we cannot score, and treating
#: it as benign is the recurring bug of this project in its purest form.
#: pytest plugins add outcomes freely; pytest-rerunfailures emits "rerun",
#: which was counted nowhere at all and left a suite of 100 passed and one
#: rerun reading as fully green.
KNOWN_OUTCOMES = frozenset({"passed", "failed", "error", "skipped",
                            "xfailed", "xpassed"})


def _extract_error(longrepr: str) -> str:
    """Pull the real message out of a pytest traceback.

    The last line is a locator like `sqlite.py:118: AttributeError`. The
    actual message is on the line pytest prefixes with `E`.
    """
    lines = longrepr.strip().splitlines()
    for line in reversed(lines):
        if line.startswith("E "):
            return line[1:].strip()
    return lines[-1].strip() if lines else ""


def _longrepr(entry: dict) -> str:
    """The traceback for a test, from whichever phase actually failed."""
    for phase in ("call", "setup", "teardown"):
        section = entry.get(phase)
        if isinstance(section, dict):
            rep = section.get("longrepr")
            if rep:
                return rep if isinstance(rep, str) else json.dumps(rep)
    return ""


@dataclass(frozen=True)
class TestResult:
    """The outcome of one suite run.

    `status` is the field that must be consulted first. "no_verdict" means
    the suite did not produce a judgement — and an empty `failed_ids` in
    that state means "we do not know what is failing", NOT "nothing is
    failing". Those subtract to opposite conclusions, so they are never the
    same value here.
    """
    #: pytest would otherwise try to COLLECT this class in any test module
    #: that imports it, because the name begins with "Test".
    __test__ = False

    status: Literal["ok", "no_verdict"] = "ok"
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    exit_code: int | None = None
    failed_ids: frozenset[str] = frozenset()
    errors: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    @property
    def is_error(self) -> bool:
        return self.status != "ok"

    @property
    def all_green(self) -> bool:
        """Every test ran and demonstrated correct behaviour.

        `passed > 0` guards against a collection failure reading as
        success: zero tests passing is never a win. `not failed_ids`
        covers errors as well as failures. Suppressed outcomes are NOT
        checked here — a suite with pre-existing xfails is legitimately
        green — but they are reported so the scorer can compare them
        against the before-state and notice new ones.
        """
        return (self.status == "ok"
                and self.passed > 0
                and not self.failed_ids)

    @property
    def suppressed(self) -> int:
        return self.xfailed + self.xpassed

    @classmethod
    def no_verdict(cls, reason: str, exit_code: int | None = None) -> TestResult:
        return cls(status="no_verdict", reason=reason, exit_code=exit_code)

    @classmethod
    def from_report(cls, data: dict) -> TestResult:
        """Build a result from a pytest-json-report document.

        Counts come from the per-test entries rather than from `summary`,
        and are cross-checked against it. A mismatch means the report is
        not describing the run we think it is, which is a no-verdict, not
        a number to publish.
        """
        exit_code = data.get("exitcode")
        if exit_code is None:
            return cls.no_verdict("report has no exitcode — cannot be trusted")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            # `"2" in {2, 3, 4, 5}` is False, so a string exit code walked
            # straight past the no-verdict check and a collection error
            # was scored as a real result.
            return cls.no_verdict(
                f"exitcode is {exit_code!r}, not an integer — the report is "
                f"not the shape this harness knows how to read")
        if exit_code in PYTEST_NO_VERDICT:
            return cls.no_verdict(
                f"pytest exited {exit_code} — the suite could not run "
                f"(collection error, e.g. invalid syntax)", exit_code)

        entries = data.get("tests")
        if entries is None:
            return cls.no_verdict("report has no per-test outcomes", exit_code)

        counts = Counter(t.get("outcome", "__missing__") for t in entries)

        unknown = sorted(set(counts) - KNOWN_OUTCOMES)
        if unknown:
            return cls.no_verdict(
                f"the report contains outcome(s) this harness cannot score: "
                f"{unknown}. They would otherwise be counted nowhere and "
                f"read as 'not failing'.", exit_code)

        nodeids = [t.get("nodeid") for t in entries]
        if len(set(nodeids)) != len(nodeids):
            # Two entries for one test means the counts and the id SET
            # disagree, and scoring reads the set while the report reads
            # the counts.
            return cls.no_verdict(
                f"{len(nodeids) - len(set(nodeids))} duplicate nodeid(s) — "
                f"the counts and the failing set cannot both be right",
                exit_code)

        summary = data.get("summary") or {}
        declared = summary.get("total")
        if declared is not None and declared != len(entries):
            return cls.no_verdict(
                f"report is inconsistent: summary says {declared} tests, "
                f"{len(entries)} were recorded", exit_code)

        failed_ids, errors = set(), {}
        for t in entries:
            if t.get("outcome") not in BAD_OUTCOMES:
                continue
            nid = t.get("nodeid")
            if not nid:
                return cls.no_verdict("a failing test has no nodeid", exit_code)
            failed_ids.add(nid)
            errors[nid] = _extract_error(_longrepr(t))

        return cls(
            status="ok",
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            errored=counts.get("error", 0),
            skipped=counts.get("skipped", 0),
            xfailed=counts.get("xfailed", 0),
            xpassed=counts.get("xpassed", 0),
            exit_code=exit_code,
            failed_ids=frozenset(failed_ids),
            errors=errors,
        )


class Tools:
    """The agent's entire surface on the repository.

    Keep this list short. Every extra tool is another thing the model can
    misuse, and another thing whose bugs look like model failures.
    """

    def __init__(self, box: Sandbox, test_command: str):
        self.box = box
        self.test_command = test_command
        #: The raw pytest-json-report document behind the last run_tests().
        #: Kept so a task's recorded baseline can be written from exactly
        #: the document the parser read, rather than re-read from the
        #: container by a second code path that could disagree with it.
        self.last_report: dict | None = None

    # --- 1. list_files ------------------------------------------------------

    def list_files(self, path: str = ".") -> str:
        """Source files under a path, excluding junk not worth attention."""
        try:
            target = self.box.resolve_in_repo(path)
        except PermissionError as e:
            return f"error: {e}"

        r = self.box.exec([
            "find", target,
            "-name", "*.py",
            "-not", "-path", "*/.git/*",
            "-not", "-path", "*/.venv/*",
            "-not", "-path", "*/__pycache__/*",
        ])
        if not r.ok:
            return f"error: {r.stderr.strip()}"
        files = sorted(line for line in r.stdout.splitlines() if line.strip())
        if not files:
            return "(no python files)"
        out = "\n".join(files[:MAX_LISTING])
        if len(files) > MAX_LISTING:
            out += f"\n\n... {len(files) - MAX_LISTING} more. Narrow the path."
        return out

    # --- 2. read_file -------------------------------------------------------

    def read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        """Read a file with line numbers.

        Line numbers matter: without them the model describes edits by
        quoting code, which is ambiguous when a line appears twice.
        """
        try:
            content = self.box.read_file(path)
        except (FileNotFoundError, PermissionError) as e:
            return f"error: {e}"

        lines = content.splitlines()
        if not lines:
            return "(empty file)"

        start = max(1, start)
        if start > len(lines):
            return f"error: {path} has {len(lines)} lines; start={start}"

        end = end or min(start + MAX_FILE_LINES - 1, len(lines))
        end = min(end, len(lines), start + MAX_FILE_LINES - 1)
        window = lines[start - 1:end]

        body = "\n".join(f"{i:>5}  {line}"
                         for i, line in enumerate(window, start=start))
        if end < len(lines):
            body += (f"\n\n... truncated at line {end} of {len(lines)}. "
                     f"Call read_file again with start={end + 1} for more.")
        return body

    # --- 3. search_code -----------------------------------------------------

    def search_code(self, pattern: str) -> str:
        """Literal grep over the repo.

        Deliberately the dumb version. Phase 6 adds embedding-based and
        agentic retrieval and measures all three against the reference
        diff; this is the baseline arm, so it stays simple and honest.
        """
        r = self.box.exec([
            "grep", "-rnF", "--include=*.py",
            "--exclude-dir=.git", "--exclude-dir=.venv",
            "--exclude-dir=__pycache__",
            "--", pattern, ".",
        ])
        # grep exits 1 on no matches, which is not an error here.
        if not r.stdout.strip():
            return f"no matches for {pattern!r}"

        hits = r.stdout.splitlines()
        out = "\n".join(hits[:MAX_SEARCH_HITS])
        if len(hits) > MAX_SEARCH_HITS:
            out += (f"\n\n... {len(hits) - MAX_SEARCH_HITS} more matches. "
                    f"Narrow the pattern.")
        return out

    # --- 4. write_file ------------------------------------------------------

    def write_file(self, path: str, content: str) -> str:
        """Overwrite a file wholesale.

        Whole-file writes rather than patches: patch application fails in
        ways hard to distinguish from the model being wrong, and telling
        those two apart is the entire point of this project.
        """
        try:
            self.box.write_file(path, content)
        except PermissionError as e:
            return (f"error: {e} Edit files under the repository instead.")
        except OSError as e:
            return f"error: {e}"
        return f"wrote {path} ({len(content.splitlines())} lines)"

    # --- 5. package_version -------------------------------------------------

    def package_version(self, name: str) -> str:
        """Report the installed version of a package.

        Added after observing a run stall on this: the model asked seven
        times to check the installed SQLAlchemy version, had no way to do
        it, and stopped without editing anything — despite having already
        diagnosed the problem correctly.

        The gap was structural, not a model failure. A human engineer runs
        `pip show`; the agent could not. Withholding it does not measure
        capability, it measures a missing tool.
        """
        r = self.box.exec([
            "python", "-c",
            "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))",
            name,
        ])
        if r.ok and r.stdout.strip():
            return f"{name}=={r.stdout.strip()}"
        return (f"{name} is not installed, or the name differs from its "
                f"import name")

    # --- 6. package_source --------------------------------------------------

    def package_source(self, name: str) -> str:
        """Locate an installed package's source directory on disk.

        Added after reading traces where agents repeatedly guessed at
        paths like /usr/local/lib/python3.11/site-packages/sqlalchemy —
        wrong minor version, so every attempt failed. Blocked from reading
        the library, they inferred API signatures from error messages
        alone and oscillated: fix, regress, fix again.

        A human engineer opens the installed source. Denying that measures
        the harness, not the agent. Combine with read_file to inspect the
        actual signature of whatever changed.
        """
        r = self.box.exec([
            "python", "-c",
            "import importlib.util as u, sys; s = u.find_spec(sys.argv[1]); "
            "print(s.submodule_search_locations[0] if s and "
            "s.submodule_search_locations else (s.origin if s else ''))",
            name,
        ])
        path = r.stdout.strip()
        if not r.ok or not path:
            return f"{name} is not importable"

        listing = self.box.exec(["ls", path])
        return (f"{name} source: {path}\n\n"
                f"contents:\n{listing.stdout.strip()[:1500]}")

    def read_package_file(self, path: str, start: int = 1,
                          end: int | None = None) -> str:
        """Read a file from an installed package, outside the repository.

        read_file is confined to the repo so the agent cannot edit its way
        into site-packages. Reading library source is legitimate and
        necessary, so it gets its own explicitly read-only door rather than
        a hole in the write confinement.
        """
        r = self.box.exec(["cat", path])
        if not r.ok:
            return f"error: {path}: {r.stderr.strip()}"

        lines = r.stdout.splitlines()
        start = max(1, start)
        end = end or min(start + MAX_FILE_LINES - 1, len(lines))
        end = min(end, len(lines), start + MAX_FILE_LINES - 1)
        body = "\n".join(f"{i:>5}  {line}"
                         for i, line in enumerate(lines[start - 1:end], start=start))
        if end < len(lines):
            body += (f"\n\n... truncated at line {end} of {len(lines)}. "
                     f"Call again with start={end + 1} for more.")
        return body or "(empty file)"

    # --- 7. run_tests -------------------------------------------------------

    def run_tests(self, timeout: int = 300) -> TestResult:
        """Run the suite and return structured results.

        Also the scoring primitive: the eval layer calls this and diffs the
        result against the task's recorded broken state.

        Waits for a report that PARSES rather than for pytest to exit — see
        Sandbox.run_until for both halves of why.
        """
        cmd = (f"{self.test_command} -q "
               f"--json-report --json-report-file={REPORT_PATH} "
               f"> {STDOUT_PATH} 2>&1")

        try:
            # run_until deletes the old report first, so a timed-out run
            # cannot read the PREVIOUS run's numbers and return them as fact.
            waited = self.box.run_until_json(
                cmd, REPORT_PATH, timeout=timeout,
                required_keys=("exitcode", "summary", "tests"))
        except SandboxError:
            # The container is gone. This is not a test outcome and the
            # model cannot act on it; let it propagate and end the run.
            raise

        self.last_report = None
        if not waited.ok:
            if waited.status == "timeout":
                return TestResult.no_verdict(
                    f"the test run timed out after {timeout}s")
            return TestResult.no_verdict(
                f"the report was never readable — {waited.detail}")

        self.last_report = waited.value
        return TestResult.from_report(waited.value)

    def run_tests_summary(self, timeout: int = 300) -> str:
        """run_tests, formatted for the model.

        Groups by error message rather than listing every failure: one root
        cause routinely produces dozens of failures (task 002: a single Row
        API change breaks 55 tests). Listing all of them wastes context and
        hides the structure.
        """
        result = self.run_tests(timeout=timeout)

        if result.is_error:
            return f"The test suite could not run at all. {result.reason}"
        if result.all_green:
            note = ""
            if result.suppressed:
                note = (f" ({result.xfailed} xfailed, {result.xpassed} xpassed "
                        f"— these are suppressed, not fixed)")
            return (f"All tests pass: {result.passed} passed, "
                    f"{result.skipped} skipped.{note}")

        by_error: dict[str, list[str]] = {}
        for nid, msg in result.errors.items():
            by_error.setdefault(msg or "(no message)", []).append(nid)

        head = (f"{result.passed} passed, {result.failed} failed, "
                f"{result.errored} errored, {result.skipped} skipped.")
        lines = [head, ""]
        for msg, ids in sorted(by_error.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"[{len(ids)} tests] {msg}")
            for nid in sorted(ids)[:3]:
                lines.append(f"    {nid}")
            if len(ids) > 3:
                lines.append(f"    ... and {len(ids) - 3} more")
            lines.append("")
        return "\n".join(lines)
