"""Tests for the agent tools.

Still no LLM. Every failure above this line must be the model's fault, not
the plumbing's — that is the whole point of testing this layer first.

    pytest test_tools.py -v -s

Use -s: several of these run the full suite inside the container, and
without streamed output a slow test is indistinguishable from a hang.

One container is shared across the module and reset between tests. A fresh
container per test would mean running the suite repeatedly for no extra
isolation — reset already guarantees each test starts from the recorded
broken state.
"""

import pytest

from agentcheck.sandbox import Sandbox

#: Everything in this module talks to a real container.
#: Run the fast layers alone with: pytest -m "not container"
pytestmark = pytest.mark.container
from agentcheck.tools import TestResult, Tools, _extract_error

IMAGE = "agentcheck/002-databases"
ENV = {"TEST_DATABASE_URLS": "sqlite:///testsuite,sqlite+aiosqlite:///testsuite"}
TEST_CMD = "pytest tests/ --ignore=tests/test_connection_options.py"

# Recorded broken state for task 002.
EXPECTED_PASSED = 60
EXPECTED_FAILED = 55


@pytest.fixture(scope="module")
def _box():
    with Sandbox(IMAGE, env=ENV) as s:
        yield s


@pytest.fixture
def tools(_box):
    t = Tools(_box, TEST_CMD)
    yield t
    _box.reset()


# --- report parsing (no container) ------------------------------------------

def report(**over):
    data = {"exitcode": 1, "summary": {"total": 2}, "tests": [
        {"nodeid": "t::a", "outcome": "passed"},
        {"nodeid": "t::b", "outcome": "failed",
         "call": {"longrepr": "E   AttributeError: nope"}},
    ]}
    data.update(over)
    return data


def test_extract_error_prefers_the_E_line():
    longrepr = (
        "    result = await database.fetch_all(query)\n"
        "E   AttributeError: type object 'Row' has no attribute 'foo'\n"
        "\n"
        "databases/backends/sqlite.py:118: AttributeError"
    )
    assert _extract_error(longrepr) == (
        "AttributeError: type object 'Row' has no attribute 'foo'")


def test_extract_error_falls_back_to_last_line():
    assert _extract_error("something odd\nfinal line") == "final line"


def test_counts_come_from_the_tests_array():
    r = TestResult.from_report(report())
    assert (r.passed, r.failed) == (1, 1)
    assert r.failed_ids == frozenset({"t::b"})
    assert r.errors["t::b"] == "AttributeError: nope"


def test_an_ERRORED_test_is_not_a_pass():
    """The bug this closes. pytest-json-report omits summary keys whose
    count is zero, and `error` is a separate key from `failed`. Reading
    summary["failed"] with a default of 0 made a suite where every test
    errors in a fixture indistinguishable from one that passes."""
    data = report(summary={"passed": 1, "error": 1, "total": 2}, tests=[
        {"nodeid": "t::a", "outcome": "passed"},
        {"nodeid": "t::b", "outcome": "error",
         "setup": {"longrepr": "E   RuntimeError: fixture boom"}},
    ])
    r = TestResult.from_report(data)
    assert r.errored == 1
    assert r.failed_ids == frozenset({"t::b"})
    assert not r.all_green


def test_an_XFAILED_test_is_not_a_pass_either():
    """Adding xfail markers takes pytest to exit code 0 with no `failed`
    key at all. That must not read as a solved task."""
    data = report(exitcode=0, summary={"passed": 1, "xfailed": 1, "total": 2},
                  tests=[{"nodeid": "t::a", "outcome": "passed"},
                         {"nodeid": "t::b", "outcome": "xfailed"}])
    r = TestResult.from_report(data)
    assert r.all_green                      # nothing is failing, truthfully
    assert r.suppressed == 1                # but one test was silenced
    assert r.xfailed == 1


def test_a_collection_error_is_no_verdict():
    r = TestResult.from_report({"exitcode": 2, "summary": {"total": 0},
                                "tests": []})
    assert r.is_error and not r.all_green
    assert "could not run" in r.reason


def test_a_report_with_no_exitcode_is_no_verdict():
    r = TestResult.from_report({"summary": {}, "tests": []})
    assert r.is_error


def test_an_inconsistent_report_is_no_verdict():
    """A summary that disagrees with the per-test entries is not describing
    the run we think it is."""
    r = TestResult.from_report(report(summary={"total": 99}))
    assert r.is_error and "inconsistent" in r.reason


def test_zero_passing_is_never_green():
    r = TestResult.from_report({"exitcode": 0, "summary": {"total": 0},
                                "tests": []})
    assert not r.all_green


def test_no_verdict_and_no_failures_are_different_values():
    """They are the same count and opposite conclusions."""
    empty_ok = TestResult.from_report({"exitcode": 0, "summary": {"total": 1},
                                       "tests": [{"nodeid": "a",
                                                  "outcome": "passed"}]})
    empty_bad = TestResult.no_verdict("timed out")
    assert empty_ok.failed_ids == empty_bad.failed_ids == frozenset()
    assert empty_ok.all_green and not empty_bad.all_green


# --- list_files -------------------------------------------------------------

def test_list_files_finds_source(tools):
    out = tools.list_files("databases")
    assert "databases/core.py" in out
    assert "__pycache__" not in out


def test_list_files_outside_the_repo_is_refused(tools):
    assert "error" in tools.list_files("/usr/local/lib").lower()


# --- read_file --------------------------------------------------------------

def test_read_file_has_line_numbers(tools):
    out = tools.read_file("setup.py")
    assert "    1  " in out
    assert "sqlalchemy" in out


def test_read_file_missing(tools):
    assert "error" in tools.read_file("nope.py").lower()


def test_read_file_truncates_and_says_so(tools):
    out = tools.read_file("databases/core.py", start=1, end=5)
    assert "truncated" in out
    assert "start=6" in out


def test_read_file_window(tools):
    out = tools.read_file("setup.py", start=3, end=5)
    assert "    3  " in out
    assert "    1  " not in out


def test_read_file_beyond_the_end_says_so(tools):
    assert "error" in tools.read_file("setup.py", start=10_000)


def test_read_file_cannot_leave_the_repo(tools):
    assert "error" in tools.read_file("/etc/passwd").lower()


# --- search_code ------------------------------------------------------------

def test_search_finds_matches(tools):
    assert ".py:" in tools.search_code("sqlalchemy")


def test_search_no_matches_is_not_an_error(tools):
    assert "no matches" in tools.search_code("zzz_not_present_zzz")


def test_search_treats_the_pattern_as_a_literal(tools):
    """A model searching for `select([` must not have it read as a regex."""
    assert "error" not in tools.search_code("select([").lower()


# --- write_file -------------------------------------------------------------

def test_write_then_read(tools):
    tools.write_file("scratch.py", "x = 1\ny = 2\n")
    out = tools.read_file("scratch.py")
    assert "x = 1" in out
    assert "    2  y = 2" in out


def test_write_reports_line_count(tools):
    assert "3 lines" in tools.write_file("scratch.py", "a\nb\nc\n")


def test_write_outside_the_repo_is_refused_with_guidance(tools):
    """Editing installed site-packages is invisible to git, survives reset,
    and contaminates every later run. The model is told where to write
    instead rather than being left with an opaque failure."""
    out = tools.write_file("/usr/local/lib/python3.10/site-packages/x.py", "x=1\n")
    assert "error" in out.lower()
    assert "repository" in out.lower()


# --- package inspection -----------------------------------------------------

def test_package_version_reports_installed_not_declared(tools):
    """setup.py declares sqlalchemy<1.5, but 2.x is what is installed.
    A run stalled on exactly this contradiction with no way to resolve it."""
    assert tools.package_version("sqlalchemy").startswith("sqlalchemy==2")


def test_package_version_handles_missing_package(tools):
    assert "not installed" in tools.package_version("definitely_not_xyz")


def test_package_version_is_not_shell_injectable(tools):
    assert "not installed" in tools.package_version("x'); import os; ('")


def test_package_source_finds_the_real_path(tools):
    """Traces showed agents guessing python3.11 paths on a 3.10 image and
    failing every time. The path must be discovered, not guessed."""
    out = tools.package_source("sqlalchemy")
    assert "site-packages/sqlalchemy" in out
    assert "engine" in out


def test_package_source_handles_missing_package(tools):
    assert "not importable" in tools.package_source("definitely_not_here_xyz")


def test_library_source_is_readable_but_not_writable(tools):
    """Reading library source is the point of package_source; writing to it
    is not. The read gets its own explicitly read-only door rather than a
    hole in the write confinement."""
    path = tools.package_source("sqlalchemy").splitlines()[0].split(": ", 1)[1]
    assert "class Row" in tools.read_package_file(f"{path}/engine/row.py")
    assert "error" in tools.write_file(f"{path}/engine/row.py", "x=1\n").lower()


# --- run_tests --------------------------------------------------------------
# These run the full suite (~20s each). Slow by nature, not broken.

def test_run_tests_reproduces_broken_state(tools):
    """The most important test here. If the tools cannot reproduce the
    recorded broken state, nothing measured on top of them means anything."""
    r = tools.run_tests()
    assert not r.is_error
    assert (r.passed, r.failed) == (EXPECTED_PASSED, EXPECTED_FAILED)
    assert len(r.failed_ids) == EXPECTED_FAILED


def test_run_tests_captures_real_error_messages(tools):
    assert "Row" in " ".join(tools.run_tests().errors.values())


def test_summary_groups_by_root_cause(tools):
    """One API change breaks 55 tests. The summary must show that as a
    cluster, not 55 separate lines."""
    out = tools.run_tests_summary()
    assert "60 passed, 55 failed" in out
    assert "tests]" in out
    assert "more" in out


def test_broken_syntax_is_an_error_not_a_pass(tools):
    """If the agent writes something unparseable, pytest collects nothing
    and reports 0 passed / 0 failed. Reading that as success would score a
    destroyed codebase as a clean run."""
    tools.write_file("databases/core.py", "def broken(:\n")
    r = tools.run_tests()
    assert r.is_error and not r.all_green
    assert "could not run" in tools.run_tests_summary()


def test_stale_report_is_not_reused(tools):
    """A timed-out run must not return the previous run's numbers."""
    first = tools.run_tests()
    assert first.passed == EXPECTED_PASSED
    r = tools.run_tests(timeout=3)
    assert r.is_error
    assert r.passed == 0
    assert "timed out" in r.reason


def test_reset_restores_broken_state(tools):
    """After reset, the suite must return to exactly the recorded broken
    state — not to whatever the previous edit left behind."""
    tools.write_file("databases/core.py", "# clobbered\n")
    tools.box.reset()
    r = tools.run_tests()
    assert (r.passed, r.failed) == (EXPECTED_PASSED, EXPECTED_FAILED)
