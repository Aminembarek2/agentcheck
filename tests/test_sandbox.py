"""Tests for the sandbox layer.

Run these before writing a line of agent code. Tool bugs look exactly like
model failures, and debugging a model when the bug is in `cat` will cost
you days.

The isolated network is created automatically by ensure_network().

    pytest test_sandbox.py -v
"""


import uuid

import pytest

from agentcheck.sandbox import (
    OWNER_LABEL,
    Sandbox,
    SandboxError,
    ensure_network,
    sweep_orphans,
)

#: Everything in this module talks to a real container.
#: Run the fast layers alone with: pytest -m "not container"
pytestmark = pytest.mark.container

IMAGE = "agentcheck/002-databases"
ENV = {"TEST_DATABASE_URLS": "sqlite:///testsuite,sqlite+aiosqlite:///testsuite"}


@pytest.fixture
def box():
    with Sandbox(IMAGE, env=ENV) as s:
        yield s


# --- basics -----------------------------------------------------------------

def test_starts_and_execs(box):
    r = box.exec(["pwd"])
    assert r.ok
    assert r.stdout.strip() == "/work/repo"


def test_repo_is_present(box):
    r = box.exec(["ls"])
    assert "databases" in r.stdout
    assert "tests" in r.stdout


def test_env_reaches_the_container(box):
    assert "sqlite" in box.exec(["printenv", "TEST_DATABASE_URLS"]).stdout


def test_procps_is_available(box):
    """pkill is needed to reap test processes that hang at shutdown."""
    assert box.exec(["which", "pkill"]).ok


def test_exec_on_a_stopped_sandbox_raises_rather_than_returning_a_result():
    """A dead container is a harness failure, not a command that failed.
    Returning an ExecResult lets it be reported to the model as data."""
    s = Sandbox(IMAGE)
    with pytest.raises(SandboxError):
        s.exec(["ls"])


# --- file access ------------------------------------------------------------

def test_read_file(box):
    assert "sqlalchemy" in box.read_file("setup.py")


def test_read_missing_file_raises(box):
    with pytest.raises(FileNotFoundError):
        box.read_file("does_not_exist.py")


def test_write_then_read_roundtrip(box):
    # Deliberately nasty: quotes, backslashes, shell metacharacters. This is
    # what breaks naive shell interpolation, and generated code is full of it.
    content = 'x = "it\'s \\n fine"\n# comment $HOME `date`\n'
    box.write_file("scratch.py", content)
    assert box.read_file("scratch.py") == content


def test_write_creates_missing_directories(box):
    box.write_file("databases/backends/common/new.py", "x = 1\n")
    assert "x = 1" in box.read_file("databases/backends/common/new.py")


# --- write confinement ------------------------------------------------------

def test_writing_outside_the_repo_is_refused(box):
    """A write to site-packages is invisible to git, survives reset(), and
    contaminates every later run in the container. It is not a cheat the
    scorer could ever detect, so it is prevented rather than measured."""
    with pytest.raises(PermissionError):
        box.write_file("/usr/local/lib/python3.10/site-packages/x.py", "x=1\n")


def test_escaping_with_dot_dot_is_refused(box):
    with pytest.raises(PermissionError):
        box.write_file("../broken-report.json", "{}\n")


def test_an_absolute_path_inside_the_repo_is_allowed(box):
    box.write_file("/work/repo/ok.py", "x = 1\n")
    assert "x = 1" in box.read_file("ok.py")


def test_reading_outside_the_repo_is_refused_too(box):
    with pytest.raises(PermissionError):
        box.read_file("/etc/passwd")


# --- working tree -----------------------------------------------------------

def test_diff_sees_edits(box):
    box.write_file("setup.py", "# clobbered\n")
    assert "setup.py" in box.changed_files()
    assert "clobbered" in box.diff()


def test_diff_sees_CREATED_files(box):
    """`git diff` alone shows nothing for an untracked file. An agent that
    creates a root conftest.py to skip the failing tests produced an empty
    diff and scored as a clean run."""
    box.write_file("conftest.py", "collect_ignore = ['tests']\n")
    assert "conftest.py" in box.changed_files()
    assert "collect_ignore" in box.diff()


def test_reset_discards_edits(box):
    original = box.read_file("setup.py")
    box.write_file("setup.py", "# clobbered\n")
    box.reset()
    assert box.read_file("setup.py") == original
    assert box.changed_files() == []


def test_reset_removes_new_files(box):
    box.write_file("junk.py", "x = 1\n")
    box.reset()
    assert not box.exec(["ls", "junk.py"]).ok


def test_reset_removes_new_files_even_after_a_diff(box):
    """diff() marks new files intent-to-add, which puts them in the index
    and protects them from `git clean` unless the index is reset first."""
    box.write_file("junk.py", "x = 1\n")
    box.diff()
    box.reset()
    assert not box.exec(["ls", "junk.py"]).ok
    assert box.changed_files() == []


def test_reset_removes_ignored_bytecode(box):
    """`git clean -fd` without -x leaves __pycache__, so run N+1 can import
    bytecode compiled from run N's source."""
    box.exec(["python", "-c", "import databases"])
    box.reset()
    assert not box.exec(["sh", "-c", "ls databases/__pycache__"]).ok


# --- isolation and safety ---------------------------------------------------

def test_no_internet(box):
    """The agent must not be able to pip install its way around a problem.
    The isolated network has loopback but no route out."""
    r = box.exec(["python", "-c",
                  "import socket; socket.create_connection(('pypi.org', 443), 3)"])
    assert not r.ok


def test_the_network_is_verified_to_be_internal():
    """`docker network create agentcheck-isolated` without --internal gives
    full outbound access under the expected name, and every container on it
    would silently have internet. Verified, not assumed."""
    ensure_network()
    import subprocess
    out = subprocess.run(
        ["docker", "network", "inspect", "-f", "{{.Internal}}",
         "agentcheck-isolated"], capture_output=True, text=True)
    assert out.stdout.strip() == "true"


def test_a_non_internal_network_is_refused():
    import subprocess
    name = "agentcheck-test-leaky"
    subprocess.run(["docker", "network", "rm", name], capture_output=True)
    subprocess.run(["docker", "network", "create", name], capture_output=True)
    try:
        with pytest.raises(SandboxError) as e:
            ensure_network(name)
        assert "NOT internal" in str(e.value)
    finally:
        subprocess.run(["docker", "network", "rm", name], capture_output=True)


def test_the_answer_key_is_not_in_the_container(box):
    """broken-report.json lists every failing nodeid with its traceback.
    read_file can reach anything under /work, so shipping it would hand the
    agent the answer sheet."""
    assert not box.exec(["test", "-f", "/work/broken-report.json"]).ok


def test_the_upstream_history_is_not_in_the_container(box):
    """A full clone carries the maintainer's fix — the commit the task asks
    the agent to reproduce."""
    log = box.exec(["git", "log", "--oneline"])
    assert len(log.stdout.strip().splitlines()) == 1


def test_timeout_is_enforced(box):
    assert box.exec(["sleep", "10"], timeout=2).exit_code == 124


def test_sandboxes_are_isolated():
    with Sandbox(IMAGE, env=ENV) as a, Sandbox(IMAGE, env=ENV) as b:
        a.write_file("only_in_a.py", "x = 1\n")
        assert not b.exec(["ls", "only_in_a.py"]).ok


def test_a_failed_start_does_not_leak_a_container():
    """If docker run succeeds and the next step raises, __enter__ raises and
    __exit__ never runs — one leaked container per failed start."""
    s = Sandbox("agentcheck/does-not-exist")
    with pytest.raises(SandboxError):
        s.start()
    assert not s._started


def test_orphan_sweep_is_scoped_and_takes_nothing_else():
    """Reaping works, and takes nothing that is not its own.

    This test used to call `sweep_orphans(0.0)`, which removed EVERY
    container carrying the harness-wide owner label. Every process uses
    that label, so it destroyed whatever else was running: a live sweep
    lost containers to the test suite, and the suite lost containers to a
    live sweep. Five failures in one day traced back to it, each looking
    like flaky infrastructure first.

    Two things are asserted. The scoped container goes — that is the
    mechanism. The bystander stays — that is the part that was missing,
    and the reason the old version happily destroyed other processes' work
    while passing.

    The orphan is created with a dead owner PID, because that is now what
    orphan MEANS; flipping `_started` proves nothing when liveness is read
    from the process table.
    """
    import subprocess

    from agentcheck.sandbox import PID_LABEL_KEY

    scope = f"agentcheck.test={uuid.uuid4().hex[:8]}"
    name = f"agentcheck-orphan-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name,
         "--label", OWNER_LABEL, "--label", scope,
         "--label", f"{PID_LABEL_KEY}=999999",
         IMAGE, "sleep", "infinity"],
        capture_output=True, text=True)
    bystander = Sandbox(IMAGE, env=ENV).start()
    try:
        reaped = sweep_orphans(0.0, label=scope)
        assert len(reaped) == 1, (
            f"exactly one container carried {scope}; got {reaped}")

        gone = subprocess.run(["docker", "ps", "-aq", "-f", f"name={name}"],
                              capture_output=True, text=True)
        assert not gone.stdout.strip(), "the scoped orphan must be reaped"

        alive = subprocess.run(["docker", "ps", "-q", "-f",
                                f"name={bystander.name}"],
                               capture_output=True, text=True)
        assert alive.stdout.strip(), (
            "a container outside the scope must survive — an unscoped reap "
            "is what destroyed other processes' containers all day")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        bystander.stop()


def test_run_until_json_returns_the_parsed_value(box):
    marker = "/tmp/agentcheck/x.json"
    r = box.run_until_json(f'echo \'{{"a": 1}}\' > {marker}', marker, timeout=30)
    assert r.ok and r.value == {"a": 1}


def test_run_until_json_waits_for_a_PARTIAL_write_to_finish(box):
    """The race existence-polling could not win. A file exists from the
    moment it is created, which is before its first byte is written — so
    `test -f` plus a sleep is a guess. Parsing is proof."""
    marker = "/tmp/agentcheck/slow.json"
    cmd = (f'printf \'{{"exitcode": 0, "sum\' > {marker}; sleep 3; '
           f'printf \'mary": 2}}\' >> {marker}')
    r = box.run_until_json(cmd, marker, timeout=30)
    assert r.ok and r.value["summary"] == 2


def test_run_until_json_requires_the_declared_keys(box):
    """pytest-json-report writes its top-level keys last, so a document can
    be valid JSON while still missing `tests`."""
    marker = "/tmp/agentcheck/thin.json"
    r = box.run_until_json(f'echo \'{{"a": 1}}\' > {marker}', marker,
                           timeout=6, required_keys=("tests",))
    assert not r.ok
    assert "tests" in r.detail


def test_run_until_times_out(box):
    marker = "/tmp/agentcheck/never.txt"
    r = box.run_until(f"sleep 60; touch {marker}", marker, str, timeout=4)
    assert r.status == "timeout"


def test_run_until_clears_a_stale_marker(box):
    """A previous run's artifact must never be mistaken for this run's."""
    marker = "/tmp/agentcheck/stale.json"
    box.exec(["sh", "-c", f"echo '{{\"old\": true}}' > {marker}"])
    r = box.run_until_json(f"sleep 60; touch {marker}", marker, timeout=4)
    assert not r.ok


def test_run_until_returns_before_the_process_exits(box):
    """The whole point: the artifact is enough. A process that writes its
    report and then hangs at interpreter shutdown must not block us."""
    marker = "/tmp/agentcheck/early.json"
    r = box.run_until_json(f'echo \'{{"a": 1}}\' > {marker}; sleep 120',
                           marker, timeout=25)
    assert r.ok


# --- baseline artifacts -----------------------------------------------------

def test_test_run_artifacts_are_excluded_from_the_diff(box):
    """Running the suite leaves a `testsuite` SQLite file in the repo root,
    and it is not in .gitignore. Once the diff started reporting created
    files, that artifact showed up as a change the agent made."""
    box.exec(["sh", "-c", "echo db > testsuite"])
    box.mark_baseline()
    box.write_file("real_edit.py", "x = 1\n")
    assert box.changed_files() == ["real_edit.py"]
    assert "testsuite" not in box.diff()


def test_without_a_baseline_everything_untracked_still_shows(box):
    box.exec(["sh", "-c", "echo db > testsuite"])
    assert "testsuite" in box.changed_files()


def test_reset_clears_the_baseline(box):
    box.exec(["sh", "-c", "echo db > testsuite"])
    box.mark_baseline()
    box.reset()
    box.exec(["sh", "-c", "echo db > testsuite"])
    assert "testsuite" in box.changed_files()


def test_a_live_owners_container_is_never_reaped():
    """The fix that makes concurrency safe rather than unsupported.

    "Orphan" means the process that started the container is gone. Age was
    the earlier proxy — older than the wall cap is PROBABLY abandoned — and
    probably is not good enough for an operation that, when wrong, deletes
    a container another process is mid-command on and makes the victim
    record a failure it did not cause. That happened five times in one day,
    each time looking like flaky infrastructure.

    This container's owner is THIS process, which is alive by definition,
    so no threshold may remove it — including the zero threshold that used
    to mean "take everything".
    """
    import subprocess

    s = Sandbox(IMAGE, env=ENV).start()
    try:
        sweep_orphans(0.0)
        alive = subprocess.run(["docker", "ps", "-q", "-f", f"name={s.name}"],
                               capture_output=True, text=True)
        assert alive.stdout.strip(), (
            "a container whose owning process is still running must survive "
            "any reap, at any age threshold")
    finally:
        s.stop()


def test_a_dead_owners_container_is_reaped_whatever_its_age():
    """And the other direction: a dead owner means nothing will ever
    collect it, so age is irrelevant. A brand-new container from a crashed
    process is an orphan the moment the process dies."""
    import subprocess

    from agentcheck.sandbox import PID_LABEL_KEY

    name = f"agentcheck-orphan-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name,
         "--label", OWNER_LABEL,
         "--label", f"{PID_LABEL_KEY}=999999",     # a PID that is not running
         IMAGE, "sleep", "infinity"],
        capture_output=True, text=True)
    try:
        # An hour-long threshold: age would spare it, the dead owner does not.
        sweep_orphans(3600.0)
        gone = subprocess.run(["docker", "ps", "-aq", "-f", f"name={name}"],
                              capture_output=True, text=True)
        assert not gone.stdout.strip()
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
