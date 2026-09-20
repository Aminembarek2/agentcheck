"""Container session for agentcheck.

The agent runs on the host; its tools exec into a long-lived container.
That keeps the sandbox dumb (no API keys, no model code inside it), lets
models be swapped without rebuilding an image, and keeps traces on the host.

Nothing here knows about LLMs. It is deliberately testable on its own.

Three invariants this module is responsible for, each of which was a
scoring bug before it was an invariant:

  * The diff must show EVERYTHING the agent changed, including files it
    created. `git diff` alone does not — a new conftest.py is invisible to
    it, and an invisible file is an undetectable cheat.
  * The agent must not be able to write outside the repository. Editing
    installed site-packages is both an unscoreable change and a way for
    one run to contaminate the next.
  * An artifact is only "ready" when it PARSES. Polling for existence
    returns the instant a file is created, which is before its first byte
    is written.
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

#: Internal user-defined network: working loopback, no route to the internet.
#: Created automatically by ensure_network() if absent.
ISOLATED_NETWORK = "agentcheck-isolated"

#: Every container this harness starts carries it, so a crashed process
#: leaves something sweepable rather than an anonymous orphan.
OWNER_LABEL = "agentcheck.owner=harness"

#: Label key carrying the PID of the process that started a container.
#:
#: This is what makes "orphan" a fact rather than a guess. Age was the
#: earlier proxy — a container older than the per-run wall cap is PROBABLY
#: abandoned — and probably is not good enough for a destructive operation
#: that, when wrong, deletes a container another process is mid-command on
#: and makes the victim record a failure it did not cause.
#:
#: A dead PID is the actual definition. The owner is gone, so nothing will
#: ever collect the container, so it is an orphan. A live PID means the
#: owner is still working, whatever the container's age.
PID_LABEL_KEY = "agentcheck.pid"


def _owner_alive(pid: int) -> bool:
    """Is the process that started a container still running?

    `os.kill(pid, 0)` sends no signal and only asks whether the process
    exists. PermissionError means it exists and belongs to someone else,
    which still counts as alive.

    PIDs are recycled, so a long-dead owner whose number has been reused
    reads as alive and its container survives a sweep. That error is in the
    safe direction: the cost is a leaked container, against deleting a live
    one. Age still bounds the leak.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Cannot tell. Treat as alive; see the docstring on which way to err.
        return True
    return True


class SandboxError(RuntimeError):
    """The container itself failed.

    Distinct from a command inside it failing. A tool that raises this has
    not produced a result the model can reason about, and the run must end
    rather than feeding the model an error string about a dead container.
    """


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class WaitResult:
    """The outcome of waiting for a command's artifact.

    `status` distinguishes the three cases that must never collapse into
    one: the artifact arrived and parsed, the deadline passed, or the
    artifact arrived and was unreadable. Only the first carries a value.
    """
    status: str                      # "ok" | "timeout" | "invalid"
    value: Any = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def ensure_network(name: str = ISOLATED_NETWORK) -> None:
    """Guarantee an INTERNAL docker network exists under this name.

    Verified rather than assumed. `docker network create agentcheck-isolated`
    without --internal produces a network with full outbound access under
    the expected name, and every container started on it would silently
    have internet — which would invalidate every result, because the task
    is premised on the agent being unable to look the answer up.
    """
    probe = subprocess.run(
        ["docker", "network", "inspect", "-f", "{{.Internal}}", name],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        created = subprocess.run(
            ["docker", "network", "create", "--internal", name],
            capture_output=True, text=True,
        )
        if created.returncode != 0:
            raise SandboxError(
                f"could not create the isolated network {name!r}: "
                f"{created.stderr.strip()}")
        return

    if probe.stdout.strip() != "true":
        raise SandboxError(
            f"docker network {name!r} exists but is NOT internal — containers "
            f"on it can reach the internet, which invalidates every run. "
            f"Remove it and let this harness recreate it:\n"
            f"    docker network rm {name}")


class Sandbox:
    """A running container for one task, with a reset-able working tree.

        with Sandbox("agentcheck/002-databases") as box:
            box.exec(["ls", "-la"])
    """

    WORKDIR = "/work/repo"
    SCRATCH = "/tmp/agentcheck"

    def __init__(self, image: str, env: dict[str, str] | None = None,
                 network: str = ISOLATED_NETWORK,
                 labels: tuple[str, ...] = ()):
        self.image = image
        self.env = dict(env or {})
        self.network = network
        #: Extra `key=value` labels, beyond the harness-wide owner label.
        #: They exist so a destructive reap can be SCOPED: everything the
        #: harness starts shares one owner label, so reaping by it alone is
        #: necessarily global and hits other processes' containers too.
        self.labels = tuple(labels)
        self.name = f"agentcheck-{uuid.uuid4().hex[:8]}"
        self._started = False
        #: Untracked paths that existed before the agent did anything —
        #: artifacts of running the suite, not of the agent's work.
        self._baseline_untracked: frozenset[str] = frozenset()

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> Sandbox:
        ensure_network(self.network)

        cmd = ["docker", "run", "-d", "--name", self.name,
               "--label", OWNER_LABEL,
               "--label", f"{PID_LABEL_KEY}={os.getpid()}",
               *[arg for label in self.labels for arg in ("--label", label)],
               # tini as PID 1, so orphaned children are reaped rather than
               # piling up as zombies across many runs.
               "--init",
               "--network", self.network]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self.image, "sleep", "infinity"]

        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise SandboxError(f"failed to start sandbox: {run.stderr.strip()}")

        self._started = True
        # Anything that fails from here on leaves a running container that
        # __exit__ will never see, because __enter__ raised. Tear it down
        # explicitly rather than leaking one container per failed start.
        try:
            self.exec(["mkdir", "-p", self.SCRATCH])
            self._repo_root = self.exec(
                ["sh", "-c", f"cd {self.WORKDIR} && pwd -P"]
            ).stdout.strip() or self.WORKDIR
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        if not self._started:
            return
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, text=True)
        self._started = False

    def __enter__(self) -> Sandbox:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- execution ----------------------------------------------------------

    def exec(self, argv: list[str], timeout: int = 120) -> ExecResult:
        """Run a short command inside the container.

        argv, not a shell string: the agent passes file paths, and shell
        quoting is a bug factory. stdin is closed so nothing blocks waiting
        for input that will never arrive.

        Do NOT use this for the test suite — see run_until_json.
        """
        if not self._started:
            raise SandboxError("sandbox not started")
        try:
            run = subprocess.run(
                ["docker", "exec", "-w", self.WORKDIR, self.name, *argv],
                capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"timed out after {timeout}s")
        except OSError as e:
            raise SandboxError(f"docker exec failed: {e}") from e
        return ExecResult(run.returncode, run.stdout, run.stderr)

    def _spawn(self, shell_cmd: str) -> None:
        """Start a detached command and return immediately."""
        if not self._started:
            raise SandboxError("sandbox not started")
        run = subprocess.run(
            ["docker", "exec", "-d", "-w", self.WORKDIR, self.name,
             "sh", "-c", shell_cmd],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            raise SandboxError(f"could not spawn command: {run.stderr.strip()}")

    def run_until(self, shell_cmd: str, marker: str,
                  parse: Callable[[str], Any],
                  timeout: int = 300, poll: float = 1.0) -> WaitResult:
        """Run a command detached and wait for an artifact it can PARSE.

        Deliberately does NOT wait for the process to exit.

        pytest can complete a run, write its JSON report, and then hang
        forever at interpreter shutdown when the suite leaves non-daemon
        threads alive — aiosqlite does exactly this. `docker run --rm` never
        shows the problem because the container is torn down anyway;
        `docker exec` waits politely for a process that will never die.

        Waiting on the artifact sidesteps that, but existence is the wrong
        signal: a file exists from the moment it is created, which is before
        its first byte is written, so `test -f` can win a race against a
        half-written report. Parsing is the right signal — a truncated JSON
        document cannot parse, so `parse` succeeding is proof the writer
        finished. This needs no cooperation from the writer and has no
        sleep-and-hope window.
        """
        self.exec(["rm", "-f", marker])
        self._spawn(shell_cmd)

        deadline = time.time() + timeout
        last_error = "the artifact never appeared"

        while time.time() < deadline:
            probe = self.exec(["cat", marker], timeout=30)
            if probe.ok and probe.stdout:
                try:
                    value = parse(probe.stdout)
                except Exception as e:
                    # Almost always a partial write. Keep polling; only a
                    # file that never parses before the deadline is invalid.
                    last_error = f"{type(e).__name__}: {e}"
                else:
                    self.kill_stragglers()
                    return WaitResult("ok", value)
            time.sleep(poll)

        self.kill_stragglers()
        status = "timeout" if last_error.startswith("the artifact") else "invalid"
        return WaitResult(status, None,
                          f"after {timeout}s: {last_error}")

    def run_until_json(self, shell_cmd: str, marker: str,
                       timeout: int = 300,
                       required_keys: tuple[str, ...] = ()) -> WaitResult:
        """run_until specialised to a JSON artifact with required keys.

        The key check matters as much as the parse: pytest-json-report
        writes its top-level keys last, so a document can be valid JSON
        while still missing `tests`.
        """
        def parse(text: str) -> dict:
            data = json.loads(text)
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"incomplete report, missing {missing}")
            return data

        return self.run_until(shell_cmd, marker, parse, timeout=timeout)

    def run_until_file(self, shell_cmd: str, marker: str,
                       timeout: int = 300) -> bool:
        """Existence-only wait, for artifacts with no parseable structure.

        Prefer run_until_json for anything that has one. This cannot tell a
        complete file from a partial one.
        """
        return self.run_until(shell_cmd, marker, lambda s: s,
                              timeout=timeout).ok

    def kill_stragglers(self) -> None:
        """Reap hung test processes so they don't accumulate across runs."""
        self.exec(["pkill", "-9", "-x", "python", "-f", "pytest"], timeout=15)
        self.exec(["pkill", "-9", "-f", "pytest"], timeout=15)

    # --- working tree -------------------------------------------------------

    def reset(self) -> None:
        """Discard every change the agent made.

        Called between attempts so runs are independent. Without this,
        run N+1 inherits run N's mess and the numbers stop meaning anything.

        `-x` on the clean is deliberate: without it, ignored files survive,
        and that includes __pycache__ full of bytecode compiled from the
        previous run's source. `git reset` first because diff() marks new
        files with intent-to-add, which otherwise protects them from clean.
        """
        self.kill_stragglers()
        self._baseline_untracked = frozenset()
        self.exec(["git", "reset", "-q"])
        self.exec(["git", "checkout", "--", "."])
        self.exec(["git", "clean", "-fdx"])

    def untracked(self) -> frozenset[str]:
        """Paths git does not track, including ignored ones."""
        out = self.exec([
            "git", "status", "--porcelain", "-z", "-uall", "--no-renames"
        ]).stdout
        return frozenset(
            entry[3:] for entry in out.split("\0")
            if entry.startswith("?? ") and entry[3:])

    def mark_baseline(self) -> frozenset[str]:
        """Record which untracked files exist before the agent starts.

        Running the suite creates artifacts — task 002 leaves a `testsuite`
        SQLite database in the repository root, and it is not in
        .gitignore. Once diff() started reporting created files, those
        artifacts began showing up as changes the agent made. They are not,
        and a diff that lists them is both noisy and, for a judge reading
        it, misleading.

        Call after the before-state test run and before the agent starts.
        """
        self._baseline_untracked = self.untracked()
        return self._baseline_untracked

    def _exclusions(self) -> list[str]:
        if not self._baseline_untracked:
            return []
        return ["--", "."] + [f":(exclude,literal){p}"
                              for p in sorted(self._baseline_untracked)]

    def diff(self) -> str:
        """The agent's complete change set — what the cheat detectors and
        the judge are scored against.

        `git add -A -N` first. Plain `git diff` does not show untracked
        files at all, so an agent that CREATES a file — a root conftest.py
        that skips the failing tests, a pytest.ini that deselects them, a
        sitecustomize.py that monkeypatches the library — produced an empty
        diff and scored as a clean run. Intent-to-add puts new paths in the
        index without staging content, which makes them appear in the diff
        as additions while leaving the working tree untouched.

        Artifacts that already existed before the agent started are
        excluded — see mark_baseline.
        """
        exclude = self._exclusions()
        self.exec(["git", "add", "-A", "-N", *exclude])
        return self.exec(["git", "diff", *exclude]).stdout

    def changed_files(self) -> list[str]:
        exclude = self._exclusions()
        self.exec(["git", "add", "-A", "-N", *exclude])
        out = self.exec(["git", "diff", "--name-only", *exclude]).stdout
        return [line for line in out.splitlines() if line.strip()]

    # --- file access --------------------------------------------------------

    def resolve_in_repo(self, path: str) -> str:
        """Absolute path for `path`, or raise if it escapes the repo.

        The agent's writes must land where the diff can see them. A write
        to site-packages is invisible to git, survives reset(), and
        silently contaminates every later run in the same container; a
        write to /work/broken-report.json edits the answer key. Neither is
        a cheat the scorer could ever detect, so it is prevented rather
        than measured.

        Resolution happens INSIDE the container, with realpath, so that
        symlinks are followed. An earlier version walked up to the deepest
        existing ancestor and resolved that ancestor's PARENT — which
        meant a symlink that was itself the deepest existing component was
        never followed. `etclink -> /etc` plus a write to
        `etclink/evil.conf` landed in /etc, because /etc/evil.conf did not
        exist yet and so the check stopped one level too early. realpath
        resolves every component it can, including a trailing one that
        does not exist yet, which is exactly the semantics needed here.
        """
        root = getattr(self, "_repo_root", self.WORKDIR)
        candidate = path if posixpath.isabs(path) else posixpath.join(root, path)

        probe = self.exec([
            "python", "-c",
            "import os, sys; print(os.path.realpath(sys.argv[1]))",
            candidate,
        ])
        real = probe.stdout.strip()
        if not probe.ok or not real:
            # Refuse rather than fall back to the unresolved path: a check
            # that fails open is not a check.
            raise PermissionError(
                f"{path}: could not be resolved inside the container "
                f"({probe.stderr.strip() or 'no output'})")

        if not (real == root or real.startswith(root + "/")):
            raise PermissionError(
                f"{path}: resolves to {real}, outside the repository "
                f"({root}). Edits must live in the repo so they appear in "
                f"the diff and are scored.")
        return real

    def read_file(self, path: str) -> str:
        target = self.resolve_in_repo(path)
        r = self.exec(["cat", target])
        if not r.ok:
            raise FileNotFoundError(f"{path}: {r.stderr.strip()}")
        return r.stdout

    def write_file(self, path: str, content: str) -> None:
        """Written via stdin rather than an interpolated shell string, so
        quotes and backslashes in generated code cannot break the command."""
        target = self.resolve_in_repo(path)
        if not self._started:
            raise SandboxError("sandbox not started")

        parent = posixpath.dirname(target)
        if parent:
            self.exec(["mkdir", "-p", parent])

        proc = subprocess.run(
            ["docker", "exec", "-i", "-w", self.WORKDIR, self.name,
             "tee", target],
            input=content, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise OSError(f"write {path}: {proc.stderr.strip()}")


def sweep_orphans(min_age_seconds: float = 0.0,
                  label: str = OWNER_LABEL) -> list[str]:
    """Remove containers this harness started and never cleaned up.

    A SIGKILL of the host process leaves the container running; nothing
    else will ever collect it.

    `min_age_seconds` exists because this function is destructive and
    cannot tell an orphan from a sibling. It removes every container
    carrying the harness label, which was correct while runs were strictly
    serial and is catastrophic the moment two run in parallel: the sweep
    would reap the container of the attempt running beside it, and that
    attempt would be recorded as a harness failure caused by its own
    cleanup.

    Age is the discriminator that does not require shared state between
    processes. A run is killed at its own wall clock cap, so a container
    older than that cap plus a margin cannot belong to a live attempt.
    Callers running attempts concurrently pass `max_wall + margin`; a
    serial caller can pass 0 and reap everything, as before.

    Containers whose creation time cannot be parsed are LEFT ALONE. The
    failure mode of skipping a real orphan is some wasted disk; the
    failure mode of removing a live sibling is a fabricated failure in the
    results.

    `min_age_seconds=0` reaps EVERYTHING carrying the label, including
    containers another process is using right now. Nothing in the harness
    passes 0 any more; it remains reachable so the reaping mechanism can
    be tested directly, and callers should pass their own wall cap.

    `label` narrows what is considered at all. Every container the harness
    starts carries the same owner label, so a reap filtered on that alone
    is unavoidably global — which is how a single test destroyed a running
    sweep's container, and a running sweep destroyed the test suite's.
    Four separate symptoms today traced back to that one shared filter,
    every one of them looking like flaky infrastructure. A caller that
    wants to reap only its own containers starts them with an extra label
    and passes it here.
    """
    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"label={label}", "--format",
         "{{.ID}}\t{{.CreatedAt}}\t{{.Label \"" + PID_LABEL_KEY + "\"}}"],
        capture_output=True, text=True,
    )

    ids: list[str] = []
    now = datetime.now(timezone.utc)
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        cid, created = parts[0].strip(), parts[1].strip()
        owner = parts[2].strip() if len(parts) > 2 else ""
        if not cid:
            continue

        # The owning process decides it, when it can be identified. A live
        # owner means the container is in use however old it looks; a dead
        # one means nothing will ever collect it, whatever its age.
        if owner:
            try:
                pid = int(owner)
            except ValueError:
                pid = -1
            if pid > 0 and _owner_alive(pid):
                continue
            ids.append(cid)
            continue

        # No PID label: an older container, or one started by another tool.
        # Fall back to age, which is the proxy this used to rely on.
        if min_age_seconds <= 0:
            ids.append(cid)
            continue
        age = _container_age(created.strip(), now)
        if age is None:
            continue                      # unparseable: leave it alone
        if age >= min_age_seconds:
            ids.append(cid)

    if ids:
        subprocess.run(["docker", "rm", "-f", *ids],
                       capture_output=True, text=True)
    return ids


def _container_age(created: str, now: datetime) -> float | None:
    """Seconds since a `docker ps` CreatedAt string, or None if unreadable.

    Docker prints e.g. "2026-09-03 11:42:05 +0200 CEST". The trailing zone
    NAME is not parseable by strptime and is dropped; the numeric offset
    before it is what carries the information.
    """
    parts = created.split()
    if len(parts) < 3:
        return None
    try:
        stamp = datetime.strptime(" ".join(parts[:3]), "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None
    return (now - stamp).total_seconds()
