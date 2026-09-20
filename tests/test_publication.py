"""Publication checks must catch leaks without printing their contents."""

import subprocess

import pytest

from scripts.check_publication import (
    check,
    local_links,
    private_path,
    secret_findings,
)


@pytest.mark.parametrize("prefix", ["sk-or-v1-", "sk-proj-", "ghp_", "xoxb-"])
def test_tokens_are_detected_without_becoming_diagnostics(prefix):
    token = prefix + "A1b2C3d4" * 8
    findings = secret_findings(("example\n" + token).encode())
    assert findings and findings[0][0] == 2
    assert token not in repr(findings)


@pytest.mark.parametrize("name", [".env", ".env.production", "key.pem",
                                 ".local/notes.json", "tasks/a/runs/report.json"])
def test_private_paths_are_rejected(name):
    from pathlib import Path
    assert private_path(Path(name))


def test_placeholder_and_environment_lookup_are_not_keys():
    assert not secret_findings(b'os.environ["OPENROUTER_API_KEY"]')
    assert not secret_findings(b"OPENROUTER_API_KEY=...")


def test_only_broken_local_links_are_flagged(tmp_path):
    (tmp_path / "actual.md").write_text("exists")
    source = tmp_path / "README.md"
    source.write_text("[ok](actual.md#section) [bad](missing.md) "
                      "[web](https://example.com) [anchor](#here)")
    assert local_links(source, tmp_path) == ["missing.md"]


def test_deleted_secret_in_history_is_still_detected(tmp_path, capsys):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    token = "sk-" + "A1b2C3d4" * 8
    path = tmp_path / "old.txt"
    path.write_text(token)
    git("add", "old.txt")
    git("commit", "-qm", "fixture")
    path.unlink()
    git("add", "-u")
    git("commit", "-qm", "remove fixture")
    assert check(tmp_path) == 0
    assert check(tmp_path, history=True) == 1
    output = capsys.readouterr()
    assert "history blob" in output.err
    assert token not in output.out + output.err
