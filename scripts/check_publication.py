"""Check publishable files for common secrets, scratch files and dead links.

    python scripts/check_publication.py
    python scripts/check_publication.py --history

Uses tracked files plus non-ignored new files. Deleted files are skipped.
Diagnostics name locations and rules, never matched credentials. This is a
pattern-based safeguard, not proof that arbitrary secrets are absent.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent

SECRET_RULES = {
    "private key": rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    "provider token": rb"\bsk-(?:(?:proj|svcacct|ant-api\d+|or-v1)-)?"
                      rb"[A-Za-z0-9_-]{24,}",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|"
                    rb"github_pat_[A-Za-z0-9_]{40,})",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "Google API key": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "Slack token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
}


def secret_findings(data: bytes) -> list[tuple[int, str]]:
    return [(data.count(b"\n", 0, m.start()) + 1, label)
            for label, pattern in SECRET_RULES.items()
            for m in re.finditer(pattern, data)]


def private_path(path: Path) -> bool:
    name = path.name.lower()
    return (
        ((name == ".env" or name.startswith(".env."))
         and name != ".env.example")
        or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
        or name in {"credentials.json", ".ds_store"}
        or (name.startswith("service-account") and name.endswith(".json"))
        or bool(set(path.parts) & {".local", ".venv", "__pycache__",
                                  ".pytest_cache", ".mypy_cache",
                                  ".ruff_cache"})
        or (len(path.parts) >= 3 and path.parts[0] == "tasks"
            and path.parts[2] == "runs")
    )


def local_links(path: Path, root: Path) -> list[str]:
    """Check file targets; external URLs and same-page anchors stay local."""
    source = path.read_text()
    # Shell/code examples may contain Markdown-like syntax that is not a link.
    source = re.sub(r"```.*?```", "", source, flags=re.S)
    targets = re.findall(r"\]\(([^)\s]+)\)", source)
    targets += re.findall(r'(?:src|srcset)="([^"\s]+)"', source)
    broken = []
    for target in targets:
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        rel = Path(unquote(parsed.path))
        resolved = (path.parent / rel).resolve()
        if (rel.is_absolute() or not resolved.is_relative_to(root.resolve())
                or not resolved.exists()):
            broken.append(target)
    return broken


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def history_findings(root: Path) -> tuple[int, list[str]]:
    """Scan every reachable blob, including files removed from the tree."""
    count = 0
    findings = []
    for row in git(root, "rev-list", "--objects", "--all").splitlines():
        oid, _, _name = row.partition(b" ")
        object_id = oid.decode()
        if git(root, "cat-file", "-t", object_id).strip() != b"blob":
            continue
        count += 1
        for line, rule in secret_findings(
                git(root, "cat-file", "blob", object_id)):
            findings.append(f"history blob {object_id[:12]}:{line}: {rule}")
    return count, findings


def check(root: Path, history: bool = False) -> int:
    paths = sorted(set(git(root, "ls-files", "-co", "--exclude-standard",
                           "-z").decode().split("\0")) - {""})
    findings = []
    checked = 0
    for name in paths:
        path = root / name
        if path.is_symlink():
            findings.append(f"{name}: review symlink before publication")
            continue
        if not path.exists():
            continue
        if not path.is_file():
            continue
        checked += 1
        if private_path(Path(name)):
            findings.append(f"{name}: private or generated scratch path")
        for line, rule in secret_findings(path.read_bytes()):
            findings.append(f"{name}:{line}: {rule}")
        if path.suffix == ".md":
            for target in local_links(path, root):
                findings.append(f"{name}: unresolved local link {target}")
    if history:
        count, historical = history_findings(root)
        findings.extend(historical)
        print(f"Scanned {count} reachable history blobs for known secret patterns.")
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print(f"Checked {checked} publishable files: no known secret patterns, "
          "private scratch paths or broken local file links found.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--history", action="store_true",
                        help="also scan every reachable Git blob")
    args = parser.parse_args()
    return check(ROOT, args.history)


if __name__ == "__main__":
    raise SystemExit(main())
