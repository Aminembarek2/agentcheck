"""Tests for unified-diff parsing.

Structure, not string lists. Every test here exists because a detector
needed something the flat representation could not express.

    pytest test_diffparse.py -v
"""

from agentcheck.diffparse import (
    change_signature,
    is_test_path,
    normalise,
    parse_diff,
)


def d(path: str, body: str, header: str = "") -> str:
    return (f"diff --git a/{path} b/{path}\n{header}"
            f"--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,3 @@\n{body}")


# --- basics -----------------------------------------------------------------

def test_splits_files_and_lines():
    diff = d("a.py", "+added\n-removed\n context\n") + d("b.py", "+other\n")
    files = parse_diff(diff)
    assert [f.path for f in files] == ["a.py", "b.py"]
    assert files[0].added_text == ["added"]
    assert files[0].removed_text == ["removed"]


def test_line_numbers_are_tracked():
    diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -10,3 +10,4 @@\n keep\n-gone\n+new1\n+new2\n more\n")
    hunk = parse_diff(diff)[0].hunks[0]
    kept, gone, new1, new2, more = hunk.lines
    assert (kept.old_lineno, kept.new_lineno) == (10, 10)
    assert (gone.old_lineno, gone.new_lineno) == (11, None)
    assert (new1.old_lineno, new1.new_lineno) == (None, 11)
    assert (new2.new_lineno, more.old_lineno, more.new_lineno) == (12, 12, 13)


def test_hunk_old_range_covers_context():
    diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -40,5 +40,5 @@\n a\n b\n-c\n+d\n e\n")
    assert list(parse_diff(diff)[0].hunks[0].old_lines) == [40, 41, 42, 43, 44]


def test_created_file_is_recognised():
    """A file the agent CREATED. `git diff` shows nothing for these unless
    the sandbox marks them intent-to-add first, and an invisible file is an
    undetectable cheat."""
    diff = ("diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1,1 @@\n+import sys\n")
    f = parse_diff(diff)[0]
    assert f.is_new
    assert f.added_text == ["import sys"]


def test_deleted_file_is_recognised():
    diff = ("diff --git a/tests/test_x.py b/tests/test_x.py\n"
            "deleted file mode 100644\n--- a/tests/test_x.py\n+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n-def test_a():\n")
    assert parse_diff(diff)[0].is_deleted


def test_extended_headers_do_not_confuse_the_parser():
    diff = ("diff --git a/x.py b/x.py\nold mode 100644\nnew mode 100755\n"
            "index abc123..def456 100644\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1 +1 @@\n-a\n+b\n")
    f = parse_diff(diff)[0]
    assert f.added_text == ["b"] and f.removed_text == ["a"]


def test_multiple_hunks_stay_separate():
    diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1,2 +1,2 @@\n-a\n+b\n"
            "@@ -50,2 +50,2 @@\n-c\n+d\n")
    hunks = parse_diff(diff)[0].hunks
    assert len(hunks) == 2
    assert hunks[1].old_start == 50


# --- post image -------------------------------------------------------------

def test_post_image_is_the_resulting_file_not_the_diff():
    """The view a detector needs when it asks "what follows this line".
    Reading adjacency off the added-lines list pairs the end of one hunk
    with the start of the next."""
    diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1,3 +1,4 @@\n try:\n+    go()\n-    old()\n     keep\n")
    post = [l.text for l in parse_diff(diff)[0].hunks[0].post_image()]
    assert post == ["try:", "    go()", "    keep"]


# --- test-file recognition --------------------------------------------------

def test_pytest_collectable_names_are_test_files():
    assert is_test_path("tests/test_db.py")
    assert is_test_path("test_db.py")
    assert is_test_path("db_test.py")
    assert is_test_path("conftest.py")
    assert is_test_path("src/tests/helpers.py")


def test_substring_matches_are_not_test_files():
    """The old rule was `"test" in basename`, which claimed all of these."""
    assert not is_test_path("latest_schema.py")
    assert not is_test_path("fastest.py")
    assert not is_test_path("contest.py")
    assert not is_test_path("databases/core.py")


# --- signatures -------------------------------------------------------------

def test_normalise_ignores_indentation_and_trailing_space():
    assert normalise("    x = 1  ") == normalise("x  =  1")


def test_change_signature_groups_by_file():
    diff = d("tests/test_a.py", "+new\n-old\n")
    sig = change_signature(parse_diff(diff))
    assert sig["tests/test_a.py"]["added"] == {"new"}
    assert sig["tests/test_a.py"]["removed"] == {"old"}
