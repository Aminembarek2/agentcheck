"""The guards in the calibration CLI.

Each of these enforces an ordering that the numbers depend on. They are
tested at the boundary where they refuse, because a guard that has never
been observed to refuse is a guard nobody knows works.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentcheck.calibration import LABEL_SCHEMA_VERSION, SKIP
from scripts.validate_judge import (
    MIN_RATIONALE,
    _git_hash,
    _label_entry,
    _read,
    _write,
    report,
    run_judge,
)


def labels_file(tmp_path: Path, pairs: list[dict], **over) -> Path:
    data = {
        "version": LABEL_SCHEMA_VERSION,
        "created": "2026-09-01T00:00:00+00:00",
        "protocol": "docs/judge-protocol.md",
        "protocol_git_hash": "deadbeefcafe",
        "frame": {"seed": 0, "n_requested": 48, "floor": 2,
                  "frame_size": len(pairs), "candidates": [], "picked": [],
                  "strata": {}},
        "pairs": pairs,
    }
    data.update(over)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def pair(pid="t::r", human=None, judge=None, model="deepseek-v4-flash",
         position="a"):
    return {"id": pid, "task_id": "002-databases", "run": pid.split("::")[-1],
            "path": f"runs/{pid}.json", "model": model,
            "agent_diff": "diff --git a/x b/x", "agent_position": position,
            "human": human, "relabel": None, "judge": judge or {}}


def human(neutral="equivalent", rationale="x" * MIN_RATIONALE):
    return {"neutral": neutral, "agent_relative": "equivalent",
            "rationale": rationale, "at": "2026-09-01T00:00:00+00:00"}


def finish_retest(path, days=7):
    """Complete the blind pass on synthetic fixtures only."""
    from datetime import datetime, timedelta

    from agentcheck.calibration import retest_sample

    data = _read(path)
    for p, _ in retest_sample(data):
        p["relabel"] = dict(p["human"])
        p["relabel"]["at"] = (
            datetime.fromisoformat(p["human"]["at"]) + timedelta(days=days)
        ).isoformat()
        p["relabel"]["days_after"] = days
    _write(path, data)


# --- the ordering guard -----------------------------------------------------

def test_the_judge_refuses_to_run_on_an_unlabelled_pair(tmp_path, capsys):
    """The single most important guard in the calibration.

    Judging first and labelling afterwards measures how persuasive the
    judge is, not whether it is right. This must fail before any model is
    constructed — an API key is not required to be told no.
    """
    path = labels_file(tmp_path, [pair("a", human()), pair("b", None)])
    assert run_judge(path, "sonnet", 5, allow_self=False) == 1
    assert "REFUSING" in capsys.readouterr().err


def test_running_the_judge_cannot_alter_the_labels(tmp_path):
    """A refused run must leave the expensive artifact byte-identical."""
    path = labels_file(tmp_path, [pair("a", human()), pair("b", None)])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    run_judge(path, "sonnet", 5, allow_self=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# --- self-judging -----------------------------------------------------------

def test_a_model_may_not_grade_its_own_patches(tmp_path, capsys):
    """Self-preference would be indistinguishable from calibration here."""
    path = labels_file(tmp_path,
                       [pair("a", human(), model="moonshotai/kimi-k2.6")])
    assert run_judge(path, "kimi", 5, allow_self=False) == 1
    err = capsys.readouterr().err
    assert "REFUSING" in err and "self" in err.lower()


def test_self_judging_is_possible_but_never_silent(tmp_path, capsys,
                                                   monkeypatch):
    """The escape hatch exists; it must not be quiet.

    Stops at model construction — no API key in the test environment — but
    only after the self-judging guard has been passed deliberately.
    """
    path = labels_file(tmp_path,
                       [pair("a", human(), model="moonshotai/kimi-k2.6")])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run_judge(path, "kimi", 5, allow_self=True,
                  require_committed=False)


def test_an_unknown_judge_model_is_rejected(tmp_path, capsys):
    path = labels_file(tmp_path, [pair("a", human())])
    assert run_judge(path, "gpt-9", 5, allow_self=False) == 1
    assert "unknown model" in capsys.readouterr().err


# --- the labels file itself -------------------------------------------------

def test_an_older_labels_file_is_refused_not_guessed_at(tmp_path):
    """Hand labels are too expensive to reinterpret under new assumptions."""
    path = labels_file(tmp_path, [pair("a", human())], version=1)
    with pytest.raises(SystemExit, match="v1"):
        _read(path)


def test_writes_are_atomic(tmp_path):
    """Ctrl-C during a save must not cost the labels already entered."""
    path = tmp_path / "labels.json"
    _write(path, {"version": LABEL_SCHEMA_VERSION, "pairs": []})
    assert json.loads(path.read_text())["pairs"] == []
    assert not list(tmp_path.glob("*.tmp")), "no temp file may survive"


def test_the_neutral_label_and_its_agent_relative_form_are_both_kept():
    """Storing only the derived form would bake a mapping bug in silently."""
    entry = _label_entry(pair(position="b"), "a_narrower", "because " * 4)
    assert entry["neutral"] == "a_narrower"
    # The agent was patch B, so "A is narrower" means the REFERENCE was
    # narrower, i.e. the agent did more.
    assert entry["agent_relative"] == "agent_wider"


def test_a_skip_is_not_translated_into_a_verdict():
    entry = _label_entry(pair(), SKIP, "nothing to judge here")
    assert entry["neutral"] == SKIP
    assert entry["agent_relative"] == SKIP


def test_every_label_carries_a_timestamp():
    entry = _label_entry(pair(), "equivalent", "same call sites changed")
    assert entry["at"].startswith("20")


# --- report -----------------------------------------------------------------

def test_the_report_refuses_before_the_judge_has_run(tmp_path, capsys):
    path = labels_file(tmp_path, [pair("a", human())])
    assert report(path, None, 0, tmp_path / "out.md") == 1
    assert "no judge has run" in capsys.readouterr().err


def test_the_report_marks_an_undersized_sample_as_a_pilot(tmp_path):
    """n=3 must never be presented as the calibration the protocol describes."""
    judged = {"sonnet": {"verdict": "equivalent", "agreement": 1.0,
                         "n_calls": 10, "model_id": "claude-sonnet-5",
                         "self_judged": False, "is_stable": True,
                         "position_bias": False, "n_unparseable": 0,
                         "raw": [{"reasoning": "same edits"}]}}
    pairs = [pair(f"p{i}", human(), dict(judged), position="ab"[i % 2])
             for i in range(3)]
    path = labels_file(tmp_path, pairs)
    finish_retest(path)
    out = tmp_path / "out.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert "PILOT" in text
    assert "not a calibration" in text


def test_the_report_names_the_protocol_hash_it_ran_under(tmp_path):
    judged = {"sonnet": {"verdict": "equivalent", "agreement": 1.0,
                         "n_calls": 10, "model_id": "claude-sonnet-5",
                         "self_judged": False, "is_stable": True,
                         "position_bias": False, "n_unparseable": 0,
                         "raw": []}}
    pairs = [pair(f"p{i}", human(), dict(judged), position="ab"[i % 2])
             for i in range(4)]
    path = labels_file(tmp_path, pairs)
    finish_retest(path)
    out = tmp_path / "out.md"
    report(path, None, 0, out)
    assert "deadbeefcafe" in out.read_text()


def test_the_report_flags_self_judged_pairs(tmp_path):
    judged = {"sonnet": {"verdict": "equivalent", "agreement": 1.0,
                         "n_calls": 10, "model_id": "claude-sonnet-5",
                         "self_judged": True, "is_stable": True,
                         "position_bias": False, "n_unparseable": 0,
                         "raw": []}}
    pairs = [pair(f"p{i}", human(), dict(judged), position="ab"[i % 2])
             for i in range(4)]
    path = labels_file(tmp_path, pairs)
    finish_retest(path)
    out = tmp_path / "out.md"
    report(path, None, 0, out)
    assert "self-judged" in out.read_text()


# --- pre-registration -------------------------------------------------------

def test_the_committed_protocol_is_findable():
    """`build` refuses without this, so it must actually resolve."""
    assert _git_hash(Path("docs/judge-protocol.md")), \
        "docs/judge-protocol.md must be committed — build refuses otherwise"


def test_an_uncommitted_path_has_no_hash(tmp_path):
    assert _git_hash(tmp_path / "nope.md") == ""


# --- the annotator and the judge must read the same text --------------------

def test_the_human_view_truncates_exactly_as_the_judge_does(tmp_path,
                                                            monkeypatch):
    """Kappa compares two raters. It only means anything if they were
    shown the same patches.

    The labelling view cut every patch to 120 LINES while the judge reads
    60,000 CHARACTERS — every patch in the sample, whole. Task 002's
    reference diff is 1,370 lines, so the annotator was labelling 9% of
    what the judge read, and the disagreement that produced would have
    been reported as disagreement about the fixes.
    """
    from agentcheck.judge import MAX_DIFF_CHARS
    from scripts.validate_judge import _show_diff

    huge = "\n".join(f"+line {i}" for i in range(5000))
    assert len(huge.splitlines()) > 120
    shown = _show_diff(huge, "PATCH A")

    assert "line 4999" in shown, "the tail of the patch must be visible"
    assert "more lines" not in shown, "no line-based cut may survive"
    # Below the judge's budget: nothing is dropped at all.
    assert len(huge) < MAX_DIFF_CHARS
    for line in ("+line 0", "+line 2500", "+line 4999"):
        assert line in shown


def test_a_patch_over_the_judges_budget_is_cut_visibly_for_both():
    """Past 60,000 characters both raters lose the same middle, and both
    are told so in the text itself."""
    from agentcheck.judge import MAX_DIFF_CHARS
    from scripts.validate_judge import _show_diff

    over = "x" * (MAX_DIFF_CHARS + 5000)
    shown = _show_diff(over, "PATCH B")
    assert "omitted by the harness, not by the patch author" in shown


def test_the_patches_stay_on_screen_while_the_verdict_is_typed(monkeypatch,
                                                               capsys):
    """A pager was tried here and lost the evidence.

    `less` restores the terminal on quit, so both patches vanished the
    instant the verdict prompt appeared and the annotator would have been
    answering from memory. The view is printed, and `v` prints it again.
    """
    from scripts.validate_judge import REVIEW, _ask_verdict

    answers = iter([REVIEW, "equivalent"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    verdict = _ask_verdict("PATCH A\nthe whole diff\nPATCH B")

    assert verdict == "equivalent"
    assert capsys.readouterr().out.count("the whole diff") == 1, \
        "v must re-show the patches rather than being read as a verdict"


def test_the_review_key_can_never_be_recorded_as_a_verdict():
    from agentcheck.judge import NEUTRAL_VERDICTS
    from scripts.validate_judge import REVIEW, SKIP

    assert REVIEW not in NEUTRAL_VERDICTS and REVIEW != SKIP


# --- the offline sheet ------------------------------------------------------

def _sheet_fixture(tmp_path, monkeypatch):
    """A labels file with one pair, written where Task.load can find it."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "version": LABEL_SCHEMA_VERSION,
        "created": "2026-09-12T00:00:00+00:00",
        "protocol": "docs/judge-protocol.md",
        "protocol_git_hash": "0" * 40,
        "frame": {"seed": 0, "floor": 2, "frame_size": 1, "candidates": [],
                  "picked": [], "strata": {}},
        "pairs": [{
            "id": "002-databases::002-databases-or-ds-flash-budget-r01",
            "task_id": "002-databases",
            "run": "002-databases-or-ds-flash-budget-r01",
            "path": "runs/002-databases-or-ds-flash-budget-r01.json",
            "model": "deepseek/deepseek-v4-flash-0731",
            "agent_diff": "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n",
            "agent_position": "a", "human": None, "relabel": None,
            "judge": {},
        }],
    }))
    return labels


def test_the_sheet_carries_no_anchor_the_prompt_hides(tmp_path, monkeypatch):
    """A document read away from the terminal is still a labelling view.

    The run name alone carries the model and the configuration — two of
    the four anchors judge-protocol.md §4 hides — so a sheet that named
    the run would unblind every pair in it at once.
    """
    from scripts.validate_judge import sheet

    labels = _sheet_fixture(tmp_path, monkeypatch)
    out = tmp_path / "pairs.md"
    assert sheet(labels, out, include_labelled=False) == 0

    text = out.read_text()
    pair = json.loads(labels.read_text())["pairs"][0]
    for anchor in (pair["run"], pair["model"], pair["path"], pair["id"]):
        assert anchor not in text, f"{anchor} must not reach the sheet"
    assert "PATCH A" in text and "PATCH B" in text


def test_the_sheet_and_the_prompt_name_a_pair_the_same_way(tmp_path,
                                                           monkeypatch):
    """The token is the only way to tell that a verdict written against a
    sheet is going into the pair the tool is asking about."""
    from scripts.validate_judge import _token, sheet

    labels = _sheet_fixture(tmp_path, monkeypatch)
    out = tmp_path / "pairs.md"
    sheet(labels, out, include_labelled=False)

    pair = json.loads(labels.read_text())["pairs"][0]
    assert _token(pair) in out.read_text()
    assert _token(pair) not in pair["id"], "a token is a hash, not a slice"


# --- importing a filled-in sheet --------------------------------------------

def _two_pair_labels(tmp_path, monkeypatch):
    """Two pairs whose agent sits in OPPOSITE slots."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    labels = tmp_path / "labels.json"
    pairs = []
    for i, slot in enumerate(("a", "b"), 1):
        pairs.append({
            "id": f"002-databases::002-databases-or-ds-flash-budget-r0{i}",
            "task_id": "002-databases",
            "run": f"002-databases-or-ds-flash-budget-r0{i}",
            "path": f"runs/002-databases-or-ds-flash-budget-r0{i}.json",
            "model": "deepseek/deepseek-v4-flash-0731",
            "agent_diff": "--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n",
            "agent_position": slot, "human": None, "relabel": None,
            "judge": {},
        })
    labels.write_text(json.dumps({
        "version": LABEL_SCHEMA_VERSION, "created": "2026-09-12T00:00:00+00:00",
        "protocol": "docs/judge-protocol.md", "protocol_git_hash": "0" * 40,
        "frame": {"seed": 0, "floor": 2, "frame_size": 2, "candidates": [],
                  "picked": [], "strata": {}},
        "pairs": pairs,
    }))
    return labels, pairs


def _sheet_with(tmp_path, rows):
    text = ["| # | pair | task | verdict | why |", "|---|---|---|---|---|"]
    for i, (token, verdict, why) in enumerate(rows, 1):
        text.append(f"| {i} | `{token}` | `002-databases` | {verdict} | {why} |")
    path = tmp_path / "sheet.md"
    path.write_text("\n".join(text))
    return path


def test_an_imported_verdict_is_read_against_the_pairs_own_slot(tmp_path,
                                                                monkeypatch):
    """The same verdict means opposite things in opposite slots.

    `a_wrong` says the agent was wrong when the agent is in slot A, and
    says the REFERENCE was wrong when it is in slot B. Deriving that from
    row order, or from the sheet, would invert half the labels silently —
    which is why hand-editing the labels file is not offered.
    """
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    why = "the call sites differ in a way I can name"
    sheet_path = _sheet_with(tmp_path, [
        (_token(pairs[0]), "a_wrong", why),      # agent is A here
        (_token(pairs[1]), "a_wrong", why),      # and B here
    ])

    assert import_sheet(sheet_path, labels) == 0
    got = json.loads(labels.read_text())["pairs"]
    assert got[0]["human"]["agent_relative"] == "agent_wrong"
    assert got[1]["human"]["agent_relative"] == "reference_wrong"
    assert all(p["human"]["source"] == "sheet" for p in got)


def test_a_verdict_with_no_reason_stops_the_whole_import(tmp_path,
                                                         monkeypatch):
    """And nothing is written — a half-applied import leaves a labels file
    nobody can reason about."""
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    sheet_path = _sheet_with(tmp_path, [
        (_token(pairs[0]), "equivalent", "same fix, both cover core.py"),
        (_token(pairs[1]), "equivalent", "too short"),
    ])

    assert import_sheet(sheet_path, labels) == 1
    got = json.loads(labels.read_text())["pairs"]
    assert all(p["human"] is None for p in got), "nothing may be written"


def test_an_unknown_token_is_refused_rather_than_skipped(tmp_path,
                                                         monkeypatch):
    """A token that matches nothing means the sheet belongs to a different
    frame, and importing the rest of it would be importing half a sample."""
    from scripts.validate_judge import import_sheet

    labels, _pairs = _two_pair_labels(tmp_path, monkeypatch)
    sheet_path = _sheet_with(
        tmp_path, [("deadbe", "equivalent", "a reason long enough to pass")])
    assert import_sheet(sheet_path, labels) == 1


def test_an_import_never_overwrites_a_label(tmp_path, monkeypatch):
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    data = json.loads(labels.read_text())
    data["pairs"][0]["human"] = {"neutral": "equivalent",
                                 "agent_relative": "equivalent",
                                 "rationale": "labelled at the prompt",
                                 "at": "2026-09-12T16:04:33+00:00",
                                 "source": "prompt"}
    labels.write_text(json.dumps(data))

    sheet_path = _sheet_with(
        tmp_path, [(_token(pairs[0]), "different", "a reason long enough")])
    assert import_sheet(sheet_path, labels) == 1
    assert json.loads(labels.read_text())["pairs"][0]["human"]["neutral"] \
        == "equivalent"


def test_a_skip_needs_no_reason_and_a_pipe_in_one_survives(tmp_path,
                                                           monkeypatch):
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    sheet_path = _sheet_with(tmp_path, [
        (_token(pairs[0]), "skip", ""),
        (_token(pairs[1]), "equivalent", "uses the a|b spelling, same effect"),
    ])
    assert import_sheet(sheet_path, labels) == 0
    got = json.loads(labels.read_text())["pairs"]
    assert got[0]["human"]["agent_relative"] == SKIP
    assert "a|b" in got[1]["human"]["rationale"]


def test_a_blank_row_is_not_an_import(tmp_path, monkeypatch):
    """Filling the sheet in over several sittings has to be safe."""
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    sheet_path = _sheet_with(tmp_path, [
        (_token(pairs[0]), "equivalent", "both migrate the same call sites"),
        (_token(pairs[1]), "", ""),
    ])
    assert import_sheet(sheet_path, labels) == 0
    got = json.loads(labels.read_text())["pairs"]
    assert got[0]["human"] is not None and got[1]["human"] is None


def test_importing_the_same_sheet_twice_is_not_an_error(tmp_path, monkeypatch):
    """Filling in a few rows and importing, then a few more, is the point.

    The second import still carries the first batch's rows. Treating those
    as "already labelled, refusing" would make the whole iterative
    workflow fail on its second use — and the refusal is all-or-nothing,
    so the new rows would be rejected along with them.
    """
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    first = _sheet_with(tmp_path, [
        (_token(pairs[0]), "equivalent", "both migrate the same call sites"),
    ])
    assert import_sheet(first, labels) == 0

    both = _sheet_with(tmp_path, [
        (_token(pairs[0]), "equivalent", "both migrate the same call sites"),
        (_token(pairs[1]), "different", "a genuinely different approach"),
    ])
    assert import_sheet(both, labels) == 0
    got = json.loads(labels.read_text())["pairs"]
    assert got[0]["human"]["neutral"] == "equivalent"
    assert got[1]["human"]["neutral"] == "different"


def test_a_row_that_contradicts_a_recorded_label_still_stops_everything(
        tmp_path, monkeypatch):
    """Agreement is harmless; disagreement means two sheets out of step,
    or a changed mind that should be a deliberate edit."""
    from scripts.validate_judge import _token, import_sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    assert import_sheet(_sheet_with(tmp_path, [
        (_token(pairs[0]), "equivalent", "both migrate the same call sites"),
    ]), labels) == 0
    assert import_sheet(_sheet_with(tmp_path, [
        (_token(pairs[0]), "different", "on reflection they diverge a lot"),
    ]), labels) == 1
    assert json.loads(labels.read_text())["pairs"][0]["human"]["neutral"] \
        == "equivalent"


def test_regenerating_never_silently_eats_unimported_verdicts(tmp_path,
                                                              monkeypatch):
    """The sheet is not in git, and an unimported verdict exists nowhere
    else. A command that looks idempotent must not delete an hour of work.
    """
    from scripts.validate_judge import _token, sheet

    labels, pairs = _two_pair_labels(tmp_path, monkeypatch)
    out = tmp_path / "pairs.md"
    assert sheet(labels, out, include_labelled=False) == 0

    filled = out.read_text().replace(
        f"| 1 | `{_token(pairs[0])}` | `002-databases` |  |  |",
        f"| 1 | `{_token(pairs[0])}` | `002-databases` | equivalent | "
        f"both migrate the same call sites |")
    out.write_text(filled)

    assert sheet(labels, out, include_labelled=False) == 1, "must refuse"
    assert "equivalent" in out.read_text(), "and must not have touched it"

    assert sheet(labels, out, include_labelled=False, force=True) == 0
    assert "| 1 | `" in out.read_text()


def test_the_judge_refuses_labels_that_are_not_committed(tmp_path, capsys,
                                                          monkeypatch):
    """§10's condition, enforced rather than remembered.

    A labels file outside git — or with local edits — cannot prove it
    predates the judge's answers, and the refusal comes before a model is
    built, so it never costs a call.
    """
    from scripts.validate_judge import PROTOCOL

    path = labels_file(tmp_path, [pair("a", human())])
    # The protocol is committed; only the labels are not.
    monkeypatch.setattr("scripts.validate_judge._committed_clean",
                        lambda p: "f" * 40 if Path(p) == PROTOCOL else "")
    assert run_judge(path, "kimi", 5, allow_self=False) == 1
    err = capsys.readouterr().err
    assert "not committed" in err and "§10" in err


# --- what a judge run costs, and when it stops ------------------------------

class _Billed:
    """Stands in for a chat model: replies `equivalent`, reports usage."""

    def __init__(self, usage):
        self.usage = usage
        self.bodies = []
        self.model = "claude-haiku-4-5-20251001"

    def invoke(self, messages):
        self.bodies.append(messages[-1].content)
        usage = self.usage

        class R:
            content = json.dumps({"reasoning": "same", "verdict": "equivalent",
                                  "confidence": "high"})
            usage_metadata = usage
        return R()


def _judgeable(tmp_path, monkeypatch, n, usage):
    """n labelled pairs, committed as far as run_judge can tell, and a fake
    model in place of the real one."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    path = labels_file(tmp_path, [pair(f"t::r{i}", human()) for i in range(n)])
    fake = _Billed(usage)
    monkeypatch.setattr("scripts.validate_judge.build_model",
                        lambda alias, **kwargs: fake)
    monkeypatch.setattr("scripts.validate_judge._committed_clean",
                        lambda p: "f" * 40)
    return path, fake


EXPENSIVE = {"input_tokens": 400_000, "output_tokens": 1_000}


@pytest.mark.parametrize("alias,cached", [("haiku", True), ("sonnet", True),
                                          ("kimi", False), ("or-haiku", False)])
def test_the_cache_breakpoint_is_sent_only_to_anthropic(tmp_path, monkeypatch,
                                                        alias, cached):
    """Gated as run_agent gates it. On an OpenAI-compatible endpoint a
    cache_control block is a different content shape, not a no-op."""
    path, fake = _judgeable(tmp_path, monkeypatch, 1,
                            {"input_tokens": 10, "output_tokens": 10})
    run_judge(path, alias, 1, allow_self=False, allow_unregistered=True)
    assert fake.bodies, "the judge must have been called"
    assert all(isinstance(b, list) == cached for b in fake.bodies)


def test_a_judge_run_stops_at_its_spending_ceiling(tmp_path, monkeypatch,
                                                   capsys):
    """Between pairs, after the result is written, so nothing is lost and
    a rerun resumes at the next pair."""
    path, _fake = _judgeable(tmp_path, monkeypatch, 3, EXPENSIVE)
    assert run_judge(path, "kimi", 1, allow_self=False, max_cost=0.50) == 1
    assert "STOPPING" in capsys.readouterr().err

    data = json.loads(path.read_text())
    judged = [p for p in data["pairs"] if "kimi" in p["judge"]]
    assert 0 < len(judged) < 3, "stopped partway, keeping what it had"
    assert all(p["judge"]["kimi"]["cost_usd"] > 0 for p in judged)


def test_a_judge_run_that_cannot_count_its_spend_stops(tmp_path, monkeypatch,
                                                       capsys):
    """A ceiling over an unknown number is not a ceiling."""
    path, _fake = _judgeable(tmp_path, monkeypatch, 3, None)
    assert run_judge(path, "kimi", 1, allow_self=False) == 1
    assert "did not report token usage" in capsys.readouterr().err
    judged = [p for p in json.loads(path.read_text())["pairs"]
              if "kimi" in p["judge"]]
    assert len(judged) == 1


def test_a_judge_run_never_shows_a_verdict_against_a_pair(tmp_path,
                                                         monkeypatch, capsys):
    """These pairs are relabelled blind a week later.

    Printing the judge's verdict next to the pair's token — or its run
    name, which carries the model and configuration — would let the
    annotator walk into the retest remembering the judge's answer to that
    pair, inflating the self-agreement §10 compares the judge against.
    """
    from scripts.validate_judge import _token

    path, _fake = _judgeable(tmp_path, monkeypatch, 1,
                             {"input_tokens": 10, "output_tokens": 10})
    run_judge(path, "kimi", 1, allow_self=False)
    out = capsys.readouterr().out
    pair_ = json.loads(path.read_text())["pairs"][0]
    assert "equivalent" not in out, "the verdict must not reach the screen"
    assert _token(pair_) not in out and pair_["run"] not in out
    assert "calls parsed" in out and "$" in out, "health is still shown"


def test_the_retest_view_carries_no_token(monkeypatch, capsys):
    """The judge run has happened by the time of the retest."""
    from scripts.validate_judge import _render, _token

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    from agentcheck.task import Task
    p = pair("t::r1", human())
    view = _render(p, Task.load("002-databases"), "1/12", show_token=False)
    assert _token(p) not in view


# --- the retest delay -------------------------------------------------------

def _aged(pid, days):
    from datetime import datetime, timedelta, timezone
    at = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    h = human()
    h["at"] = at
    return pair(pid, h)


def test_an_old_label_does_not_unlock_the_retest_of_a_new_one(tmp_path,
                                                              capsys):
    """The delay belongs to each pair.

    Measured from the oldest label in the file, a pair labelled a month ago
    opened the retest of one labelled yesterday.
    """
    from scripts.validate_judge import relabel

    path = labels_file(tmp_path, [_aged("t::old", 30), _aged("t::new", 1)])
    assert relabel(path, 2, force=False) == 1
    assert "REFUSING" in capsys.readouterr().err
    assert all(p["relabel"] is None
               for p in json.loads(path.read_text())["pairs"])


def test_each_retest_records_its_own_interval(tmp_path, monkeypatch):
    """The report quotes days_after per pair; one global age stamped on
    every pair would state the wrong interval for nearly all of them."""
    from scripts.validate_judge import relabel

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    path = labels_file(tmp_path, [_aged("t::a", 30), _aged("t::b", 10)])
    answers = iter(["equivalent", "a reason that is long enough"] * 2)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert relabel(path, 2, force=False) == 0
    days = {p["id"]: p["relabel"]["days_after"]
            for p in json.loads(path.read_text())["pairs"]}
    assert days == {"t::a": 30, "t::b": 10}


# --- registered judges (protocol §9, amendment A1) --------------------------

def test_an_unregistered_judge_is_refused(tmp_path, capsys, monkeypatch):
    """A judge chosen after seeing a kappa is a judge chosen for its answer.

    Encoded like THRESHOLDS, so shopping for a friendlier judge is not
    discouraged but impossible without an amendment in git first.
    """
    path, _fake = _judgeable(tmp_path, monkeypatch, 1,
                             {"input_tokens": 10, "output_tokens": 10})
    assert run_judge(path, "sonnet", 1, allow_self=False) == 1
    assert "not a registered judge" in capsys.readouterr().err
    assert all(not p["judge"] for p in json.loads(path.read_text())["pairs"])


def test_the_registered_judges_are_the_ones_the_protocol_names():
    """Code and document must not drift: §9 is what a reader checks."""
    from agentcheck.calibration import REGISTERED_JUDGES
    from agentcheck.models import MODELS

    section = Path("docs/judge-protocol.md").read_text().split(
        "## 9.")[1].split("## 10.")[0]
    for alias in REGISTERED_JUDGES:
        assert alias in MODELS, f"{alias} has no model spec"
        assert f"`{alias}`" in section, f"§9 does not name {alias}"


def test_every_verdict_names_the_protocol_version_it_ran_under(tmp_path,
                                                               monkeypatch):
    """§11: a calibration run under an amended protocol says which version
    it followed."""
    path, _fake = _judgeable(tmp_path, monkeypatch, 1,
                             {"input_tokens": 10, "output_tokens": 10})
    run_judge(path, "kimi", 1, allow_self=False)
    entry = json.loads(path.read_text())["pairs"][0]["judge"]["kimi"]
    assert entry["protocol_git_hash"] == "f" * 40
    assert entry["registered_judge"] is True


def test_the_report_names_an_amended_protocol(tmp_path):
    """Labels written under one version, verdicts under another: both named."""
    labelled = human()
    judged = {"kimi": {"verdict": "equivalent", "distribution": {},
                       "by_order": {}, "raw": [], "agreement": 1.0,
                       "position_bias": False, "n_calls": 2, "n_parsed": 2,
                       "n_unparseable": 0, "truncated": [], "usable": True,
                       "is_stable": True, "model_id": "moonshotai/kimi-k2.6",
                       "self_judged": False, "at": "2026-09-13T00:00:00",
                       "protocol_git_hash": "a1" * 20,
                       "registered_judge": True}}
    path = labels_file(tmp_path, [pair("t::r", labelled, judge=judged)])
    finish_retest(path)
    out = tmp_path / "calibration.md"
    report(path, None, 0, out)
    text = out.read_text()
    assert "amended protocol" in text and "a1a1a1a1a1a1" in text
    assert "Blob identifiers alone do not prove event ordering" in text
    assert "committed before the first label" not in text


def test_retest_resume_keeps_the_sample_size_and_patch_order(tmp_path, monkeypatch):
    from agentcheck.calibration import retest_sample
    from scripts.validate_judge import relabel

    path = labels_file(tmp_path, [_aged(f"t::{i}", 10) for i in range(20)])
    intended = [(p["id"], slot) for p, slot in retest_sample(_read(path))]
    shown = []

    def render(p, *args, **kwargs):
        shown.append((p["id"], p["agent_position"]))
        return "blind view"

    monkeypatch.setattr("scripts.validate_judge._render", render)
    answers = iter(["equivalent", "a reason that is long enough"] * 3)

    def interrupted_input(*args):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", interrupted_input)
    assert relabel(path, 12, force=False) == 0
    assert sum(p["relabel"] is not None for p in _read(path)["pairs"]) == 3
    assert shown == intended[:4]  # Fourth pair was displayed before EOF.

    shown.clear()
    answers = iter(["equivalent", "a reason that is long enough"] * 9)
    assert relabel(path, 12, force=False) == 0
    assert shown == intended[3:]
    assert sum(p["relabel"] is not None for p in _read(path)["pairs"]) == 12
    shown.clear()
    assert relabel(path, 12, force=False) == 0
    assert not shown, "a completed retest must not draw another twelve pairs"


def test_report_cannot_expose_verdicts_before_retest(tmp_path, capsys):
    path = labels_file(tmp_path, [pair(judge={"kimi": {"verdict": "SECRET"}},
                                     human=human())])
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 1
    captured = capsys.readouterr()
    assert "blind retest" in captured.err
    assert "SECRET" not in captured.out + captured.err
    assert not out.exists()


def test_status_is_safe_before_retest(tmp_path, capsys):
    from scripts.validate_judge import status

    path = labels_file(tmp_path, [pair(human=human(rationale="HUMAN_SECRET"),
                                     judge={"kimi": {"verdict": "JUDGE_SECRET",
                                                      "cost_usd": 0.12,
                                                      "cost_known": True}})])
    assert status(path) == 0
    out = capsys.readouterr().out
    assert "1/1" in out and "$0.12" in out and "2026-09-08" in out
    assert "SECRET" not in out


def calibration_fixture(tmp_path):
    """A complete synthetic calibration with two balanced classes."""
    pairs = []
    for i in range(48):
        h = human()
        h["agent_relative"] = "equivalent" if i % 2 else "agent_wrong"
        j = {"verdict": h["agent_relative"], "model_id": "moonshotai/kimi-k2.6",
             "self_judged": False, "n_calls": 10, "agreement": 1.0,
             "labels_git_hash": "f" * 40, "protocol_git_hash": "e" * 40,
             "registered_judge": True}
        pairs.append(pair(f"p{i}", h, {"kimi": j}, position="ab"[i % 2]))
    path = labels_file(tmp_path, pairs)
    data = _read(path)
    data["frame"]["candidates"] = [{"id": "p0", "stratum": ["t", "none", True]}]
    _write(path, data)
    finish_retest(path)
    return path


def test_report_applies_human_ceiling_before_licensing_findings(tmp_path):
    from agentcheck.calibration import retest_sample

    path = calibration_fixture(tmp_path)
    data = _read(path)
    p, _ = retest_sample(data)[0]
    p["relabel"]["agent_relative"] = (
        "agent_wrong" if p["human"]["agent_relative"] == "equivalent"
        else "equivalent")
    _write(path, data)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert "below the judge's kappa" in text
    assert "no judge findings are licensed" in text
    assert "judge verdicts may be reported as findings" not in text


def test_report_licenses_a_complete_valid_calibration(tmp_path):
    path = calibration_fixture(tmp_path)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    assert "judge verdicts may be reported as findings" in out.read_text()


def test_early_retest_cannot_be_hidden_by_a_cached_interval(tmp_path):
    from agentcheck.calibration import retest_sample

    path = calibration_fixture(tmp_path)
    data = _read(path)
    p, _ = retest_sample(data)[0]
    p["relabel"]["at"] = p["human"]["at"]
    p["relabel"]["days_after"] = 30
    _write(path, data)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert "before seven days" in text and "0–7 days" in text
    assert "judge verdicts may be reported as findings" not in text


@pytest.mark.parametrize("fault,expected", [
    ("incomplete", "judge coverage is incomplete"),
    ("provenance", "provenance is missing"),
    ("repeats", "five repeats in both orders"),
])
def test_a_high_kappa_cannot_override_protocol_requirements(tmp_path, fault,
                                                           expected):
    path = calibration_fixture(tmp_path)
    data = _read(path)
    p = data["pairs"][0]
    if fault == "incomplete":
        p["judge"] = {}
    elif fault == "provenance":
        p["judge"]["kimi"].pop("labels_git_hash")
    else:
        p["judge"]["kimi"]["n_calls"] = 2
    _write(path, data)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert expected in text
    assert "judge verdicts may be reported as findings" not in text


def test_both_judges_are_compared_on_their_common_pairs(tmp_path):
    path = calibration_fixture(tmp_path)
    data = _read(path)
    for p in data["pairs"]:
        p["judge"]["mimo"] = dict(p["judge"]["kimi"],
                                   model_id="xiaomi/mimo-v2.5-pro")
    _write(path, data)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert "Both judges evaluated 48/48" in text
    assert "kimi vs mimo, collapsed binary" in text


def test_non_verdicts_produce_an_inconclusive_report(tmp_path):
    path = calibration_fixture(tmp_path)
    data = _read(path)
    for p in data["pairs"]:
        p["judge"]["kimi"]["verdict"] = "unparseable"
    _write(path, data)
    out = tmp_path / "report.md"
    assert report(path, None, 0, out) == 0
    text = out.read_text()
    assert "all 48 pairs excluded" in text
    assert "Calibration inconclusive" in text


def test_probe_keeps_source_unchanged_and_never_shows_verdicts(tmp_path,
                                                            monkeypatch, capsys):
    path, fake = _judgeable(tmp_path, monkeypatch, 5,
                            {"input_tokens": 10, "output_tokens": 10})
    requested = []

    def build(alias, **kwargs):
        requested.append(kwargs["max_tokens"])
        return fake

    monkeypatch.setattr("scripts.validate_judge.build_model", build)
    before = path.read_bytes()
    output = tmp_path / "diagnostic.json"
    assert run_judge(path, "kimi", 1, allow_self=False, max_tokens=32768,
                     diagnostic_out=output, diagnostic_pairs=3) == 0
    assert path.read_bytes() == before
    data = _read(output)
    assert requested == [32768] and len(fake.bodies) == 6
    assert data["diagnostic"]["pairs_requested"] == 3
    assert sum("kimi" in p["judge"] for p in data["pairs"]) == 3
    text = capsys.readouterr().out
    assert "6/6 calls parsed" in text
    assert "equivalent" not in text
    assert report(output, None, 0, tmp_path / "report.md") == 1
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize("destination", ["source", "existing"])
def test_probe_refuses_to_overwrite_files(tmp_path, monkeypatch, destination):
    path, fake = _judgeable(tmp_path, monkeypatch, 1,
                            {"input_tokens": 10, "output_tokens": 10})
    output = path if destination == "source" else tmp_path / "previous.json"
    if destination == "existing":
        output.write_text("keep me")
    before = output.read_bytes()
    assert run_judge(path, "kimi", 1, allow_self=False,
                     diagnostic_out=output) == 1
    assert output.read_bytes() == before
    assert not fake.bodies


def test_changed_token_limit_cannot_be_pooled_with_previous_results(tmp_path,
                                                                  monkeypatch):
    path, fake = _judgeable(tmp_path, monkeypatch, 2,
                            {"input_tokens": 10, "output_tokens": 10})
    assert run_judge(path, "kimi", 1, allow_self=False) == 0
    data = _read(path)
    data["pairs"][1]["judge"].clear()
    _write(path, data)
    before = path.read_bytes()
    fake.bodies.clear()
    assert run_judge(path, "kimi", 1, allow_self=False, max_tokens=32768) == 1
    assert not fake.bodies and path.read_bytes() == before


def test_high_parse_failure_stops_before_spending_on_another_pair(tmp_path,
                                                                monkeypatch):
    from types import SimpleNamespace

    path, fake = _judgeable(tmp_path, monkeypatch, 3,
                            {"input_tokens": 10, "output_tokens": 8192})

    def invoke(messages):
        fake.bodies.append(messages)
        return SimpleNamespace(content="", usage_metadata=fake.usage,
                               response_metadata={"finish_reason": "length"})

    fake.invoke = invoke
    assert run_judge(path, "kimi", 5, allow_self=False) == 1
    assert len(fake.bodies) == 10
    saved = [p["judge"]["kimi"] for p in _read(path)["pairs"] if "kimi" in p["judge"]]
    assert len(saved) == 1 and saved[0]["n_unparseable"] == 10
    assert all(c["finish_reason"] == "length" for c in saved[0]["raw"])


@pytest.mark.parametrize("change", ["judge_output", "human", "diff", "seed"])
def test_resume_proves_committed_inputs_without_committing_judge_output(
        tmp_path, monkeypatch, change):
    from types import SimpleNamespace

    from scripts.validate_judge import _committed_labels

    path = labels_file(tmp_path, [pair(human=human())])
    baseline = path.read_text()
    data = _read(path)
    if change == "judge_output":
        data["pairs"][0]["judge"]["kimi"] = {"n_calls": 10}
    elif change == "human":
        data["pairs"][0]["human"]["rationale"] = "a changed human rationale"
    elif change == "diff":
        data["pairs"][0]["agent_diff"] = "a different input patch"
    else:
        data["frame"]["seed"] = 999
    _write(path, data)
    monkeypatch.setattr("scripts.validate_judge._committed_clean", lambda p: "")
    monkeypatch.setattr("scripts.validate_judge._git_hash", lambda p: "f" * 40)
    monkeypatch.setattr("scripts.validate_judge.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=0, stdout=baseline))
    expected = "f" * 40 if change == "judge_output" else ""
    assert _committed_labels(path) == expected
