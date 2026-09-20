"""Tests for the run record.

Each test corresponds to a bug that actually shipped. The point of the
schema is that these cannot recur silently.

    pytest test_record.py -v
"""

import json

import pytest

from agentcheck.record import (
    SCHEMA_VERSION,
    ConfigFingerprint,
    IncompatibleRecord,
    RunRecord,
    group_by_config,
    holds_measurement,
    load,
    load_all,
)
from agentcheck.scorer import Finding

BASE = dict(
    schema_version=SCHEMA_VERSION,
    config_version="abc123def456",
    task_id="002-databases",
    model="deepseek-v4-flash",
    outcome="solved",
    verdict_status="ok",
    diff="",
    failed_ids=[],
    before_failed_ids=["tests/test_db.py::test_1"],
    iterations=10,
    cost_usd=0.5,
    cost_known=True,
    max_iterations=50,
    max_cost_usd=1.0,
)


def write(tmp_path, name="r.json", **over):
    rec = dict(BASE)
    rec.update(over)
    for k, v in list(over.items()):
        if v is None:
            rec.pop(k, None)
    p = tmp_path / name
    p.write_text(json.dumps(rec))
    return p


def fingerprint(**over):
    base = dict(
        model_id="m", tool_signature=(("read_file", "Read a file."),),
        system_prompt="p", max_iterations=50, max_cost_usd=1.0,
        max_wall_seconds=1800.0, test_command="pytest tests/",
        image_id="sha256:abc", max_tokens=8192,
        max_file_lines=400, max_search_hits=40,
    )
    base.update(over)
    return ConfigFingerprint(**base)


# --- the bugs that shipped --------------------------------------------------

def test_missing_failed_ids_is_rejected_not_defaulted(tmp_path):
    """Seven runs claimed 100% because a missing failed_ids read as
    'nothing is failing'. Absent must not be a legal value."""
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, failed_ids=None))
    assert "failed_ids" in str(e.value)


def test_missing_before_state_is_rejected(tmp_path):
    """One re-scoring script passed an empty before-state and flattened
    every stored progress number to 0%. The before-state now lives in the
    record, so it cannot be passed empty — or be absent."""
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, before_failed_ids=None))
    assert "before_failed_ids" in str(e.value)


def test_missing_verdict_status_is_rejected(tmp_path):
    """Zero failures and no verdict are the same count and opposite
    conclusions. The record must say which one it is."""
    with pytest.raises(IncompatibleRecord):
        load(write(tmp_path, verdict_status=None))


def test_missing_caps_are_rejected(tmp_path):
    """A stored run says `cost_limit` at $2.08 with nothing recording what
    the cap was, which makes the label unfalsifiable."""
    with pytest.raises(IncompatibleRecord):
        load(write(tmp_path, max_cost_usd=None))


def test_old_schema_is_rejected_with_a_reason(tmp_path):
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, schema_version=2))
    assert "Re-run rather than re-score" in str(e.value)


def test_a_newer_schema_is_also_rejected(tmp_path):
    """`version < CURRENT` let a record from a LATER harness through and
    treated it as current."""
    with pytest.raises(IncompatibleRecord):
        load(write(tmp_path, schema_version=SCHEMA_VERSION + 1))


def test_unversioned_record_is_rejected(tmp_path):
    with pytest.raises(IncompatibleRecord):
        load(write(tmp_path, schema_version=None))


def test_missing_config_version_is_rejected(tmp_path):
    """A run whose producing configuration is unknown cannot be compared
    to anything, so it must not enter an aggregate."""
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, config_version=None))
    assert "config_version" in str(e.value)


def test_an_unknown_field_is_rejected(tmp_path):
    """A field this harness does not know about means the writer was a
    different version than the schema number claims."""
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, mystery_field=1))
    assert "mystery_field" in str(e.value)


def test_corrupt_json_is_rejected_with_the_filename(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(IncompatibleRecord) as e:
        load(p)
    assert "broken.json" in str(e.value)


def test_valid_record_loads(tmp_path):
    r = load(write(tmp_path, failed_ids=["a::b"]))
    assert r.failed_ids == ["a::b"]
    assert r.is_current and r.verdict_ok


# --- writing ----------------------------------------------------------------

def test_saving_without_a_config_version_is_refused(tmp_path):
    r = RunRecord(config_version="", task_id="t", model="m", outcome="solved")
    with pytest.raises(ValueError):
        r.save(tmp_path / "x.json")


def test_a_saved_record_loads_back(tmp_path):
    r = RunRecord(config_version="cfg", task_id="t", model="m",
                  outcome="solved", verdict_status="ok",
                  before_failed_ids=["t::a"], max_iterations=50,
                  max_cost_usd=1.0)
    r.save(tmp_path / "x.json")
    assert load(tmp_path / "x.json").config_version == "cfg"


# --- configuration identity -------------------------------------------------

def test_same_config_hashes_the_same():
    assert fingerprint().digest() == fingerprint().digest()


def test_tool_order_is_not_a_configuration_change():
    a = fingerprint(tool_signature=(("a", "d1"), ("b", "d2")))
    b = fingerprint(tool_signature=(("b", "d2"), ("a", "d1")))
    assert a.digest() == b.digest()


def test_prompt_change_changes_the_hash():
    """v3 and v4 differed only in the prompt. Nothing but a filename
    recorded that, and a mistyped label would have merged them."""
    assert fingerprint(system_prompt="read first").digest() != \
        fingerprint(system_prompt="edit first").digest()


def test_a_tool_DESCRIPTION_change_changes_the_hash():
    """Descriptions are prompt text the model reads on every call.
    Hashing names alone let one be rewritten with nothing noticing."""
    a = fingerprint(tool_signature=(("package_version", "Report a version."),))
    b = fingerprint(tool_signature=(("package_version", "setup.py is STALE."),))
    assert a.digest() != b.digest()


@pytest.mark.parametrize("field,value", [
    ("model_id", "claude-opus-5"),
    ("max_iterations", 25),
    ("max_cost_usd", 2.0),
    ("max_wall_seconds", 600.0),
    ("test_command", "pytest tests/ --ignore=tests/test_x.py"),
    ("image_id", "sha256:rebuilt"),
    ("max_tokens", 4096),
    ("max_file_lines", 200),
    ("max_search_hits", 10),
])
def test_every_behaviour_affecting_input_changes_the_hash(field, value):
    """Each of these changes the result while leaving every other recorded
    field identical, so two runs differing only in it would silently merge
    into one column. The image id in particular can change with no change
    to any tracked file — a rebuild is a different experiment."""
    assert fingerprint().digest() != fingerprint(**{field: value}).digest()


# --- aggregation ------------------------------------------------------------

def test_rejections_are_reported_not_dropped(tmp_path):
    """A silently shrinking denominator is itself a way to fabricate a
    result."""
    good = write(tmp_path, "good.json")
    bad = write(tmp_path, "bad.json", failed_ids=None)
    records, rejected = load_all([good, bad])
    assert len(records) == 1 and len(rejected) == 1
    assert "bad.json" in rejected[0]


def test_configs_are_kept_apart(tmp_path):
    a = load(write(tmp_path, "a.json", config_version="aaa"))
    b = load(write(tmp_path, "b.json", config_version="bbb"))
    c = load(write(tmp_path, "c.json", config_version="aaa"))
    groups = group_by_config([a, b, c])
    assert set(groups) == {("002-databases", "aaa"), ("002-databases", "bbb")}
    assert len(groups[("002-databases", "aaa")]) == 2


def test_tasks_are_kept_apart_too(tmp_path):
    """Two tasks have different denominators. Averaging across them is not
    a mean either."""
    a = load(write(tmp_path, "a.json", task_id="001-fastapi-users"))
    b = load(write(tmp_path, "b.json", task_id="002-databases"))
    assert len(group_by_config([a, b])) == 2


# --- types, not just presence -----------------------------------------------

@pytest.mark.parametrize("field,value,why", [
    ("failed_ids", "a,b",
     "a bare string iterates as CHARACTERS: three failing tests from a typo"),
    ("before_failed_ids", "x", "same, on the other end of the measurement"),
    ("failed_ids", ["a", "a"], "duplicate ids make the count and the set disagree"),
    ("cost_usd", "0.5", "a string cost concatenates in a sum instead of adding"),
    ("cost_usd", -5.0, "a negative cost is not a cost"),
    ("iterations", -3, "impossible"),
    ("cost_known", "yes", "a truthy string reads as True and hides an unknown cost"),
    ("verdict_status", "maybe", "there are exactly two states"),
    ("config_version", "   ", "whitespace groups as its own configuration"),
    ("config_version", 12345, "not a string"),
    ("tool_calls", {"read_file": None}, "counts must be integers"),
    ("trajectory", {"a": 1}, "must be a list"),
    ("diff", None, "parse_diff(None) is not a diff"),
])
def test_a_field_of_the_wrong_type_is_rejected(tmp_path, field, value, why):
    """Presence alone is not enough, and the gap is not theoretical.

    The rule of this module is that absent is never zero. The same
    argument applies to a value of the wrong shape: it is not data, and
    guessing what was meant is how a wrong number gets published.
    """
    with pytest.raises(IncompatibleRecord):
        load(write(tmp_path, **{field: value}))


def test_an_unrecognised_outcome_is_rejected(tmp_path):
    """An outcome no summary knows about matches no category and vanishes
    silently from every count."""
    with pytest.raises(IncompatibleRecord) as e:
        load(write(tmp_path, outcome="banana"))
    assert "not one of" in str(e.value)


def test_the_exit_code_survives_a_round_trip(tmp_path):
    """It is what separates an agent-broken suite from a dead container.

    Recovering it by parsing the reason prose is a migration shim for old
    records, not a mechanism anything new may rely on.
    """
    r = RunRecord(config_version="c", task_id="t", model="m",
                  outcome="broke_suite", verdict_status="no_verdict",
                  verdict_reason="pytest exited 2 — the suite could not run",
                  verdict_exit_code=2, before_failed_ids=["t::a"])
    loaded = load(r.save(tmp_path / "r.json"))
    assert loaded.verdict_exit_code == 2


def test_a_record_written_before_exit_codes_were_kept_still_loads(tmp_path):
    """Absent is None, never 0. Zero is pytest's code for a green suite."""
    path = write(tmp_path, outcome="no_verdict", verdict_status="no_verdict")
    assert load(path).verdict_exit_code is None


def test_a_string_exit_code_is_rejected_rather_than_compared(tmp_path):
    """`"2" in {2, 5}` is False, so a string would quietly reclassify an
    agent-broken suite as a harness failure. That substitution has already
    shipped once, in `tools.from_report`."""
    with pytest.raises(IncompatibleRecord, match="verdict_exit_code"):
        load(write(tmp_path, verdict_exit_code="2"))


def test_every_outcome_the_scorer_can_produce_is_a_legal_outcome():
    """The two modules must not drift: a label scorer.final_outcome emits
    and record.load rejects would make a run unloadable the moment it was
    written."""
    from agentcheck.record import OUTCOMES
    from agentcheck.scorer import Score, SuiteState, final_outcome

    green = SuiteState(ok=True, failed_ids=frozenset())
    failing = SuiteState(ok=True, failed_ids=frozenset({"t::a"}))
    void = Score(status="unscoreable", unscoreable_reason="no verdict")
    dirty = Score(findings=[Finding("test_deleted", "cheat", "t.py", "x")])

    broke = Score(status="unscoreable", unscoreable_reason="no verdict",
                  unscoreable_cause="agent")

    produced = {
        final_outcome(green, Score(), None, False, False),
        final_outcome(green, dirty, None, False, False),
        final_outcome(green, void, None, False, False),
        final_outcome(green, broke, None, False, False),
        final_outcome(failing, Score(), "gave_up", False, False),
        final_outcome(failing, Score(), None, True, False),
        final_outcome(failing, Score(), None, False, True),
        final_outcome(failing, Score(), None, False, False, True),
        final_outcome(failing, Score(), None, False, False),
        final_outcome(failing, Score(), None, False, False, harness_error="x"),
    }
    assert produced <= OUTCOMES, f"scorer emits {produced - OUTCOMES}"


def test_a_valid_record_with_real_data_still_loads(tmp_path):
    r = load(write(tmp_path, failed_ids=["t::a", "t::b"],
                   outcome="solved_with_cheats", tool_calls={"read_file": 3},
                   trajectory=[{"type": "AIMessage"}]))
    assert r.failed_ids == ["t::a", "t::b"]
    assert r.tool_calls == {"read_file": 3}


# --- the predicate both entry points share ---------------------------------

def test_a_completed_run_holds_a_measurement(tmp_path):
    assert holds_measurement(write(tmp_path))


def test_a_missing_file_holds_nothing(tmp_path):
    assert not holds_measurement(tmp_path / "never-ran.json")


def test_an_unloadable_record_holds_no_measurement(tmp_path):
    """A file that exists and cannot be scored is not a result.

    Resuming on existence alone would skip it forever, leaving a hole in
    the matrix that no amount of re-running fills.
    """
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"task_id": "t", "schema_version": 1}))
    assert not holds_measurement(path)


def test_a_harness_failure_holds_no_measurement(tmp_path):
    """The bug that cost 22 runs, then blocked their replacement.

    A provider refused service, so the agent never ran. The record exists
    and loads cleanly — but nothing was measured, and treating it as done
    counted an absence as a result.
    """
    path = write(tmp_path, harness_error="OpenAIRateLimitError: 429")
    assert not holds_measurement(path)


@pytest.mark.parametrize("outcome", ["no_verdict", "broke_suite", "gave_up",
                                     "iteration_limit", "cost_limit",
                                     "solved_with_cheats",
                                     "stopped_without_solving"])
def test_a_run_that_went_badly_is_still_a_measurement(tmp_path, outcome):
    """Every one of these is a finding, not a failure to measure.

    A run that broke test collection, gave up, ran out of budget or cheated
    its way green all produced results. Only an unreadable record or a
    harness failure disqualifies — anything wider would re-run and
    overwrite exactly the behaviour this project exists to observe.
    """
    assert holds_measurement(write(tmp_path, outcome=outcome))


def test_both_entry_points_ask_the_same_question():
    """The sweep and run_agent must not be able to disagree.

    They did: the sweep asked the loader, run_agent asked only whether the
    file existed. So the sweep correctly scheduled 22 unusable records and
    run_agent refused every one with "already exists", which the sweep read
    as consecutive failures and stopped on.
    """
    from scripts.run_agent import holds_measurement as from_run_agent
    from scripts.sweep import holds_measurement as from_sweep
    assert from_sweep is holds_measurement
    assert from_run_agent is holds_measurement


def test_the_sweeps_own_ledger_is_not_read_as_a_run(tmp_path):
    """A glob decided what counted as evidence, and got it wrong.

    `runs/.sweep-ledger.json` is the sweep's bookkeeping. `glob("*.json")`
    returned it, the loader rejected it, and RESULTS.md reported "1
    record(s) in the archive do not load under the current schema" —
    sending a reader to the archive manifest to find a run that was never
    a run. Anything else that fails to load is still a real rejection and
    still gets named.
    """
    from agentcheck.record import record_paths

    real = write(tmp_path, outcome="solved")
    (tmp_path / ".sweep-ledger.json").write_text('{"settled": {}}')
    (tmp_path / ".hidden-state.json").write_text("{}")

    found = record_paths(tmp_path)
    assert found == [real]


def test_a_genuinely_broken_record_is_still_reported(tmp_path):
    """Excluding dotfiles must not become excluding inconvenient files."""
    from agentcheck.record import record_paths

    write(tmp_path, outcome="solved")
    (tmp_path / "broken.json").write_text("{not json")
    _records, rejected = load_all(record_paths(tmp_path))
    assert len(rejected) == 1 and "broken.json" in rejected[0]
