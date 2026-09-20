"""Tests for the agent loop, using a scripted fake model.

No API calls, no cost, fully deterministic. The point is to prove the loop
behaves correctly — caps fire, give_up is honoured, tool errors do not crash
the run, a harness failure DOES stop it, and no verdict is derived here at
all.

Debug the loop here. Spend money only once it passes.

    pytest test_agent.py -v -s
"""

import pytest
from langchain_core.messages import AIMessage

from agentcheck.agent import Agent, serialize
from agentcheck.sandbox import Sandbox, SandboxError
from agentcheck.scorer import SuiteState, final_outcome, score_run
from agentcheck.tools import Tools

#: Everything in this module talks to a real container.
#: Run the fast layers alone with: pytest -m "not container"
pytestmark = pytest.mark.container

IMAGE = "agentcheck/002-databases"
ENV = {"TEST_DATABASE_URLS": "sqlite:///testsuite,sqlite+aiosqlite:///testsuite"}
TEST_CMD = "pytest tests/ --ignore=tests/test_connection_options.py"


class FakeModel:
    """Replays a scripted list of AIMessages.

    Mimics enough of a LangChain chat model for the graph: bind_tools()
    returns self, invoke() returns the next scripted message.
    """

    def __init__(self, script, model="claude-haiku-4-5-20251001", usage=True):
        self.script = list(script)
        self.model = model
        self.usage = usage
        self.calls = 0
        self.last_messages = None

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        msg = self.script.pop(0) if self.script else AIMessage(content="done")
        if self.usage:
            msg.usage_metadata = {"input_tokens": 1000, "output_tokens": 500}
        return msg


def ai(text="", calls=None):
    return AIMessage(
        content=text,
        tool_calls=[{"name": n, "args": a, "id": f"c{i}"}
                    for i, (n, a) in enumerate(calls or [])],
    )


@pytest.fixture(scope="module")
def _box():
    with Sandbox(IMAGE, env=ENV) as s:
        yield s


@pytest.fixture
def tools(_box):
    t = Tools(_box, TEST_CMD)
    yield t
    _box.reset()


def outcome_of(run, before, after=None, reference_diff=""):
    """What run_agent would label this run, scoring first."""
    after = after or SuiteState.from_result(run.verdict)
    score = score_run(run.diff, before, after, reference_diff=reference_diff)
    return final_outcome(after, score, run.stopped,
                         run.hit_iteration_cap, run.hit_cost_cap,
                         run.hit_time_cap, run.harness_error or None)


BEFORE = SuiteState(ok=True, failed_ids=frozenset({"tests/test_databases.py::x"}))


# --- basic flow -------------------------------------------------------------

def test_model_with_no_tool_calls_ends_the_run(tools):
    a = Agent(tools, model=FakeModel([ai("I have nothing to do.")]),
              max_iterations=50)
    r = a.run("some failures")
    assert r.iterations == 1
    assert r.stopped is None
    assert outcome_of(r, BEFORE) == "stopped_without_solving"


def test_tool_calls_are_executed(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("read_file", {"path": "setup.py"})]),
        ai("Now I understand."),
    ]))
    r = a.run("failures")
    assert r.iterations == 2
    assert r.tool_calls["read_file"] == 1


def test_writes_reach_the_repository(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("write_file", {"path": "marker.py", "content": "x = 1\n"})]),
        ai("Written."),
    ]))
    a.run("failures")
    assert "marker.py" in tools.box.exec(["ls"]).stdout


def test_a_created_file_appears_in_the_diff(tools):
    """`git diff` shows nothing for untracked files, so an agent that
    creates a conftest.py to skip the failing tests produced an empty diff
    and scored clean."""
    a = Agent(tools, model=FakeModel([
        ai(calls=[("write_file", {"path": "conftest.py",
                                  "content": "collect_ignore = ['tests']\n"})]),
        ai("done"),
    ]))
    r = a.run("failures")
    assert "conftest.py" in r.changed_files
    assert "collect_ignore" in r.diff


# --- limits -----------------------------------------------------------------

def test_iteration_cap_stops_a_looping_agent(tools):
    """The realistic failure mode: fix one thing, break another, forever."""
    script = [ai(calls=[("list_files", {"path": "."})]) for _ in range(50)]
    a = Agent(tools, model=FakeModel(script), max_iterations=4)
    r = a.run("failures")
    assert r.iterations == 4
    assert r.hit_iteration_cap
    assert outcome_of(r, BEFORE) == "iteration_limit"


def test_cost_cap_stops_the_run(tools):
    script = [ai(calls=[("list_files", {"path": "."})]) for _ in range(50)]
    a = Agent(tools, model=FakeModel(script),
              max_iterations=100, max_cost_usd=0.005)
    r = a.run("failures")
    assert r.cost_usd >= 0.005
    assert r.iterations < 100
    assert r.hit_cost_cap
    assert outcome_of(r, BEFORE) == "cost_limit"


def test_wall_clock_cap_stops_the_run(tools):
    """One iteration can contain many tool calls, each of which may run a
    300-second suite. A cap on iterations alone bounds nothing in time.

    The cap is a millisecond rather than zero: zero is refused up front,
    because limits are checked after the model node and a cap of zero
    would still spend one call — quietly doing something other than what
    it says.
    """
    script = [ai(calls=[("list_files", {"path": "."})]) for _ in range(50)]
    a = Agent(tools, model=FakeModel(script),
              max_iterations=100, max_wall_seconds=0.001)
    r = a.run("failures")
    assert r.iterations == 1
    assert r.hit_time_cap


@pytest.mark.parametrize("cap", [
    {"max_iterations": 0}, {"max_iterations": -1},
    {"max_cost_usd": 0.0}, {"max_wall_seconds": -1},
])
def test_a_non_positive_cap_is_refused_before_any_work(tools, cap):
    """A cap of zero does not mean "no work": limits are checked after the
    model node, so it still spends one API call against a real container.
    Refuse rather than silently doing something else."""
    with pytest.raises(ValueError):
        Agent(tools, model=FakeModel([]), **cap)


def test_the_caps_are_recorded_on_the_run(tools):
    """A stored run says `cost_limit` at $2.08 with nothing recording what
    the cap was, which makes the label unfalsifiable."""
    a = Agent(tools, model=FakeModel([ai("done")]),
              max_iterations=7, max_cost_usd=0.25)
    a.run("failures")
    assert (a.max_iterations, a.max_cost_usd) == (7, 0.25)


# --- cost accounting --------------------------------------------------------

def test_a_provider_reporting_no_usage_makes_cost_UNKNOWN_not_zero(tools):
    """Left as zero, the spend cap silently stops enforcing anything and
    the run burns to the iteration cap at full price reporting $0.00."""
    script = [ai(calls=[("list_files", {"path": "."})]) for _ in range(6)]
    a = Agent(tools, model=FakeModel(script, usage=False),
              max_iterations=3, max_cost_usd=0.0001)
    r = a.run("failures")
    assert not r.cost_known
    assert not r.hit_cost_cap          # an unknown cost cannot trip a cap
    assert r.hit_iteration_cap         # the caps still in force did stop it


def test_cost_is_accumulated(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("list_files", {"path": "."})]), ai("done")]))
    r = a.run("failures")
    assert r.cost_usd > 0 and r.cost_known


# --- give up ----------------------------------------------------------------

def test_give_up_ends_the_run_and_records_the_reason(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("give_up", {"reason": "The API change is beyond me."})]),
        ai("should never be reached"),
    ]))
    r = a.run("failures")
    assert r.stopped == "gave_up"
    assert "beyond me" in r.give_up_reason
    assert r.iterations == 1
    assert outcome_of(r, BEFORE) == "gave_up"


# --- robustness -------------------------------------------------------------

def test_unknown_tool_is_counted_not_fatal(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("grep", {"x": 1})]), ai("oh well")]))
    r = a.run("failures")
    assert r.hallucinated_tools == ["grep"]
    assert not r.harness_error


def test_tool_exception_is_reported_not_raised(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("read_file", {"path": "/does/not/exist"})]), ai("noted")]))
    r = a.run("failures")
    assert not r.harness_error


def test_a_dead_sandbox_ENDS_the_run_instead_of_being_fed_back(tools):
    """Feeding a container failure back as a string lets the model spend
    the rest of its budget reasoning against a dead sandbox and producing
    a confident, meaningless diff."""
    class Exploding:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise SandboxError("container is gone")
            return boom

    broken = Tools(Exploding(), TEST_CMD)
    a = Agent(broken, model=FakeModel([
        ai(calls=[("read_file", {"path": "setup.py"})]),
        ai(calls=[("read_file", {"path": "setup.py"})]),
        ai(calls=[("read_file", {"path": "setup.py"})]),
    ]), max_iterations=20)
    r = a.run("failures")
    assert r.harness_error
    assert r.iterations == 1
    assert r.verdict.is_error
    assert outcome_of(r, BEFORE) == "harness_error"


def test_a_crash_preserves_the_iterations_cost_and_trace(tools):
    """On an exception LangGraph hands back the state we passed IN, so a
    crashed run recorded 0 iterations, $0.00 and a two-message trajectory
    — destroying the trace at exactly the moment it was needed."""
    class Detonating(FakeModel):
        def invoke(self, messages):
            if self.calls >= 2:
                raise RuntimeError("provider exploded")
            return super().invoke(messages)

    a = Agent(tools, model=Detonating([
        ai(calls=[("list_files", {"path": "."})]),
        ai(calls=[("list_files", {"path": "."})]),
    ]), max_iterations=20)
    r = a.run("failures")
    assert r.harness_error and "provider exploded" in r.harness_error
    assert r.iterations >= 1
    assert r.cost_usd > 0
    assert len(r.trajectory) > 2
    assert any(m.get("tool_calls") for m in r.trajectory)


# --- the claim is never trusted ---------------------------------------------

def test_claiming_success_does_not_make_it_so(tools):
    """The core guarantee. A model that announces victory while the suite
    still fails must not be scored as solved — and the outcome must record
    that it stopped by choice, not by exhausting its budget."""
    a = Agent(tools, model=FakeModel([
        ai("I have fixed everything. All tests pass now."),
    ]), max_iterations=50)
    r = a.run("failures")
    assert r.verdict.failed > 0
    assert outcome_of(r, BEFORE) == "stopped_without_solving"


def test_destroying_the_codebase_is_not_success(tools):
    """Deleting the source makes pytest collect nothing: 0 passed,
    0 failed. That must never read as green — and it is the AGENT's
    result, not a missing measurement: the outcome names who caused it."""
    a = Agent(tools, model=FakeModel([
        ai(calls=[("write_file",
                   {"path": "databases/core.py", "content": "def x(:\n"})]),
        ai("Fixed!"),
    ]))
    r = a.run("failures")
    assert r.verdict.is_error
    assert outcome_of(r, BEFORE) == "broke_suite"


def test_the_loop_derives_no_verdict_of_its_own(tools):
    """AgentRun carries facts. Two stored runs were stamped `solved` while
    their own score said credible_progress 0.0, because the label was
    computed from the test result before the diff was looked at."""
    from dataclasses import fields
    names = {f.name for f in fields(type(Agent(
        tools, model=FakeModel([ai("x")])).run("f")))}
    assert "outcome" not in names
    assert "score" not in names


# --- trace ------------------------------------------------------------------

def test_diff_and_changed_files_are_recorded(tools):
    a = Agent(tools, model=FakeModel([
        ai(calls=[("write_file", {"path": "setup.py", "content": "# gone\n"})]),
        ai("done"),
    ]))
    r = a.run("failures")
    assert "setup.py" in r.changed_files
    assert "gone" in r.diff


def test_short_tool_arguments_are_stored_whole():
    """A flat 300-character cut truncated file paths and search patterns,
    making it impossible to answer which file the agent read at step 7."""
    path = "databases/backends/" + "very_long_name_" * 20 + ".py"
    entry = serialize([ai(calls=[("read_file", {"path": path})])])[0]
    assert entry["tool_calls"][0]["args"]["path"] == path


def test_a_large_write_keeps_a_hash_and_both_ends():
    """Storing every whole-file write makes traces unreadable, but a bare
    truncation loses "was the second write a revision of the first" — which
    a hash answers for free."""
    content = "line\n" * 5000
    entry = serialize([ai(calls=[("write_file",
                                  {"path": "x.py", "content": content})])])[0]
    arg = entry["tool_calls"][0]["args"]["content"]
    assert arg["truncated"] and arg["lines"] == 5001
    assert len(arg["sha256_12"]) == 12
    assert arg["head"].startswith("line")


def test_identical_writes_hash_identically():
    a = serialize([ai(calls=[("write_file", {"c": "z" * 9000})])])
    b = serialize([ai(calls=[("write_file", {"c": "z" * 9000})])])
    assert (a[0]["tool_calls"][0]["args"]["c"]["sha256_12"]
            == b[0]["tool_calls"][0]["args"]["c"]["sha256_12"])


# --- prompt caching ---------------------------------------------------------

def test_the_static_prefix_is_marked_for_caching(tools):
    """At 43 iterations — the longest stored run — the same opening tokens
    were billed 43 times, which was most of that run's $2.08."""
    fake = FakeModel([ai("done")])
    Agent(tools, model=fake, cache_prompt=True).run("failures")
    system, human = fake.last_messages[0], fake.last_messages[1]
    for m in (system, human):
        assert isinstance(m.content, list)
        assert m.content[0]["cache_control"] == {"type": "ephemeral"}


def test_caching_can_be_turned_off(tools):
    fake = FakeModel([ai("done")])
    Agent(tools, model=fake, cache_prompt=False).run("failures")
    assert isinstance(fake.last_messages[0].content, str)


def test_caching_does_not_change_what_the_model_sees(tools):
    """It changes what a call COSTS, not the token sequence — which is why
    it is deliberately not part of the configuration fingerprint."""
    on, off = FakeModel([ai("d")]), FakeModel([ai("d")])
    Agent(tools, model=on, cache_prompt=True).run("same failures")
    Agent(tools, model=off, cache_prompt=False).run("same failures")
    assert on.last_messages[0].content[0]["text"] == off.last_messages[0].content


# --- configuration identity -------------------------------------------------

def test_tool_signature_carries_descriptions(tools):
    """Descriptions are prompt text the model reads on every call."""
    sig = dict(Agent(tools, model=FakeModel([])).tool_signature())
    assert "package_version" in sig
    assert "NOT what is" in sig["package_version"]



# --- the budget-announcement arm -------------------------------------------

def test_the_agent_is_told_its_remaining_steps_when_asked():
    """The intervention, and the reason for it.

    Without this the model is never told how many steps it has: the count
    lives in the router and never reaches a message. It behaves
    accordingly — mean reads per run were 9.3, 23.3, 48.4 and 51.9 at caps
    of 5, 15, 40 and 100, and 51.3 at 150. Reading saturates around fifty
    calls whatever the cap is, so the cap truncates the strategy rather
    than changing it.
    """
    from agentcheck.agent import BUDGET_NOTICE
    notice = BUDGET_NOTICE.format(used=3, total=40, left=37)
    assert "3" in notice and "40" in notice and "37" in notice


def test_the_two_arms_are_different_configurations():
    """Announcing the budget changes behaviour, so it must not pool.

    Folding the arms together would make the intervention invisible in
    exactly the comparison it exists to inform.
    """
    from agentcheck.record import ConfigFingerprint
    base = dict(model_id="m", tool_signature=(("read_file", "R"),),
                system_prompt="p", max_iterations=40, max_cost_usd=1.0,
                max_wall_seconds=1800.0, test_command="pytest",
                image_id="sha256:a", max_tokens=8192, max_file_lines=400,
                max_search_hits=40)
    assert (ConfigFingerprint(**base).digest()
            != ConfigFingerprint(**base, announce_budget=True).digest())


def test_announcing_is_off_by_default():
    """The control arm must stay the default, or every historical
    comparison silently changes meaning."""
    from agentcheck.record import ConfigFingerprint
    assert ConfigFingerprint(
        model_id="m", tool_signature=(), system_prompt="p", max_iterations=1,
        max_cost_usd=1.0, max_wall_seconds=1.0, test_command="t",
        image_id="i", max_tokens=1, max_file_lines=1,
        max_search_hits=1).announce_budget is False


def test_an_empty_reply_is_a_harness_error_not_a_decision(tools):
    """The router cannot tell "returned nothing" from "chose to stop".

    Both are an AIMessage with no tool calls, and `stopped_without_solving`
    is one of the findings this project reports — 11 of 64 runs in the
    ladder. A provider returning nothing, counted as a considered stop,
    inflates exactly the number being reported.

    Two runs in the budget-announcement arm ended at iteration 1 having
    read nothing and written nothing, recorded as decisions.
    """
    a = Agent(tools, model=FakeModel([ai("")]), max_iterations=50)
    r = a.run("some failures")
    assert r.harness_error, "an empty reply must not read as a decision"
    assert "empty reply" in r.harness_error
    assert outcome_of(r, BEFORE) == "harness_error"


def test_a_considered_stop_is_still_a_considered_stop(tools):
    """Prose with no tool calls IS the agent deciding it is finished.

    The guard above must not swallow the behaviour it sits next to — that
    behaviour is one of the results.
    """
    a = Agent(tools, model=FakeModel([ai("I cannot work this out.")]),
              max_iterations=50)
    r = a.run("some failures")
    assert not r.harness_error
    assert outcome_of(r, BEFORE) == "stopped_without_solving"


def test_the_budget_notice_does_not_add_a_turn():
    """Appending a HumanMessage put two human turns back to back, and the
    model answered with nothing — destroying the runs the intervention
    existed to measure. The text folds into the last message instead."""
    from langchain_core.messages import HumanMessage
    original = [HumanMessage(content="failures here")]
    amended = Agent._append_notice(original, "Budget: step 1 of 40.")
    assert len(amended) == len(original), "no extra turn"
    assert "Budget: step 1 of 40." in amended[-1].content
    assert original[-1].content == "failures here", "must not mutate state"
