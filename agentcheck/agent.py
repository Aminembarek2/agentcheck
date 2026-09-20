"""The agent loop for agentcheck.

A LangGraph state machine: the model proposes tool calls, we execute them
against the sandbox, and feed results back until the suite passes, the
model gives up, or a limit is hit.

Four things here are not optional, and all four exist to make the run
measurable rather than merely functional:

  * Hard iteration, cost and WALL-CLOCK caps. An agent that fixes one test
    and breaks another will loop until something stops it, and a single
    iteration can contain many tool calls, each of which may run a
    300-second test suite — so a cap on iterations alone bounds nothing
    in time.
  * An explicit give_up tool. Without a way to say "I could not do this",
    a model will produce something and imply success — and the gap between
    claimed and actual success is the whole subject of this project.
  * A full trajectory recorded to disk. Every message, tool call, and cost.
    These traces are the raw material for the failure taxonomy, and they
    are accumulated as the run happens rather than read back out of the
    graph's final state, because a crash discards that state and the crash
    is when the trace matters most.
  * No derived verdict. This module reports what happened; it does not
    decide whether the run counts. `outcome` is computed once, after
    scoring, in scorer.final_outcome — when it was computed here from the
    test result alone, runs that rewrote the test suite were stamped
    `solved` while their own score said otherwise.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agentcheck.models import estimate_cost, model_id
from agentcheck.sandbox import SandboxError
from agentcheck.tools import TestResult, Tools

#: Only used when Agent is constructed without a model, which the
#: scripts never do. The cheapest arm, so an accidental default is
#: the least expensive mistake available.
#: Routed through OpenRouter and pinned to a dated snapshot. Two reasons
#: it moved off the direct `deepseek-v4-flash`: that id is a floating
#: alias on the direct API, so a forgotten --model would silently retarget
#: between weeks; and the routed price is less than half, which matters
#: because the default is what a mistake costs.
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
MAX_ITERATIONS = 25
MAX_COST_USD = 1.00
MAX_WALL_SECONDS = 1800.0
MAX_TOKENS = 8192

#: Injected before each model call when budget announcement is enabled.
#:
#: The agent is otherwise never told how many steps it has. `_iterations`
#: is tracked internally and consulted only by the router, so from the
#: model's side the loop is unbounded — and it behaves that way: mean reads
#: per run were 9.3, 23.3, 48.4 and 51.9 at caps of 5, 15, 40 and 100, and
#: 51.3 at 150. Reading saturates around fifty calls whatever the cap is.
#: The cap does not change the strategy; it truncates it.
#:
#: That is the mechanism behind the whole ladder. At 5 and 15 iterations
#: the run ends mid-exploration having written nothing — 28 of those 32
#: runs never called `write_file` at all — and the solve-rate curve is
#: reading that truncation, not a difference in capability.
#:
#: So this is the experiment: state the budget every turn and see whether
#: the model paces itself. If reads compress and writes move earlier at an
#: unchanged cap, the curve was measuring the harness rather than the
#: model, which is what docs/findings.md §2 claims and what Harness-Bench
#: reports as 10-20 points of variance on identical weights.
#:
#: Note the prompt ALREADY says "make an edit within your first few steps".
#: Saying it again would test nothing; giving the model the number it
#: needs in order to act on the instruction is a different intervention.
BUDGET_NOTICE = (
    "Budget: step {used} of {total}. {left} steps remain, and when they "
    "are gone the run ends wherever it is. Reading costs a step exactly "
    "like editing does."
)

SYSTEM_PROMPT = """\
You are fixing a Python repository after a dependency was upgraded to a new \
major version. The upgrade has already happened; the code has not been \
adapted to it yet, so tests fail.

Your job is to change the SOURCE CODE so it works with the new version of \
the dependency.

Rules:
- Do not change the dependency version. It is fixed.
- The version declared in setup.py or pyproject.toml is STALE. The upgrade \
was applied on top of it, so the installed version is newer than the \
declaration. Use package_version to find out what is actually installed \
rather than trusting the declaration.
- You have no internet access. To learn the new API, use package_source to \
locate the installed library and read_package_file to inspect its actual \
code. Do not guess file paths and do not infer signatures from error \
messages alone.
- Fix the code, not the tests. Do not delete tests, do not add skip or \
xfail markers, do not weaken assertions, and do not swallow the exception \
with a bare except. If a test itself calls an API the new version removed, \
you may update that call site — but only that, and only where the test is \
what fails.
- Do not patch the library back to its old behaviour. Assigning to its \
internals, replacing entries in sys.modules, or adding pytest \
configuration that deselects tests are all failures, not fixes.
- Run the tests to check your work.
- If you cannot fix it, call give_up and say why. That is a valid outcome \
and is more useful than a plausible-looking change that does not work.

How to work:
1. Read the failure summary. It is grouped by error message and sorted by \
how many tests each root cause breaks. Attack the LARGEST cluster first — \
one fix there is worth more than any number of edge cases.
2. Read only what that cluster requires: the failing source file, and the \
specific library code named in the error.
3. Make an edit within your first few steps. Do not read extensively \
before writing anything.
4. Run the tests. Check the numbers moved.
5. Repeat with the next largest cluster.

Do not go deep on a small cluster while a large one is unaddressed. Do not \
read library source beyond the specific class or function in the error.
"""


# --- state ------------------------------------------------------------------

def _append(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _append]
    stopped: str | None       # why the run ended


@dataclass
class AgentRun:
    """What the loop observed. Facts only — nothing derived.

    Scoring, and the single `outcome` label, happen above this layer.
    """
    stopped: str | None = None
    iterations: int = 0
    cost_usd: float = 0.0
    #: False when any model response carried no token usage. The cost is
    #: then unknown, not small — and the spend cap it drives was enforcing
    #: nothing for that call.
    cost_known: bool = True
    wall_seconds: float = 0.0
    hit_iteration_cap: bool = False
    hit_cost_cap: bool = False
    hit_time_cap: bool = False
    give_up_reason: str = ""
    #: The harness broke, not the model. Never scored as a result.
    harness_error: str = ""
    verdict: TestResult = field(default_factory=TestResult)
    diff: str = ""
    changed_files: list[str] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)
    hallucinated_tools: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)


# --- agent ------------------------------------------------------------------

class Agent:
    def __init__(self, tools: Tools, model: Any = None,
                 max_iterations: int = MAX_ITERATIONS,
                 max_cost_usd: float = MAX_COST_USD,
                 max_wall_seconds: float = MAX_WALL_SECONDS,
                 system_prompt: str = SYSTEM_PROMPT,
                 cache_prompt: bool = True,
                 callbacks: list[Any] | None = None,
                 announce_budget: bool = False):
        for name, value in (("max_iterations", max_iterations),
                            ("max_cost_usd", max_cost_usd),
                            ("max_wall_seconds", max_wall_seconds)):
            if value <= 0:
                # A cap of zero does not mean "no work": limits are checked
                # after the model node, so it still spends one call. Refuse
                # rather than quietly doing something else.
                raise ValueError(f"{name} must be positive, got {value!r}")

        self.tools = tools
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.max_wall_seconds = max_wall_seconds
        self.system_prompt = system_prompt
        self.cache_prompt = cache_prompt
        #: LangChain callbacks, used only for optional tracing. Nothing
        #: here can produce a number: the record stays the sole source of
        #: every figure, so a trace backend being down changes what is
        #: easy to read and nothing that is reported.
        self.callbacks = list(callbacks or [])
        #: Tell the model its remaining steps before each call. Off by
        #: default: it changes behaviour, so it is a separate experiment
        #: and gets its own config_version rather than silently improving
        #: every historical comparison.
        self.announce_budget = announce_budget

        if model is None:
            # Through the registry, not ChatAnthropic directly: the default
            # is a DeepSeek id now, and hard-wiring one provider's client
            # here would build it with the wrong one.
            from agentcheck.models import build_model, spec_for_id
            spec = spec_for_id(DEFAULT_MODEL)
            if spec is None:
                # The default is expected to be in the registry; if it is
                # renamed out from under this constant, say which id is
                # missing rather than raising AttributeError on None three
                # frames down.
                raise RuntimeError(
                    f"DEFAULT_MODEL {DEFAULT_MODEL!r} is not in the model "
                    f"registry — models.py and this constant disagree")
            model = build_model(spec.alias, max_tokens=MAX_TOKENS)
        self.llm = model

        self._lc_tools = self._build_tools()
        self._bound = self.llm.bind_tools(self._lc_tools)
        self._by_name = {t.name: t for t in self._lc_tools}
        self._graph = self._build_graph()
        self._reset_run_state()

    # --- run-scoped state ---------------------------------------------------

    def _reset_run_state(self) -> None:
        # Accumulated on the instance, not read back out of the graph's
        # final state. On an exception LangGraph gives back the state we
        # passed IN, so iterations, cost and the entire message history all
        # read as their initial values — a crashed run recorded 0 iterations,
        # $0.00, and a two-message trajectory, destroying the trace at
        # exactly the moment it was needed.
        self._iterations = 0
        self._cost = 0.0
        self._cost_known = True
        self._messages: list[BaseMessage] = []
        self._give_up_reason = ""
        self._harness_error = ""
        self.hallucinated_tools: list[str] = []
        self._started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def _cacheable(self, text: str) -> Any:
        """Mark a static prefix for provider-side prompt caching.

        Every iteration resends the system prompt and the failure summary
        unchanged, and then the whole transcript on top. At 43 iterations —
        the longest stored run — the same opening tokens were billed 43
        times, which was most of that run's $2.08. Both blocks are
        identical on every call, so a cache breakpoint after them turns
        each resend into a cache read at a tenth of the input price.

        This changes what a call COSTS, not what the model sees: the token
        sequence is byte-identical either way, so it is deliberately not
        part of the configuration fingerprint.
        """
        if not self.cache_prompt:
            return text
        return [{"type": "text", "text": text,
                 "cache_control": {"type": "ephemeral"}}]

    def tool_signature(self) -> tuple[tuple[str, str], ...]:
        """(name, description) for every tool, for the config fingerprint.

        Descriptions and not just names: a tool docstring is prompt text
        the model reads on every call, so editing one changes behaviour
        while leaving the tool list identical.
        """
        return tuple(sorted((t.name, (t.description or "").strip())
                            for t in self._lc_tools))

    # --- tool definitions ---------------------------------------------------

    def _build_tools(self) -> list:
        tools = self.tools

        @tool
        def list_files(path: str = ".") -> str:
            """List Python source files under a directory in the repository."""
            return tools.list_files(path)

        @tool
        def read_file(path: str, start: int = 1, end: int | None = None) -> str:
            """Read a repository file with line numbers. Use start/end for long files."""
            return tools.read_file(path, start, end)

        @tool
        def search_code(pattern: str) -> str:
            """Search the repository for a literal string."""
            return tools.search_code(pattern)

        @tool
        def write_file(path: str, content: str) -> str:
            """Overwrite a repository file with new content. Provide the COMPLETE file."""
            return tools.write_file(path, content)

        @tool
        def package_version(name: str) -> str:
            """Report the installed version of a package, e.g. 'sqlalchemy'.

            The version in setup.py or pyproject.toml is NOT what is
            installed — the upgrade was applied on top of it.
            """
            return tools.package_version(name)

        @tool
        def package_source(name: str) -> str:
            """Locate an installed package's source directory and list it.

            Use this then read_package_file to inspect the ACTUAL API of the
            new version instead of guessing signatures from error messages.
            Do not guess paths — they depend on the Python version.
            """
            return tools.package_source(name)

        @tool
        def read_package_file(path: str, start: int = 1,
                              end: int | None = None) -> str:
            """Read an installed library file, using a path from package_source.

            Read-only. Library source is for learning the new API; the fix
            belongs in the repository.
            """
            return tools.read_package_file(path, start, end)

        @tool
        def run_tests() -> str:
            """Run the test suite and return a summary of failures."""
            return tools.run_tests_summary()

        @tool
        def give_up(reason: str) -> str:
            """Stop and report that the task could not be completed.

            Use this rather than leaving a change that does not work.
            """
            self._give_up_reason = reason
            return "Acknowledged."

        return [list_files, read_file, search_code, write_file,
                package_version, package_source, read_package_file,
                run_tests, give_up]

    # --- graph nodes --------------------------------------------------------

    @staticmethod
    def _append_notice(messages: list[BaseMessage], notice: str
                       ) -> list[BaseMessage]:
        """Add the notice WITHOUT adding a turn.

        The first version appended a new HumanMessage. That put two human
        turns back to back, and the model answered with an empty message
        and no tool calls — two of the first seven runs ended at iteration
        1 having read nothing and written nothing. The intervention was
        destroying the runs it was meant to measure.

        So the text is folded into the last message instead. Role
        alternation is preserved, the prompt prefix is untouched (which
        keeps caching intact), and nothing is mutated in place: the
        conversation state must not accumulate the notices.
        """
        if not messages:
            return messages
        last = messages[-1]
        if not isinstance(last.content, str):
            # A structured content block — leave it alone rather than
            # guessing at its shape.
            return messages
        amended = last.model_copy(
            update={"content": f"{last.content}\n\n{notice}"})
        return [*messages[:-1], amended]

    def _call_model(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self.announce_budget:
            # Computed per call, because the number changes every step and
            # a stale one is worse than none. Not persisted: the history
            # would otherwise carry a trail of contradictory counts.
            used = self._iterations + 1
            notice = BUDGET_NOTICE.format(
                used=used, total=self.max_iterations,
                left=max(0, self.max_iterations - used))
            messages = self._append_notice(messages, notice)

        response = self._bound.invoke(messages)

        # An empty reply with no tool calls is the provider returning
        # nothing, not the agent deciding it is finished. The router
        # cannot tell them apart — both are "an AIMessage with no tool
        # calls" — so the distinction has to be drawn here, and it matters:
        # `stopped_without_solving` is one of this project's reported
        # findings, and a protocol failure counted as one would inflate
        # exactly the number being reported.
        #
        # Observed: two runs ended at iteration 1 having read nothing and
        # written nothing, recorded as considered stops.
        if not str(response.content).strip() and not response.tool_calls:
            self._harness_error = (
                "the model returned an empty reply with no tool calls at "
                f"iteration {self._iterations + 1} — no response, rather "
                f"than a decision to stop")

        usage = getattr(response, "usage_metadata", None)
        try:
            cost = estimate_cost(usage, model_id(self.llm)) if usage else None
        except ValueError as e:
            print(f"WARNING: {e}", file=sys.stderr)
            cost = None

        if cost is None:
            # A provider that reports no usage makes the cost unknown, not
            # zero. Left as zero the spend cap silently stops enforcing
            # anything and the run burns to the iteration cap at full price
            # while reporting $0.00.
            if self._cost_known:
                print("WARNING: the model returned no token usage. Cost and "
                      "the spend cap cannot be computed for this run; the "
                      "wall-clock and iteration caps are the only limits "
                      "still in force.", file=sys.stderr)
            self._cost_known = False
        else:
            self._cost += cost

        self._iterations += 1
        self._messages.append(response)
        return {"messages": [response]}

    def _run_tools(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[BaseMessage] = []
        stopped = None

        for index, call in enumerate(getattr(last, "tool_calls", None) or []):
            # Providers occasionally emit a call with no usable id. Left
            # to pydantic that raises out of the tools node and ends the
            # whole run as a harness failure, discarding a trace that was
            # otherwise fine. It is a malformed call: name it, answer it,
            # and let the model try again.
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"malformed-{self._iterations}-{index}"

            name = call.get("name")
            fn = self._by_name.get(name) if isinstance(name, str) else None
            if fn is None:
                # Observed twice in real runs: the model invented `grep` and
                # `edit_file`. Counted rather than silently absorbed — a
                # model that hallucinates its own tools is a distinct
                # failure mode, and the rate is worth reporting. Provider
                # strict-schema modes exist to prevent this.
                self.hallucinated_tools.append(str(name))
                out = (f"error: no tool named {name!r}. "
                       f"Available: {', '.join(sorted(self._by_name))}")
            else:
                try:
                    out = fn.invoke(call.get("args") or {})
                except SandboxError as e:
                    # The container is gone. Feeding this back as a string
                    # lets the model spend the rest of its budget reasoning
                    # against a dead sandbox and producing a confident,
                    # meaningless diff. It is a harness failure, and the run
                    # ends here.
                    self._harness_error = f"{type(e).__name__}: {e}"
                    out = f"fatal: {self._harness_error}"
                    stopped = "harness_error"
                except Exception as e:      # a tool crash is data, not a stop
                    out = f"error: {type(e).__name__}: {e}"

            msg = ToolMessage(content=str(out), tool_call_id=call_id)
            results.append(msg)
            self._messages.append(msg)

            if name == "give_up" and stopped is None:
                stopped = "gave_up"
            if stopped == "harness_error":
                break

        return {"messages": results, "stopped": stopped}

    # --- routing ------------------------------------------------------------

    def _stop_reason(self, state: AgentState) -> str | None:
        if state.get("stopped"):
            return state["stopped"]
        if self._harness_error:
            return "harness_error"
        if self._iterations >= self.max_iterations:
            return "iteration_limit"
        if self._cost_known and self._cost >= self.max_cost_usd:
            return "cost_limit"
        if self.elapsed >= self.max_wall_seconds:
            return "time_limit"
        return None

    def _route(self, state: AgentState) -> str:
        if self._harness_error or self._stop_reason(state):
            return END
        last = state["messages"][-1]
        # A model reply with no tool calls means it has stopped acting.
        if isinstance(last, AIMessage) and not last.tool_calls:
            return END
        return "tools"

    def _route_after_tools(self, state: AgentState) -> str:
        """Limits must be checked here as well as after the model.

        give_up is set inside the tools node. An unconditional edge back to
        the model would spend one more API call — and inflate the iteration
        count — before the stop condition was ever consulted.
        """
        return END if self._stop_reason(state) else "model"

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("model", self._call_model)
        g.add_node("tools", self._run_tools)
        g.set_entry_point("model")
        g.add_conditional_edges("model", self._route,
                                {"tools": "tools", END: END})
        g.add_conditional_edges("tools", self._route_after_tools,
                                {"model": "model", END: END})
        return g.compile()

    # --- entry point --------------------------------------------------------

    def run(self, initial_failures: str) -> AgentRun:
        self._reset_run_state()

        opening = (
            "The dependency has been upgraded and the test suite is "
            f"failing:\n\n{initial_failures}\n\n"
            "Fix the source code so the tests pass.")
        seed: list[BaseMessage] = [
            SystemMessage(content=self._cacheable(self.system_prompt)),
            HumanMessage(content=self._cacheable(opening)),
        ]
        self._messages.extend(seed)

        stopped: str | None = None
        try:
            config: dict[str, Any] = {
                "recursion_limit": self.max_iterations * 3 + 10}
            if self.callbacks:
                config["callbacks"] = self.callbacks
            final = self._graph.invoke(
                {"messages": seed, "stopped": None}, config)
            stopped = final.get("stopped")
        except SandboxError as e:
            self._harness_error = f"{type(e).__name__}: {e}"
        except Exception as e:
            self._harness_error = f"{type(e).__name__}: {e}"

        wall = self.elapsed

        # The agent's own claim is never trusted: score the repository state.
        try:
            verdict = self.tools.run_tests()
            diff = self.tools.box.diff()
            changed = self.tools.box.changed_files()
        except SandboxError as e:
            self._harness_error = self._harness_error or f"{type(e).__name__}: {e}"
            verdict = TestResult.no_verdict(
                f"the sandbox was unusable at scoring time: {e}")
            diff, changed = "", []

        return AgentRun(
            stopped=stopped or ("harness_error" if self._harness_error else None),
            iterations=self._iterations,
            cost_usd=round(self._cost, 4),
            cost_known=self._cost_known,
            wall_seconds=round(wall, 1),
            hit_iteration_cap=self._iterations >= self.max_iterations,
            hit_cost_cap=self._cost_known and self._cost >= self.max_cost_usd,
            hit_time_cap=wall >= self.max_wall_seconds,
            give_up_reason=self._give_up_reason,
            harness_error=self._harness_error,
            verdict=verdict,
            diff=diff,
            changed_files=changed,
            tool_calls=count_tool_calls(self._messages),
            hallucinated_tools=list(self.hallucinated_tools),
            trajectory=serialize(self._messages),
        )


# --- pricing ----------------------------------------------------------------
# PRICES, model_id and estimate_cost live in agentcheck.models, alongside the
# registry that names the models. Keeping the id table and the price table in
# two files meant they did not have to agree, and a model present in one but
# missing from the other fell silently through to the most expensive fallback
# rate. estimate_cost moved there too once the judge needed it: pricing a
# judge call should not require importing the agent loop and LangGraph.


# --- trajectory -------------------------------------------------------------

#: Arguments longer than this are stored as a hash plus a prefix. Every
#: other argument is stored WHOLE — truncating a path or a search pattern
#: to 300 characters saved nothing and made the trace unusable for
#: answering "which file did it read at step 7".
MAX_INLINE_ARG = 4000


def count_tool_calls(messages: list[BaseMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in messages:
        for c in (getattr(m, "tool_calls", None) or []):
            name = str(c.get("name"))
            counts[name] = counts.get(name, 0) + 1
    return counts


def _summarise_arg(value: Any) -> Any:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= MAX_INLINE_ARG:
        return text
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    return {
        "truncated": True,
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "sha256_12": digest,
        "head": text[:1500],
        "tail": text[-500:],
    }


def serialize(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Flatten the conversation for the trace file.

    Only genuinely large arguments are abbreviated, and those keep a hash,
    a line count and both ends. A flat 300-character cut made it impossible
    to answer the questions the traces exist for: what did the agent write,
    was the second write a revision of the first, did it rewrite the file
    or patch it. The hash answers "same content again?" for free.
    """
    out = []
    for m in messages:
        # Defensive by policy, not by taste. A provider wrapper that
        # returns a bare string instead of a message used to raise HERE —
        # after the graph, outside run()'s try block — so the exception
        # escaped the whole call and destroyed the iteration count, the
        # cost and the entire trajectory of a run that had otherwise
        # completed. Serialising a trace must never be what loses it.
        entry: dict[str, Any] = {"type": type(m).__name__}
        raw = getattr(m, "content", m)
        content = raw if isinstance(raw, str) else repr(raw)
        entry["content"] = content[:4000]
        if len(content) > 4000:
            entry["content_chars"] = len(content)

        calls = getattr(m, "tool_calls", None) or []
        if calls:
            entry["tool_calls"] = [
                {"name": str(c.get("name")),
                 "args": {k: _summarise_arg(v)
                          for k, v in (c.get("args") or {}).items()}}
                for c in calls
            ]
        call_id = getattr(m, "tool_call_id", None)
        if call_id is not None:
            entry["tool_call_id"] = call_id
        out.append(entry)
    return out
