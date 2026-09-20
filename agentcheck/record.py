"""The run record: one schema, one loader, one writer.

Every bug this project has hit shares one root cause: a missing value
silently became a plausible number.

  * a stale report.json read as a completed run          -> 60p/55f, wrong
  * a missing failed_ids read as "nothing failing"       -> 100% progress
  * a missing price entry read as the most expensive     -> cap fired at 9 iters
  * an empty before-state read as "nothing was broken"   -> every run 0%
  * a pre-schema field read as present                   -> seven fake 100%s
  * a summary key omitted when zero read as "none failed" -> errors scored green

None was a logic error. Each was a `.get(key, default)` where the default
happened to be a legal value. The defence is not more detectors — it is a
schema that distinguishes ABSENT from ZERO, and refuses to guess.

There used to be two types here: the one the agent produced and the one the
readers validated. Two schemas for one object is how a field went missing in
the first place, so there is now exactly one, and every entry point goes
through `load`.

The second half of the file addresses the other failure: comparing runs
that are not comparable. Tools changed, prompts changed, the model id
changed. lm-eval treats versioned code as a precondition for comparable
results, and a 2026 audit of self-adapting agents traced one of its own
published null results to version drift, recommending that evaluator
identity and version always be reported. So config_version is derived
automatically from everything that affects behaviour, and aggregation
refuses to mix versions rather than trusting a label.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: Bumped when the saved record's SHAPE changes. Records written under a
#: different schema are readable but never scoreable — the fields they lack
#: cannot be reconstructed, and inventing them is how seven runs came to
#: claim 100%.
#:
#: v3: verdict status replaces the failed==-1 sentinel; the before-state and
#:     the caps live in the record; findings are structured.
SCHEMA_VERSION = 3

#: Fields without which a run cannot be scored at all. Absent means the
#: record is rejected, not defaulted.
REQUIRED = (
    "schema_version", "config_version", "task_id", "model", "outcome",
    "diff", "failed_ids", "before_failed_ids", "verdict_status",
    "iterations", "cost_usd", "cost_known",
    "max_iterations", "max_cost_usd",
)


#: Every label `scorer.final_outcome` can produce. A record carrying
#: anything else is not describing a run this harness performed — and an
#: unrecognised outcome matches no category in any summary, so it silently
#: vanishes from every count rather than being reported.
OUTCOMES = frozenset({
    "solved", "solved_with_cheats", "gave_up", "stopped_without_solving",
    "iteration_limit", "cost_limit", "time_limit", "no_verdict",
    #: The agent's own edits stopped the suite from collecting. Distinct
    #: from `no_verdict`, which is the harness failing to measure.
    "broke_suite",
    "harness_error",
})

VERDICT_STATUSES = frozenset({"ok", "no_verdict"})


class IncompatibleRecord(Exception):
    """The record cannot be scored, and no default would make it so."""


# --- configuration identity -------------------------------------------------

@dataclass(frozen=True)
class ConfigFingerprint:
    """Everything that changes what a run does.

    Each field is here because changing it changes the result while leaving
    every other recorded field identical, so two runs that differ only in
    it would silently merge into one column.

      model_id          the obvious one.
      tool_signature    names AND descriptions. The descriptions are prompt
                        text — `package_version`'s docstring argues with the
                        model about setup.py being stale — so editing one
                        changes behaviour with the tool list unchanged.
      system_prompt     v3 and v4 differed only here, and nothing but a
                        filename recorded it.
      max_iterations    a 25-iteration run and a 50-iteration run are not
                        samples of the same thing; several stored runs end
                        at their cap, so the cap IS the result.
      max_cost_usd      same argument. `haiku-02` reports `cost_limit` at
                        $2.08 with a default cap of $1.00 recorded nowhere,
                        which makes the label unfalsifiable.
      max_wall_seconds  a run killed by the clock is a different sample.
      test_command      it decides which tests exist. Deselecting a module
                        changes the denominator of every progress number.
      image_id          the resolved image digest, not the tag. This is the
                        only thing that pins the actual dependency versions,
                        and it can change with no change to any source file
                        — a rebuilt image is a different experiment.
      max_file_lines    how much of a file the model sees per call, which
      max_search_hits   changes its search strategy. Free to hash, and
                        cheap insurance against a silent behaviour change.
      max_tokens        currently differs between Agent's default and
                        run_agent's, so it is not a constant.
      provider_route    which upstream actually served the request. Empty
                        for a direct provider key, where there is only one
                        answer. Not empty when routed through an
                        aggregator: OpenRouter serves MiMo from seven
                        different hosts, and they are not the same weights
                        on the same hardware — GLM-5.2 spans a 3x price
                        range across its providers, which is not what
                        identical service looks like. Without this field a
                        silent route change would pool two backends into
                        one column, which is the `deepseek-chat` aliasing
                        bug wearing a different hat.

    Deliberately NOT included: the label, the output filename, wall-clock
    date, and anything else an operator types by hand. Derived, not
    declared — a mislabelled run must not be able to merge two
    configurations into one.
    """
    model_id: str
    tool_signature: tuple[tuple[str, str], ...]
    system_prompt: str
    max_iterations: int
    max_cost_usd: float
    max_wall_seconds: float
    test_command: str
    image_id: str
    max_tokens: int
    max_file_lines: int
    max_search_hits: int
    #: "" for a direct provider key. Defaulted so every existing record's
    #: digest is unchanged by this field's introduction — a stamp that
    #: shifted under the archive would orphan fifteen runs.
    provider_route: str = ""

    #: Whether the agent was told its remaining step budget each turn.
    #: It changes what the model does, so the two arms are different
    #: experiments and must not pool; defaulted False so no existing
    #: digest moves.
    announce_budget: bool = False

    def digest(self) -> str:
        payload = json.dumps({
            "model": self.model_id,
            "tools": sorted(self.tool_signature),
            "prompt": self.system_prompt,
            "max_iterations": self.max_iterations,
            "max_cost_usd": round(float(self.max_cost_usd), 6),
            "max_wall_seconds": round(float(self.max_wall_seconds), 3),
            "test_command": self.test_command,
            "image_id": self.image_id,
            "max_tokens": self.max_tokens,
            "max_file_lines": self.max_file_lines,
            "max_search_hits": self.max_search_hits,
            **({"provider_route": self.provider_route}
               if self.provider_route else {}),
            **({"announce_budget": True} if self.announce_budget else {}),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_signature"] = [list(t) for t in self.tool_signature]
        return d


def config_version(**kwargs: Any) -> str:
    """Convenience wrapper: build a fingerprint and return its digest."""
    return ConfigFingerprint(**kwargs).digest()


# --- the record -------------------------------------------------------------

@dataclass
class RunRecord:
    """One agent run, complete enough to re-score without any other file.

    `before_failed_ids` and `before_errors` are stored deliberately. When
    re-scoring had to reach outside the record for the before-state, one
    script passed an empty set and flattened every stored progress number
    to 0%, and another could not recover the state at all. A record that
    carries both ends of its own measurement cannot have that bug.
    """
    config_version: str
    task_id: str
    model: str
    outcome: str

    #: Verdict of the final test run. "ok" or "no_verdict" — never inferred
    #: from a count, because zero failures and no verdict are the same
    #: number and opposite conclusions.
    verdict_status: str = "no_verdict"
    verdict_reason: str = ""
    #: pytest's exit code for that run, when there was one. Stored because
    #: it is what tells an agent-broken suite from a broken container, and
    #: the alternative is parsing the reason prose back out of a sentence
    #: this harness wrote. None on records written before it was kept.
    verdict_exit_code: int | None = None

    failed_ids: list[str] = field(default_factory=list)
    before_failed_ids: list[str] = field(default_factory=list)
    before_errors: dict[str, str] = field(default_factory=dict)

    tests_passed: int = 0
    tests_failed: int = 0
    tests_errored: int = 0
    tests_skipped: int = 0
    tests_xfailed: int = 0
    tests_xpassed: int = 0

    iterations: int = 0
    cost_usd: float = 0.0
    #: False when the provider reported no token usage. The cost figure is
    #: then not small, it is unknown — and the spend cap it drives was not
    #: enforcing anything.
    cost_known: bool = True
    wall_seconds: float = 0.0

    max_iterations: int = 0
    max_cost_usd: float = 0.0
    max_wall_seconds: float = 0.0

    give_up_reason: str = ""
    #: Set when the harness itself failed (container died, docker gone).
    #: Distinct from the model failing, and never scored as a result.
    harness_error: str = ""

    diff: str = ""
    changed_files: list[str] = field(default_factory=list)

    #: how many times each tool was called. The explore-vs-edit split
    #: separates capability tiers: strong agents localise fast and shift
    #: to editing, weak ones keep reading and never write anything.
    tool_calls: dict[str, int] = field(default_factory=dict)
    #: tool names the model invented that do not exist
    hallucinated_tools: list[str] = field(default_factory=list)

    score: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    schema_version: int = SCHEMA_VERSION
    path: Path | None = None

    # --- io -----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the record.

        Refuses to save without a config_version. A record whose producing
        configuration is unknown cannot be compared to anything, and
        writing it anyway just defers the problem to whoever reads it.
        """
        if not self.config_version:
            raise ValueError(
                "refusing to save a run with no config_version — it could "
                "not be compared to any other run")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"refusing to save schema v{self.schema_version} from a "
                f"v{SCHEMA_VERSION} harness")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(self).items() if k != "path"}

        # Written to a sibling and renamed. A crash partway through a plain
        # write leaves a truncated file that is neither a record nor
        # absent — and the sweep, which decides what to re-run by asking
        # whether the file EXISTS, would skip that attempt forever. rename
        # within a directory is atomic, so the file is either the previous
        # content or the new one.
        tmp = path.with_name(path.name + ".partial")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
        os.replace(tmp, path)
        self.path = path
        return path

    @property
    def is_current(self) -> bool:
        return self.schema_version == SCHEMA_VERSION

    @property
    def verdict_ok(self) -> bool:
        return self.verdict_status == "ok"

    @property
    def name(self) -> str:
        return self.path.stem if self.path else self.task_id


_FIELD_NAMES = {f.name for f in fields(RunRecord)} - {"path"}



def _check(raw: dict[str, Any], name: str) -> None:
    """Validate types and ranges, not just presence.

    Presence alone is not enough, and the gap is not theoretical: a
    `failed_ids` of "a,b" instead of ["a", "b"] passes every presence
    check, then gets handed to frozenset() downstream and iterates as the
    three CHARACTERS 'a', ',', 'b'. Three failing tests, from a typo, with
    nothing raising. A `cost_usd` of "0.5" survives just as far and then
    concatenates in a sum.

    The rule of this module is that absent is never zero. The same
    argument applies to a value of the wrong shape: it is not data, and
    guessing what was meant is how a wrong number gets published.
    """
    def fail(message: str) -> None:
        raise IncompatibleRecord(f"{name}: {message}")

    def want_str(key: str, allow_empty: bool = False) -> str:
        value = raw[key]
        if not isinstance(value, str):
            fail(f"{key} is {type(value).__name__}, expected a string")
        if not allow_empty and not value.strip():
            fail(f"{key} is empty")
        return value

    def want_int(key: str, minimum: int = 0) -> None:
        value = raw.get(key, minimum)
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f"{key} is {value!r}, expected an integer")
        if value < minimum:
            fail(f"{key} is {value}, which is impossible")

    def want_number(key: str, minimum: float = 0.0) -> None:
        value = raw.get(key, minimum)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f"{key} is {value!r}, expected a number")
        if value < minimum:
            fail(f"{key} is {value}, which is impossible")

    def want_id_list(key: str) -> None:
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value):
            fail(f"{key} is {type(value).__name__}, expected a list of "
                 f"strings — a bare string here iterates as characters")
        if len(set(value)) != len(value):
            fail(f"{key} contains duplicate ids")

    if not isinstance(raw["schema_version"], int) or isinstance(
            raw["schema_version"], bool):
        fail(f"schema_version is {raw['schema_version']!r}, expected an int")

    for key in ("config_version", "task_id", "model"):
        want_str(key)

    outcome = want_str("outcome")
    if outcome not in OUTCOMES:
        fail(f"outcome {outcome!r} is not one of {sorted(OUTCOMES)}")

    status = want_str("verdict_status")
    if status not in VERDICT_STATUSES:
        fail(f"verdict_status {status!r} is not one of "
             f"{sorted(VERDICT_STATUSES)}")

    want_str("diff", allow_empty=True)
    want_id_list("failed_ids")
    want_id_list("before_failed_ids")
    want_id_list("changed_files")

    for key in ("iterations", "max_iterations", "tests_passed", "tests_failed",
                "tests_errored", "tests_skipped", "tests_xfailed",
                "tests_xpassed"):
        want_int(key)
    for key in ("cost_usd", "max_cost_usd", "wall_seconds", "max_wall_seconds"):
        want_number(key)

    if not isinstance(raw["cost_known"], bool):
        fail(f"cost_known is {raw['cost_known']!r}, expected a boolean")

    # Absent is fine — records predating the field have no exit code — but
    # a wrong-shaped one is not. `"2" in {2, 5}` is False, so a string here
    # would classify an agent-broken suite as a harness failure without
    # anything raising. That exact substitution has already happened once
    # in `tools.from_report`.
    exit_code = raw.get("verdict_exit_code")
    if exit_code is not None and (isinstance(exit_code, bool)
                                  or not isinstance(exit_code, int)):
        fail(f"verdict_exit_code is {exit_code!r}, expected an integer or "
             f"nothing")

    tool_calls = raw.get("tool_calls", {})
    if not isinstance(tool_calls, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in tool_calls.items()):
        fail("tool_calls must map tool names to integer counts")

    for key in ("score", "config", "before_errors"):
        if not isinstance(raw.get(key, {}), dict):
            fail(f"{key} is {type(raw[key]).__name__}, expected an object")
    if not isinstance(raw.get("trajectory", []), list):
        fail("trajectory must be a list")


def load(path: str | Path) -> RunRecord:
    """Read one record, or raise.

    Deliberately strict. A loader that tolerates missing fields is a
    loader that fabricates results. Every entry point in the harness goes
    through here — when `inspect_run` and `rescore` read raw JSON instead,
    they averaged unversioned records from four configurations into a
    single number and reported it.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise IncompatibleRecord(f"{path.name}: not valid JSON — {e}") from e
    if not isinstance(raw, dict):
        raise IncompatibleRecord(f"{path.name}: not a JSON object")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        missing = [k for k in REQUIRED if k not in raw]
        raise IncompatibleRecord(
            f"{path.name}: schema v{version if version is not None else '?'} "
            f"(current v{SCHEMA_VERSION}); missing "
            f"{missing or 'nothing, but the shape is not this one'}. "
            f"Re-run rather than re-score — the data is not there.")

    absent = [k for k in REQUIRED if k not in raw]
    if absent:
        raise IncompatibleRecord(f"{path.name}: missing required {absent}")
    if not raw.get("config_version"):
        raise IncompatibleRecord(
            f"{path.name}: no config_version — it cannot be compared to "
            f"anything, because what produced it is unknown.")

    unknown = set(raw) - _FIELD_NAMES
    if unknown:
        raise IncompatibleRecord(
            f"{path.name}: unknown field(s) {sorted(unknown)} — written by a "
            f"different harness version than this one claims to be.")

    _check(raw, path.name)

    record = RunRecord(**{k: v for k, v in raw.items()})
    record.path = path
    return record


def holds_measurement(path: str | Path) -> bool:
    """Does this path already hold a completed, usable run?

    THE predicate for "has this cell been done", and the single place it is
    decided. Two callers ask it: the sweep, deciding what to skip on a
    resume, and `run_agent`, deciding whether writing here would destroy a
    result.

    They used to answer it differently. The sweep asked the loader;
    `run_agent` asked only whether the file existed. So when the sweep
    correctly determined that 22 rate-limited records were not
    measurements and scheduled them to run again, `run_agent` refused every
    one of them with "already exists — pick another --label", and the sweep
    read that as four consecutive failures and stopped. Both were behaving
    sensibly; they simply did not share a definition.

    Three things are deliberately NOT disqualifying, because each is a
    genuine result rather than a failure to measure:

      * a run that ended without a test verdict — the agent broke
        collection, which is a finding;
      * a run that solved nothing, or cheated;
      * a run that hit any of its caps.

    Only two things disqualify: the record cannot be loaded at all, or the
    harness itself failed and the run therefore measured nothing.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        record = load(path)
    except IncompatibleRecord:
        return False
    return not record.harness_error


def record_paths(directory: str | Path = "runs") -> list[Path]:
    """The run records in a directory — and nothing else that ends .json.

    `Path.glob("*.json")` also returns `.sweep-ledger.json`, the sweep's
    own bookkeeping. The loader then rejected it, and RESULTS.md reported
    "1 record(s) in the archive do not load under the current schema",
    pointing the reader at `runs/archive/MANIFEST.md` for a run that does
    not exist. A false statement about the data, produced by a glob.

    Leading-dot files are the harness's own state, never a measurement.
    Anything else that fails to load is a real rejection and must keep
    being reported as one — the shrinking denominator is the whole reason
    rejections are named.
    """
    return sorted(p for p in Path(directory).glob("*.json")
                  if not p.name.startswith("."))


def load_all(paths: Iterable[str | Path]
             ) -> tuple[list[RunRecord], list[str]]:
    """Load what can be loaded; report what cannot, by name and reason.

    Returns (records, rejections). Rejections are never silently dropped —
    a shrinking denominator is itself a way to fabricate a result.
    """
    records, rejected = [], []
    for p in paths:
        try:
            records.append(load(p))
        except IncompatibleRecord as e:
            rejected.append(str(e))
    return records, rejected


def group_by_config(records: Iterable[RunRecord]
                    ) -> dict[tuple[str, str], list[RunRecord]]:
    """Partition by what actually produced each run.

    Keyed by (task_id, config_version). Aggregating across configs would
    average a model that had one tool set together with one that had
    another; aggregating across tasks would average two different
    denominators. Neither is a mean, both are mixtures, and that is the
    standard way a benchmark stops meaning anything.
    """
    out: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        out.setdefault((r.task_id, r.config_version), []).append(r)
    return out


def describe_config(records: Mapping[Any, Any] | list[RunRecord]) -> str:
    """A one-line label for a group, taken from the runs themselves."""
    group = list(records)
    if not group:
        return "(empty)"
    first = group[0]
    caps = f"{first.max_iterations} iters, ${first.max_cost_usd:.2f}"
    return f"{first.task_id} · {first.model} · {caps} · cfg {first.config_version}"
