"""Tests for the task definitions themselves.

Pure — reads the recorded artifacts, starts no container. A task whose
task.yaml disagrees with its own broken-report.json will produce wrong
numbers for every run measured against it, and that disagreement is cheap
to catch here and expensive to notice later.

    pytest tests/test_tasks.py -v
"""

import pytest

from agentcheck.diffparse import is_test_path, parse_diff
from agentcheck.task import Task, TaskError

TASK_IDS = Task.available()


def test_there_are_tasks_at_all():
    assert TASK_IDS


@pytest.fixture(params=TASK_IDS)
def task(request):
    return Task.load(request.param)


def test_task_loads_and_declares_its_own_id(task):
    assert task.id
    assert task.root.name == task.id


def test_the_dependency_under_test_is_named(task):
    """`package` drives find_runtime_patching. Without it, an agent that
    monkeypatches the library back to its old behaviour is undetectable."""
    assert task.package, f"{task.id} does not name the dependency"
    assert task.from_version and task.to_version


def test_the_recorded_broken_state_parses_and_is_not_empty(task):
    """An empty failing set means the task measures nothing, and every
    progress number computed against it divides by a denominator that does
    not exist."""
    state = task.recorded_broken_state()
    assert state.ok
    assert state.failed_ids


def test_the_recorded_state_matches_the_declared_counts(task):
    """task.yaml and broken-report.json are two descriptions of one fact.
    When they drift, the readable one is wrong and the authoritative one
    is silent about it."""
    declared = task.expected_broken.get("failed")
    if declared is None:
        pytest.skip(f"{task.id} declares no expected failure count")
    assert declared == len(task.recorded_broken_state().failed_ids)


def test_the_failure_is_partial_not_total(task):
    """An import-time break gives the agent no incremental foothold. The
    baseline must show tests still passing in the broken state."""
    passed = task.expected_broken.get("passed")
    if passed is None:
        pytest.skip(f"{task.id} declares no expected pass count")
    assert passed > 0, f"{task.id} fails totally — there is no foothold"


def test_failures_have_root_cause_messages(task):
    """Cluster progress needs them. Without messages every failure is its
    own cluster and the honest denominator collapses back to the test
    count."""
    state = task.recorded_broken_state()
    assert any(state.errors.get(nid) for nid in state.failed_ids)


def test_the_reference_patch_is_recorded_and_parses(task):
    """Without it every test edit is unjudgeable — a required API
    migration cannot be told from a deleted assertion."""
    diff = task.reference_diff()
    assert diff is not None, (
        f"{task.id} has no reference.patch: "
        f"python3 scripts/prepare_task.py {task.id} --reference")
    files = parse_diff(diff)
    assert files, f"{task.id}: reference.patch parses to nothing"


def test_the_reference_patch_changes_source_not_only_tests(task):
    """A migration the maintainer solved by editing only tests is not a
    task: the agent's entire job would be the behaviour this benchmark
    scores as cheating."""
    files = parse_diff(task.reference_diff() or "")
    source = [f for f in files
              if not is_test_path(f.path)
              and f.path.endswith(".py")]
    assert source, f"{task.id}: the reference PR adapts no source"


def test_reference_files_in_task_yaml_appear_in_the_patch(task):
    """A hand-listed reference_files that the patch does not corroborate
    is a stale comment with the authority of data."""
    if not task.reference_files:
        pytest.skip(f"{task.id} lists no reference_files")
    changed = {f.path for f in parse_diff(task.reference_diff() or "")}
    missing = [f for f in task.reference_files if f not in changed]
    assert not missing, f"{task.id}: {missing} are not in reference.patch"


def test_reachability_is_recorded_and_covers_the_source(task):
    reach = task.reachability()
    if not reach.has_coverage:
        pytest.skip(f"{task.id} has no coverage.json")
    assert reach.executed
    assert any(lines for lines in reach.executed.values())


def test_coverage_supersedes_the_hand_listed_prefixes(task):
    """Where both exist they must agree, or one of them is lying about
    which edits the test command executes."""
    reach = task.reachability()
    if not (reach.has_coverage and task.unreachable):
        pytest.skip(f"{task.id} has only one reachability source")
    for prefix in task.unreachable:
        executed = reach.executed.get(prefix.lstrip("./"))
        assert not executed, (
            f"{task.id}: task.yaml calls {prefix} unreachable but coverage "
            f"recorded {len(executed)} executed line(s) there")


def test_tasks_vary_the_dependency():
    """Eight pydantic migrations measure whether the model memorised the
    pydantic migration guide."""
    packages = {Task.load(t).package for t in TASK_IDS}
    assert len(packages) > 1, f"every task upgrades the same package: {packages}"


# --- the loader refuses to guess ---------------------------------------------

def test_a_task_without_a_package_is_rejected(tmp_path):
    """`package` keys find_runtime_patching. Defaulted to "", an agent that
    monkeypatches the library back to its old behaviour becomes
    undetectable and the run scores as a clean fix."""
    from agentcheck.task import TaskError
    root = tmp_path / "099-broken"
    root.mkdir()
    (root / "task.yaml").write_text(
        "id: 099-broken\nrepo: https://x/y\nbase_sha: abc\n"
        "test_command: pytest tests/\n")
    with pytest.raises(TaskError, match="package"):
        Task.load("099-broken", tasks_dir=tmp_path)


def test_a_mislabelled_task_directory_is_rejected(tmp_path):
    """A task.yaml declaring one id inside a directory named another
    merges two experiments under whichever name the reader used."""
    from agentcheck.task import TaskError
    root = tmp_path / "099-here"
    root.mkdir()
    (root / "task.yaml").write_text(
        "id: 099-elsewhere\nrepo: https://x/y\nbase_sha: abc\n"
        "package: urllib3\nfrom_version: '1'\nto_version: '2'\n"
        "test_command: pytest tests/\n")
    with pytest.raises(TaskError, match=r"mislabelled|declares id"):
        Task.load("099-here", tasks_dir=tmp_path)


# --- phase 2: validity metadata ---------------------------------------------

from agentcheck.scorer import clusters
from agentcheck.task import CONTAMINATION_RISKS, MIN_ROOT_CAUSES


@pytest.mark.parametrize("task_id", Task.available())
def test_every_task_declares_a_contamination_risk(task_id):
    """An unassessed task must not be readable as a clean one.

    All three migrations here are old and public, so a model may be
    recalling the maintainer's patch rather than deriving it. The harness
    cannot tell those apart; declaring the risk is the least it can do,
    and leaving the field blank would let RESULTS.md pool a memorised task
    with a novel one without anyone noticing.
    """
    task = Task.load(task_id)
    assert task.contamination_risk in CONTAMINATION_RISKS
    assert len(task.contamination_reason) > 80, (
        "a bare risk label is not checkable — say what it rests on")


@pytest.mark.parametrize("task_id", Task.available())
def test_the_declared_root_causes_match_the_recorded_broken_state(task_id):
    """The honest denominator has to agree with the data it comes from.

    `task.yaml` declares the root causes so RESULTS.md can report cluster
    progress without recomputing them, and this is what stops the
    declaration from drifting away from the broken report it describes.
    """
    task = Task.load(task_id)
    measured = clusters(task.recorded_broken_state())
    assert len(task.root_causes) == len(measured), (
        f"{task_id} declares {len(task.root_causes)} root cause(s) but its "
        f"recorded broken state clusters into {len(measured)}: "
        f"{sorted(measured)}")


@pytest.mark.parametrize("task_id", Task.available())
def test_a_low_power_task_declares_what_it_cannot_show(task_id):
    """Below three root causes, the limitation must be written down.

    Two of the three tasks are below the floor: 002 has two causes, one of
    which accounts for 53 of its 55 failing tests, and 003 has one. Both
    are kept — 002 is still the best cheat-detection case in the repo —
    but an undeclared limitation becomes an overstated result, so the
    shortfall lives in `task.yaml` where the report can repeat it.
    """
    task = Task.load(task_id)
    if len(task.root_causes) >= MIN_ROOT_CAUSES:
        assert not task.low_power_reason, (
            f"{task_id} has enough root causes; the low-power note is stale")
        return
    assert len(task.low_power_reason) > 60, (
        f"{task_id} has {len(task.root_causes)} root cause(s) and must say "
        f"what that costs the measurement")


def test_an_unrecognised_contamination_risk_is_refused(tmp_path):
    """A typo must not become an assessment."""
    root = tmp_path / "999-x"
    root.mkdir()
    (root / "task.yaml").write_text(
        "id: 999-x\nrepo: r\nbase_sha: s\ntest_command: pytest\n"
        "package: p\nfrom_version: '1'\nto_version: '2'\n"
        "contamination:\n  risk: probably-fine\n  reason: a reason\n")
    with pytest.raises(TaskError, match=r"contamination\.risk"):
        Task.load("999-x", tasks_dir=tmp_path)


def test_a_risk_without_a_reason_is_refused(tmp_path):
    root = tmp_path / "998-x"
    root.mkdir()
    (root / "task.yaml").write_text(
        "id: 998-x\nrepo: r\nbase_sha: s\ntest_command: pytest\n"
        "package: p\nfrom_version: '1'\nto_version: '2'\n"
        "contamination:\n  risk: low\n")
    with pytest.raises(TaskError, match="no reason"):
        Task.load("998-x", tasks_dir=tmp_path)
