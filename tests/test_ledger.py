"""The budget ledger, tested at the race it exists to close."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agentcheck.ledger import BudgetExhausted, Ledger


def test_two_workers_cannot_both_claim_the_last_of_the_budget():
    """The read-then-act race, in one assertion.

    Under the old check both workers read `spent`, both saw room, and both
    started. The overshoot did not raise — it appeared on the bill.
    """
    def run(tmp):
        ledger = Ledger(tmp / "led.json", budget=1.0)
        assert ledger.reserve("a", 0.6)[0]
        assert not ledger.reserve("b", 0.6)[0]
    import tempfile
    run(Path(tempfile.mkdtemp()))


def test_concurrent_reservations_never_exceed_the_budget(tmp_path):
    """Twenty threads, ten affordable slots, no overshoot."""
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    granted: list[str] = []
    lock = threading.Lock()

    def claim(i: int) -> None:
        if ledger.reserve(f"cell-{i}", 0.1)[0]:
            with lock:
                granted.append(f"cell-{i}")

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == 10
    assert ledger.snapshot().committed == pytest.approx(1.0)


def test_a_reservation_is_replaced_by_what_the_attempt_really_cost(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=10.0)
    ledger.reserve("a", 1.0)
    assert ledger.snapshot().reserved == pytest.approx(1.0)
    ledger.settle("a", 0.05)
    snap = ledger.snapshot()
    assert snap.reserved == 0
    assert snap.settled == pytest.approx(0.05)
    assert snap.open_reservations == 0


def test_an_overspend_is_recorded_not_clamped(tmp_path):
    """A cap that refuses to admit spend it already incurred is a lie.

    The reservation bounds what gets STARTED. Once a run has cost more
    than its reservation, the ledger's job is to say so, not to report the
    comfortable number.
    """
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    ledger.reserve("a", 0.1)
    ledger.settle("a", 0.9)
    assert ledger.snapshot().settled == pytest.approx(0.9)


def test_an_unsettled_reservation_is_released_on_the_way_out(tmp_path):
    """An attempt that never ran must not hold budget forever."""
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    with ledger.reservation("a", 0.5):
        pass                            # never settled
    assert ledger.snapshot().committed == 0


def test_a_raising_body_still_releases(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    with pytest.raises(RuntimeError), ledger.reservation("a", 0.5):
        raise RuntimeError("container died")
    assert ledger.snapshot().committed == 0


def test_reservation_refuses_loudly_when_the_budget_is_gone(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=0.1)
    # The message now names the cause rather than asserting one.
    with pytest.raises(BudgetExhausted, match="committed"), \
            ledger.reservation("a", 0.5):
        pass


def test_a_settled_cell_cannot_be_reserved_again(tmp_path):
    """Resuming over a finished cell must not double-count it."""
    ledger = Ledger(tmp_path / "led.json", budget=10.0)
    ledger.settle("a", 0.2)
    granted, why = ledger.reserve("a", 0.2)
    assert not granted
    assert "already settled" in why
    assert ledger.snapshot().settled == pytest.approx(0.2)


def test_the_refusal_reason_distinguishes_broke_from_already_paid(tmp_path):
    """The bug: one bare False for two opposite situations.

    "Budget gone" means stop the sweep. "Already settled" means retire the
    entry and run the cell again. The caller printed "BUDGET EXHAUSTED:
    $0.56 of $9.00" while refusing a $0.04 attempt — a true sentence about
    the wrong cause, which reads as a fact and sends you looking in the
    wrong place.
    """
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    ledger.settle("done", 0.1)

    _, paid = ledger.reserve("done", 0.05)
    assert "already settled" in paid

    _, broke = ledger.reserve("new", 5.0)
    assert "committed" in broke and "already settled" not in broke


def test_retiring_frees_a_cell_without_losing_its_spend(tmp_path):
    """A cell can be settled and still need to run again.

    22 attempts were settled before the harness could tell a rate-limited
    run from one that measured nothing. The money is gone either way, so
    the total must not drop — but the cell has to be reservable.
    """
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    ledger.settle("a", 0.20)
    before = ledger.snapshot().settled

    moved = ledger.retire("a")
    assert moved == pytest.approx(0.20)
    assert ledger.snapshot().settled == pytest.approx(before), \
        "retiring must not make spent money disappear"
    assert ledger.reserve("a", 0.05)[0], "the cell must be runnable again"


def test_retiring_an_unknown_cell_is_harmless(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    assert ledger.retire("never-ran") == 0.0
    assert ledger.snapshot().settled == 0.0


def test_retiring_twice_does_not_duplicate_the_spend(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    ledger.settle("a", 0.2)
    ledger.retire("a")
    ledger.retire("a")
    assert ledger.snapshot().settled == pytest.approx(0.2)


def test_retiring_the_same_cell_twice_keeps_both_amounts(tmp_path):
    """Sunk spend must survive a second retirement of the same cell.

    The suffix on a retired key used to count the settled entries. That
    count goes DOWN when a zero-cost cell is retired, so a cell retired,
    re-run and retired again could land on a key already in use and
    overwrite the first amount. Total spend would fall by the cost of a
    run that really happened, and the budget would quietly refund itself.
    """
    ledger = Ledger(tmp_path / "led.json", budget=10.0)
    ledger.settle("free", 0.0)          # a run whose provider reported no usage
    ledger.settle("a", 0.10)
    ledger.retire("a")
    ledger.retire("free")               # shrinks the settled map
    ledger.settle("a", 0.20)            # the cell ran again
    ledger.retire("a")

    assert ledger.snapshot().settled == pytest.approx(0.30), \
        "both attempts were paid for; both must still be counted"


def test_adopting_prior_spend_happens_once(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=10.0)
    ledger.adopt("a", 0.3)
    ledger.adopt("a", 0.3)
    assert ledger.snapshot().settled == pytest.approx(0.3)


def test_two_sweeps_over_one_directory_share_a_budget(tmp_path):
    """Two terminals, one runs/ directory, one budget.

    Each Ledger instance is a view of the same file, so a second sweep
    cannot authorise the full amount again.
    """
    first = Ledger(tmp_path / "led.json", budget=1.0)
    second = Ledger(tmp_path / "led.json", budget=1.0)
    first.settle("a", 0.8)
    assert not second.reserve("b", 0.5)[0]
    assert second.snapshot().settled == pytest.approx(0.8)


def test_an_unreadable_ledger_raises_rather_than_starting_from_zero(tmp_path):
    """Absent is never zero, applied to money.

    Treating a corrupt ledger as empty would forget everything already
    spent and authorise the budget a second time.
    """
    path = tmp_path / "led.json"
    path.write_text("{ not json")
    with pytest.raises(RuntimeError, match="unreadable"):
        Ledger(path, budget=1.0).snapshot()


def test_a_foreign_json_file_is_not_treated_as_a_ledger(tmp_path):
    path = tmp_path / "led.json"
    path.write_text(json.dumps({"something": "else"}))
    with pytest.raises(RuntimeError, match="not a ledger"):
        Ledger(path, budget=1.0).snapshot()


def test_the_ledger_is_written_atomically(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    ledger.settle("a", 0.1)
    assert not list(tmp_path.glob("*.tmp"))


def test_negative_amounts_are_refused(tmp_path):
    ledger = Ledger(tmp_path / "led.json", budget=1.0)
    with pytest.raises(ValueError):
        ledger.reserve("a", -1.0)
    with pytest.raises(ValueError):
        ledger.settle("a", -1.0)
    with pytest.raises(ValueError):
        Ledger(tmp_path / "x.json", budget=-1.0)
