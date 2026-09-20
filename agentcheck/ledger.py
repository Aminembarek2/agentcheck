"""A spend ledger that two workers cannot both read as affordable.

The sweep's budget check used to be read-then-act:

    if budget - spent >= estimate:
        run_the_attempt()

Correct while attempts were strictly serial, and a race the moment they are
not. Two workers reading `spent` at the same moment both see room for one
more attempt, both start, and the sweep overshoots by however many workers
are running. That does not raise — it produces a bill slightly larger than
the cap, which is exactly the shape of failure this project exists to
avoid: a bound that quietly does not hold rather than an error that stops.

So spend is RESERVED before an attempt starts and SETTLED against the real
cost afterwards, under a lock held across both the read and the write. A
reservation is pessimistic on purpose: an attempt that has not finished has
no measured cost, and assuming it will be cheap is how a cap is exceeded.

The lock is an OS file lock rather than a `threading.Lock`, for two
reasons. Workers may become processes later, and — more immediately — two
sweeps run from two terminals against the same `runs/` directory would
otherwise share a budget they cannot see. The file is the shared state, so
the lock belongs on the file.

Nothing here estimates cost. It is given numbers and keeps them consistent.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BudgetExhausted(RuntimeError):
    """A reservation was refused because it would exceed the budget."""


@dataclass(frozen=True)
class Snapshot:
    """What the ledger holds at one instant."""
    budget: float
    settled: float
    reserved: float
    open_reservations: int

    @property
    def committed(self) -> float:
        """Everything that is spent or promised.

        The number a budget check must use. Comparing against `settled`
        alone ignores the attempts currently running, which is the race
        this class exists to close.
        """
        return self.settled + self.reserved

    @property
    def remaining(self) -> float:
        return self.budget - self.committed

    def as_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "settled": self.settled,
                "reserved": self.reserved,
                "open_reservations": self.open_reservations,
                "committed": self.committed, "remaining": self.remaining}


class Ledger:
    """File-backed, lock-protected budget accounting for one sweep.

        ledger = Ledger(Path("runs/.sweep-ledger.json"), budget=20.0)
        with ledger.reservation("002-ds-flash-r01", estimate=0.12) as claim:
            ...run the attempt...
            claim.settle(actual_cost)

    Leaving the block without settling releases the reservation: an attempt
    that never ran must not hold budget forever. Settling with a cost
    larger than the reservation is allowed and recorded — a cap that
    refused to admit an overspend it had already incurred would be
    reporting a comfortable number instead of the real one.
    """

    def __init__(self, path: Path, budget: float):
        if budget < 0:
            raise ValueError("budget must not be negative")
        self.path = Path(path)
        self.budget = float(budget)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- storage ------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        """Hold an exclusive lock across a read-modify-write.

        The lock lives on a separate file from the data. Locking the data
        file itself would mean taking the lock on a handle that a rewrite
        replaces, and the second writer would be locking a file no longer
        at that path.
        """
        with open(self._lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._read()
                yield state
                self._write(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"settled": {}, "reserved": {}}
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            # A ledger that cannot be read is not an empty ledger. Starting
            # from zero here would forget everything already spent and let
            # the sweep run the budget twice.
            raise RuntimeError(
                f"{self.path} is unreadable. It holds this sweep's spend, "
                f"and treating it as empty would authorise the budget a "
                f"second time. Inspect it, or delete it deliberately."
            ) from e
        for key in ("settled", "reserved"):
            if not isinstance(raw.get(key), dict):
                raise RuntimeError(f"{self.path}: no {key!r} map — this is "
                                   f"not a ledger this harness wrote")
        return raw

    def _write(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    # --- accounting ---------------------------------------------------------

    def snapshot(self) -> Snapshot:
        state = self._read()
        return Snapshot(
            budget=self.budget,
            settled=sum(state["settled"].values()),
            reserved=sum(state["reserved"].values()),
            open_reservations=len(state["reserved"]),
        )

    def reserve(self, key: str, estimate: float) -> tuple[bool, str]:
        """Claim `estimate` for `key`. Atomic against every other reserver.

        Returns (granted, reason-if-refused). The reason is not decoration:
        a refusal can mean the budget is gone OR that this cell was already
        paid for, and those call for opposite responses — stop the sweep,
        versus retire the old entry and run it again.

        The first version returned a bare False for both, and the caller
        printed "BUDGET EXHAUSTED: $0.56 of $9.00" while refusing a $0.04
        attempt. A true statement about the wrong cause reads as a fact and
        sends you looking in the wrong place.
        """
        if estimate < 0:
            raise ValueError("an estimate must not be negative")
        with self._locked() as state:
            if key in state["settled"]:
                return False, (
                    f"already settled at ${state['settled'][key]:.4f}; "
                    f"retire it first if the cell is being re-run")
            committed = (sum(state["settled"].values())
                         + sum(state["reserved"].values()))
            if committed + estimate > self.budget:
                return False, (
                    f"${committed:.2f} of ${self.budget:.2f} committed and "
                    f"the next attempt needs ${estimate:.3f}")
            state["reserved"][key] = estimate
            return True, ""

    def retire(self, key: str) -> float:
        """Free a settled cell for re-running, without losing its spend.

        A cell can be settled and yet need to run again: the first ladder
        sweep settled 64 cells, then 22 of them turned out to have been
        refused service by the provider and to have measured nothing. The
        money is gone either way — but the cell has to be reservable again,
        and the total must not quietly drop when it is.

        So the old amount moves to a sunk key rather than being deleted.
        Total spend is unchanged; the cell is free. Returns what was moved.
        """
        with self._locked() as state:
            amount = state["settled"].pop(key, 0.0)
            if amount:
                # The suffix counted the settled entries, which is not a
                # counter: a retire that frees a zero-cost cell shrinks
                # that count, so a later retire of the same cell can
                # produce a key already in use and overwrite the earlier
                # sunk amount. Total spend would then drop by the amount
                # of a run that really happened — a cap quietly funding
                # itself, which is the one failure this file exists to
                # prevent. Probe for a free suffix instead.
                n = 1
                while f"{key}#retired-{n}" in state["settled"]:
                    n += 1
                state["settled"][f"{key}#retired-{n}"] = amount
            return amount

    def settle(self, key: str, actual: float) -> None:
        """Replace a reservation with what the attempt really cost."""
        if actual < 0:
            raise ValueError("a cost must not be negative")
        with self._locked() as state:
            state["reserved"].pop(key, None)
            state["settled"][key] = actual

    def release(self, key: str) -> None:
        """Drop a reservation for an attempt that produced no cost.

        Used when a container never started. An attempt that failed before
        reaching the model spent nothing, and holding its reservation would
        shrink the budget for work that did happen.
        """
        with self._locked() as state:
            state["reserved"].pop(key, None)

    def adopt(self, key: str, cost: float) -> None:
        """Record spend for a cell that completed outside this sweep.

        A resumed sweep skips cells whose records already exist; their
        cost is real and has to be in the ledger, or the budget is
        authorised twice across two invocations.
        """
        with self._locked() as state:
            state["reserved"].pop(key, None)
            state["settled"][key] = cost

    @contextmanager
    def reservation(self, key: str, estimate: float) -> Iterator[Claim]:
        """Reserve, run, settle — releasing if the body never settles."""
        granted, why = self.reserve(key, estimate)
        if not granted:
            raise BudgetExhausted(
                f"cannot reserve ${estimate:.3f} for {key}: {why}")
        claim = Claim(self, key)
        try:
            yield claim
        finally:
            if not claim.settled:
                self.release(key)


@dataclass
class Claim:
    ledger: Ledger
    key: str
    settled: bool = False

    def settle(self, actual: float) -> None:
        self.ledger.settle(self.key, actual)
        self.settled = True
