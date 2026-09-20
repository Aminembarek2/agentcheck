"""Put the repository root on the path so `scripts/` is importable.

The package itself is installed (`pip install -e .`), but the scripts are
deliberately not a package — they are entry points, not a library. Their
pure helpers are still worth testing, and this is the least surprising way
to reach them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest

from agentcheck.record import SCHEMA_VERSION


@pytest.fixture
def good_record() -> dict:
    """A minimal record that loads under the current schema.

    Kept here rather than in one test module because more than one suite
    now needs "a record that is fine", and two divergent copies of it
    would eventually disagree about what fine means.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "config_version": "abc123def456",
        "task_id": "002-databases",
        "model": "deepseek-v4-flash",
        "outcome": "solved",
        "verdict_status": "ok",
        "diff": "",
        "failed_ids": [],
        "before_failed_ids": ["tests/test_db.py::test_1"],
        "iterations": 10,
        "cost_usd": 0.5,
        "cost_known": True,
        "max_iterations": 50,
        "max_cost_usd": 1.0,
    }
