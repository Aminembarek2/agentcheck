"""Optional Langfuse tracing for the agent loop.

Strictly optional, and strictly non-authoritative. Those two words carry
the whole design:

  * OPTIONAL. Disabled unless LANGFUSE_PUBLIC_KEY is set. Import failures,
    network failures and authentication failures are caught, warned about
    once, and never propagate. A tracing backend that can end a run is a
    tracing backend that can cost you an afternoon of sweep for nothing.

  * NON-AUTHORITATIVE. `RunRecord` remains the only source of any number in
    RESULTS.md. Nothing here is ever read back for a result. A figure that
    lives only in a hosted dashboard is not traceable to a run record,
    which is the constraint the whole project is built on — and a dashboard
    can be reorganised, rate-limited or deleted by someone else.

So what is it for? Reading one run. A trace view of a fifty-iteration
agent loop is genuinely better than scrolling a JSON trajectory, and
Langfuse is what the DACH job postings this project is aimed at actually
name. That is the entire justification, and it is why the budget for this
module is one evening: it makes traces easier to read and contributes no
numbers.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: Set once, so a misconfigured backend warns once rather than on every
#: iteration of every run in a thirty-attempt sweep.
_warned = False


def _warn(message: str) -> None:
    global _warned
    if not _warned:
        print(f"tracing disabled: {message}", file=sys.stderr)
        _warned = True


def enabled() -> bool:
    """Tracing is opt-in via the environment, never by default."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def callback_handler(*, run_name: str, task_id: str, model: str,
                     config_version: str, tags: tuple[str, ...] = ()
                     ) -> Any | None:
    """A LangChain callback handler that reports to Langfuse, or None.

    None is a first-class return, not an error path: every caller passes
    the result straight into `callbacks=[...]` after filtering, so tracing
    being off is indistinguishable from tracing never having existed.

    The metadata mirrors what the record already stores. Duplication is
    deliberate — the trace has to be findable from a record and the record
    from a trace, and `config_version` is the field that makes two runs
    comparable, so it must appear on both sides.
    """
    if not enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        _warn("langfuse is not installed (pip install langfuse)")
        return None

    try:
        return CallbackHandler(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST",
                                "https://cloud.langfuse.com"),
            session_id=run_name,
            metadata={"task_id": task_id, "model": model,
                      "config_version": config_version},
            tags=[*tags, task_id, model],
        )
    except Exception as e:
        # Deliberately broad. The contract of this module is that nothing
        # in it can end a run, and a tracing SDK can raise anything from a
        # TypeError on a version mismatch to a socket error. There is no
        # value to protect here — only a convenience to lose.
        _warn(f"could not construct a Langfuse handler: {e}")
        return None


def callbacks(**kwargs: Any) -> list[Any]:
    """The handler as a list, empty when tracing is off."""
    handler = callback_handler(**kwargs)
    return [handler] if handler is not None else []


def flush() -> None:
    """Push buffered events before the process exits.

    Langfuse batches. A short run can finish and exit before anything is
    sent, which looks exactly like tracing being broken.
    """
    if not enabled():
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        _warn(f"could not flush traces: {e}")
