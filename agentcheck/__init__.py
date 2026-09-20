"""agentcheck — does the agent fix the upgrade, or just make tests green?

Layered bottom-up, each layer testable without the one above it:

    sandbox   container lifecycle, file access, isolation
    tools     the agent's surface on the repository
    agent     the LangGraph loop
    scorer    deterministic cheat detection and progress
    judge     pairwise LLM comparison against the maintainer's real PR
    record    the versioned run schema everything reads and writes
    task      task definitions and their recorded artifacts
    stats     Wilson intervals, bootstrap, kappa
    diffparse structured unified-diff parsing

One rule governs all of them: ABSENT IS NEVER ZERO. Every serious bug this
project has had was a missing value that silently became a plausible
number, so nothing a result depends on is ever defaulted.
"""

__version__ = "0.3.0"

SCHEMA_VERSION = 3
