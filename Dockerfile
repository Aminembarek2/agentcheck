# agentcheck — task environment
#
# Builds a container pinned to a task's BROKEN state: the target repo at its
# pre-migration commit, with the dependency already bumped, so the suite fails
# exactly as recorded in the task's broken-report.json.
#
# This is the state the agent wakes up in.
#
# Nothing here is task-specific. Python version, repo and commit all arrive as
# build args from scripts/prepare_task.py, which reads task.yaml.

# ARG before FROM is the one place Docker allows it, and it does NOT carry past
# the FROM — it must be re-declared afterwards if needed again.
ARG PY_VERSION=3.11
FROM python:${PY_VERSION}-slim

# git             — clone the target at a specific commit
# build-essential — some era-correct pins predate wheels for newer Pythons and
#                   fall back to compiling from source
# procps          — pkill/ps, needed to reap test processes that hang at
#                   interpreter shutdown, and invaluable for debugging
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

ARG REPO_URL
ARG BASE_SHA

# Full clone, not shallow: an arbitrary SHA has to be reachable.
#
# The history is then DISCARDED and replaced with a single commit of the
# checked-out tree. A full clone of the upstream repo contains the
# maintainer's fix — the commit this task is asking the agent to
# reproduce — sitting in .git inside the sandbox. No current tool can
# reach it usefully, but that is a property of today's tool list rather
# than of the environment, and the moment a shell or git tool is added the
# answer key is one command away. Re-initialising keeps `git diff`,
# `git checkout --` and `git clean` working, which is all the harness
# needs, and leaves nothing to find.
RUN git clone "$REPO_URL" repo \
    && cd repo \
    && git checkout --detach "$BASE_SHA" \
    && rm -rf .git \
    && git init -q . \
    && git config user.email harness@agentcheck.local \
    && git config user.name agentcheck \
    && git add -A \
    && git commit -qm "task base state ($BASE_SHA)"

WORKDIR /work/repo

# Copied before the install so Docker caches this layer — editing the runner
# script won't trigger a full reinstall.
COPY broken-requirements.txt /work/broken-requirements.txt

# --no-deps: the frozen file is already a complete resolved set. Letting pip
# re-resolve would defeat the point of freezing it.
RUN pip install --no-cache-dir --no-deps -r /work/broken-requirements.txt \
    && pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir pytest-json-report

# The expected failure set is deliberately NOT copied into the image. It
# lists every failing nodeid with its full traceback, and the agent's
# read_file can reach anything under /work. Handing it the exact answer
# sheet measures the harness, not the agent. The host keeps it in
# tasks/<id>/broken-report.json, where scoring reads it.

# Reset the working tree in case installation touched anything.
RUN git checkout -- . && git clean -fd

# No CMD on purpose: the harness supplies the test command from task.yaml,
# since it differs per task (deselected modules, env vars, and so on).
