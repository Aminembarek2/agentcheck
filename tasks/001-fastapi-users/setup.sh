#!/usr/bin/env bash
# agentcheck — task 001 setup
#
# Repo:      fastapi-users/fastapi-users
# Migration: pydantic v1 -> v2
# Base SHA:  3bf0f888ee1847f52b8c5a96e30a1c90ea0f36e9  (pre-migration)
# Reference: PR #1249, merge 49ea718a6cabe53bf43edaae79eff2bfda7e0329
#
# Requires Python 3.11. Newer Pythons break the 2023-era dependency graph.
#
# Verified baseline: 546 passed in ~55s

set -euo pipefail

BASE_SHA="3bf0f888ee1847f52b8c5a96e30a1c90ea0f36e9"
IGNORE="--ignore=tests/test_authentication_strategy_redis.py"   # needs a live Redis

# --- 1. clone at the pre-migration commit -----------------------------------
git clone https://github.com/fastapi-users/fastapi-users target
cd target
git checkout "$BASE_SHA"

# --- 2. virtualenv ----------------------------------------------------------
python3.11 -m venv .venv
source .venv/bin/activate

# --- 3. install the package -------------------------------------------------
# NOTE: there is no [dev] extra. Test deps live in a hatch env, so they must
# be installed by hand.
pip install -e .

# --- 4. pin to the 2023-era stack -------------------------------------------
# Every one of these pins exists because the unpinned version broke the build:
#   pydantic >=2      -> the migration itself; must stay on v1 for the baseline
#   fastapi  >=0.100  -> first release requiring pydantic v2
#   bcrypt   5.x      -> dropped __about__, which passlib 1.7.4 reads
#   pytest   8/9.x    -> forbids marks on fixtures; this repo does that everywhere
pip install \
  "pydantic<2" \
  "fastapi<0.100" \
  "bcrypt==4.0.1" \
  "httpx-oauth==0.13.0" \
  "pytest==7.4.0" \
  "pytest-asyncio==0.21.0" \
  "pytest-mock==3.11.1" \
  httpx \
  asgi-lifespan

# --- 5. verify the baseline is green ----------------------------------------
pytest tests/ $IGNORE -q
# expected: 546 passed

# --- 6. freeze the exact resolution -----------------------------------------
# Reproducibility is not optional: an unpinned graph resolves differently on
# different days, which would make every measurement meaningless.
pip freeze > ../baseline-requirements.txt

# --- 7. introduce the break -------------------------------------------------
pip install "pydantic>=2" "fastapi>=0.100"
pytest tests/ $IGNORE -q || true
# expected: some subset of the 546 now fails -> this is the task

pip freeze > ../broken-requirements.txt
