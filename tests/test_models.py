"""Tests for the model registry.

Pure — no container, no API calls.

    pytest tests/test_models.py -v
"""

import os
from pathlib import Path

import pytest

from agentcheck.agent import estimate_cost
from agentcheck.models import (
    FALLBACK_PRICE,
    MODELS,
    PRICES,
    build_model,
    model_id,
    spec_for_id,
)


def test_every_registered_model_has_a_price():
    """The bug this makes impossible. When the id table and the price table
    lived in two files they did not have to agree, and a model present in
    one but missing from the other fell silently through to the
    most-expensive fallback — a run then hit its "$0.50" cap after nine
    iterations having actually spent about two cents."""
    for alias, spec in MODELS.items():
        assert spec.id in PRICES, f"{alias} ({spec.id}) has no price"
        assert PRICES[spec.id]["input"] > 0
        assert PRICES[spec.id]["output"] > 0


def test_the_fallback_is_the_most_expensive_entry():
    assert FALLBACK_PRICE["output"] == max(p["output"] for p in PRICES.values())


def test_an_unregistered_alias_is_refused_not_guessed():
    with pytest.raises(SystemExit):
        build_model("gpt-9")


def test_a_missing_api_key_is_refused_before_any_container_work():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            build_model("haiku")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_model_id_is_read_from_either_provider_attribute():
    """`model` on ChatAnthropic, `model_name` on ChatOpenAI. Reading only
    one lands every run of the other provider on the fallback price."""
    class Anthropic:
        model = "claude-opus-5"

    class OpenAI:
        model_name = "deepseek-v4-flash"
    assert model_id(Anthropic()) == "claude-opus-5"
    assert model_id(OpenAI()) == "deepseek-v4-flash"
    assert model_id(object()) == ""


def test_an_unknown_model_is_billed_at_the_most_expensive_rate():
    """A cap that under-counts is not a cap."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert estimate_cost(usage, "renamed-model-9") >= \
        estimate_cost(usage, "claude-opus-5")


def test_cached_input_is_billed_at_the_cached_rate():
    plain = {"input_tokens": 100_000, "output_tokens": 0}
    cached = {"input_tokens": 100_000, "output_tokens": 0,
              "input_token_details": {"cache_read": 99_000}}
    m = "claude-haiku-4-5-20251001"
    assert estimate_cost(cached, m) < estimate_cost(plain, m) / 5


def test_writing_the_cache_costs_more_than_a_plain_read():
    m = "claude-haiku-4-5-20251001"
    write = {"input_tokens": 100_000, "output_tokens": 0,
             "input_token_details": {"cache_creation": 100_000}}
    plain = {"input_tokens": 100_000, "output_tokens": 0}
    assert estimate_cost(write, m) > estimate_cost(plain, m)


@pytest.mark.parametrize("model", [
    "qwen/qwen3-coder-next", "xiaomi/mimo-v2.5-pro", "unknown-provider",
])
def test_unknown_cache_rates_do_not_discount_the_spend_cap(model):
    plain = {"input_tokens": 100_000, "output_tokens": 1_000}
    cached = {**plain, "input_token_details": {"cache_read": 90_000}}
    assert estimate_cost(cached, model) == estimate_cost(plain, model)


def test_no_usage_costs_nothing_here_and_is_flagged_by_the_caller():
    """estimate_cost cannot represent "unknown" in a float. The agent
    tracks that separately as cost_known — see test_agent.py."""
    assert estimate_cost({}, "claude-opus-5") == 0.0


def test_legacy_aliases_are_marked():
    """Migration aliases retarget under you, so results from different
    weeks stop being comparable."""
    assert MODELS["deepseek"].legacy
    assert not MODELS["ds-flash"].legacy


def test_spec_lookup_by_id_round_trips():
    for spec in MODELS.values():
        assert spec_for_id(spec.id) is spec
    assert spec_for_id("nope") is None


def test_a_partial_usage_dict_raises_rather_than_under_counting():
    """Worse than an absent one. Absent is caught by the caller and flips
    cost_known; a dict that is present but missing output_tokens defaults
    to zero, under-counts silently, and takes the spend cap with it."""
    with pytest.raises(ValueError, match="incomplete token usage"):
        estimate_cost({"input_tokens": 1000}, "claude-opus-5")
    with pytest.raises(ValueError):
        estimate_cost({"output_tokens": 1000}, "claude-opus-5")


def test_only_anthropic_models_declare_prompt_caching():
    """`cache_control` blocks are Anthropic syntax. Sending them to an
    OpenAI-compatible endpoint is not a no-op — it is a different content
    shape that the provider may reject or silently mangle, so the flag has
    to be consulted rather than defaulted on."""
    for spec in MODELS.values():
        if spec.supports_caching:
            assert spec.provider == "anthropic", spec.alias


def test_run_agent_gates_caching_on_the_provider():
    """The field existed but nothing read it: every model got caching on,
    including DeepSeek."""
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "run_agent.py").read_text()
    assert "supports_caching" in source, \
        "run_agent must consult supports_caching, not just --no-cache"


def test_the_default_model_is_the_cheapest_arm():
    """A forgotten --model should be the least expensive mistake available.
    Your own runs: haiku-02 cost $2.08, a capped ds-flash run $0.50."""
    from agentcheck.agent import DEFAULT_MODEL
    spec = spec_for_id(DEFAULT_MODEL)
    assert spec is not None, f"{DEFAULT_MODEL} is not in the registry"
    cheapest = min(MODELS.values(), key=lambda s: s.output_price)
    assert spec.output_price == cheapest.output_price


def test_the_default_model_is_built_through_the_registry():
    """Hard-wiring ChatAnthropic here built an Anthropic client with a
    DeepSeek model id the moment the default changed provider."""
    source = (Path(__file__).resolve().parent.parent
              / "agentcheck" / "agent.py").read_text()
    head = source[:source.index("def _build_tools")]
    assert "ChatAnthropic(" not in head, \
        "agent.py must not construct a provider client directly"


# --- phase 2: routing through an aggregator ---------------------------------

from agentcheck.models import OPENROUTER_BASE_URL
from agentcheck.record import ConfigFingerprint


def _fingerprint(**over):
    base = dict(model_id="m", tool_signature=(("read_file", "Read."),),
                system_prompt="p", max_iterations=50, max_cost_usd=1.0,
                max_wall_seconds=1800.0, test_command="pytest",
                image_id="sha256:abc", max_tokens=8192, max_file_lines=400,
                max_search_hits=40)
    base.update(over)
    return ConfigFingerprint(**base)


def test_an_unpinned_route_does_not_claim_a_provider():
    """`provider_route` must not assert what nothing verified.

    Routing is left to OpenRouter, so the serving endpoint is unknown at
    run time. The route label says "openrouter" and stops there — naming a
    provider would put a claim into config_version that no part of the run
    checked, which is the same shape as a filename standing in for a
    configuration.
    """
    for alias, spec in MODELS.items():
        if spec.base_url == OPENROUTER_BASE_URL:
            assert spec.route == "openrouter", alias
            assert "UNPINNED" not in spec.route


def test_pinning_still_works_if_it_is_ever_wanted_again():
    """The mechanism stays; only the policy changed.

    A shared pool rate-limited a third of the first ladder sweep, and
    pinning with fallbacks off is the lever for that. Keeping it exercised
    means it will not have quietly rotted the day it is needed.
    """
    from dataclasses import replace
    pinned = replace(MODELS["or-ds-flash"], provider_pin="deepinfra")
    assert pinned.route == "openrouter:deepinfra"


def test_a_direct_spec_contributes_no_route():
    """So the fifteen archived runs' digests are unchanged by this field."""
    assert MODELS["ds-flash"].route == ""
    assert _fingerprint().digest() == _fingerprint(provider_route="").digest()


def test_two_upstreams_are_two_configurations():
    a = _fingerprint(provider_route="openrouter:deepseek").digest()
    b = _fingerprint(provider_route="openrouter:novita").digest()
    assert a != b


def test_the_archives_direct_model_ids_still_price():
    """Deleting the direct specs would drop 15 runs onto the fallback rate.

    `spec_for_id` is what values the stored records; the OpenRouter entries
    carry different ids, so both sets have to stay registered.
    """
    for archived_id in ("deepseek-v4-flash", "claude-haiku-4-5-20251001"):
        assert spec_for_id(archived_id) is not None, archived_id


def test_a_routed_model_never_claims_anthropic_caching():
    """run_agent only emits cache_control for provider == 'anthropic'.

    A routed Anthropic model is built as an OpenAI-compatible client, so
    claiming support here would promise a discount that is never taken and
    make the cost forecast wrong in the optimistic direction.
    """
    for spec in MODELS.values():
        if spec.base_url == OPENROUTER_BASE_URL:
            assert not spec.supports_caching, spec.alias


def test_no_registered_model_is_a_floating_alias_except_the_known_one():
    """OpenRouter serves `~deepseek/deepseek-v4-flash-latest`, which
    retargets without notice. Only the one legacy entry kept for
    reproducing old runs may be a moving target, and it warns."""
    for alias, spec in MODELS.items():
        if spec.legacy:
            continue
        assert "latest" not in spec.id, f"{alias} pins a floating alias"
