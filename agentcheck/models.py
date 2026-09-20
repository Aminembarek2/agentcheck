"""Model registry: ids, pricing, and construction, in one place.

These three facts have to agree, and when they lived in two files they did
not have to. `run_agent.MODELS` named the model ids; `agent._PRICES` priced
them; nothing checked that a key in one existed in the other. A model added
to the first and forgotten in the second silently fell through to the
unknown-model fallback, which bills at the most expensive rate in the table
— and a run then hit its "$0.50" cap after nine iterations having actually
spent about two cents.

There is now one table, and a test asserts every registered model has a
price.

Model ids are pinned explicitly. Migration aliases like `deepseek-chat`
retarget under you without notice, so results from different weeks stop
being comparable; one is kept only to reproduce earlier runs, and using it
warns.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

DEFAULT_MAX_TOKENS = 8192

#: Retries inside the provider SDK, for transient HTTP failures — 429s,
#: 5xx, dropped connections. The default of 2 is too thin for an unattended
#: sweep: a single rate-limit blip four hours in would otherwise end that
#: run as a harness error and cost the attempt. These are retries of the
#: same call, so they do not affect what the agent does or what a run
#: means, only whether a network hiccup destroys it.
MAX_HTTP_RETRIES = 6


#: OpenRouter's OpenAI-compatible endpoint. One key, one balance, one base
#: URL for every model below — which is the point: four direct provider
#: accounts is four places for a key to expire mid-sweep.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    id: str
    provider: str                      # "anthropic" | "openai_compatible"
    env: str
    #: USD per million tokens.
    input_price: float
    output_price: float
    base_url: str = ""
    #: An alias whose target can change under you. Usable, but never for a
    #: number you intend to report.
    legacy: bool = False
    #: Anthropic-style prompt caching. Reads cost a tenth of the input
    #: rate; writing the cache costs 1.25x.
    supports_caching: bool = False

    #: The vendor's model, independent of the route it is reached through
    #: and of the snapshot pin. `deepseek-v4-flash` (direct API) and
    #: `deepseek/deepseek-v4-flash-0731` (OpenRouter) are two ids for one
    #: model, and a report that counts ids counts two models where there
    #: is one — the same shape of error as reading a missing value as a
    #: plausible number, and worse here, because the project's scope asks
    #: for two models and a reader would take the count as evidence it was
    #: met.
    #:
    #: Left EMPTY where the target is genuinely unknown (a floating alias).
    #: An empty family falls back to the id, so an unknown model is never
    #: silently merged with a known one.
    family: str = ""

    #: Optionally, the single upstream this model may be served from.
    #: EMPTY EVERYWHERE at present: routing is left to OpenRouter, which
    #: picks among the endpoints serving a model.
    #:
    #: The cost of that is real and is recorded in
    #: docs/threats-to-validity.md rather than argued about here — this
    #: model has thirty endpoints whose prices span 9x, so two runs of
    #: "the same model" can be served by different hosts, and the record
    #: cannot say which.
    #:
    #: The benefit is that it works. Pinning with `allow_fallbacks: false`
    #: means a busy or renamed provider fails the attempt outright, and a
    #: pinned shared pool rate-limited a third of the first ladder sweep.
    #: Setting this again is one edit; `build_model` still honours it.
    provider_pin: str = ""

    @property
    def route(self) -> str:
        """What goes into `ConfigFingerprint.provider_route`.

        "openrouter" alone, when unpinned, because that is all that is
        actually known. Naming a provider here would put a claim into
        `config_version` that no part of the run verified — the field
        exists to separate configurations, not to assert one.
        """
        if self.base_url != OPENROUTER_BASE_URL:
            return ""
        return (f"openrouter:{self.provider_pin}" if self.provider_pin
                else "openrouter")

    @property
    def model_family(self) -> str:
        """What to group by when the question is about a MODEL.

        Never the id: two ids can be one model, and grouping by id turns
        one arm into two that look comparable.
        """
        return self.family or self.id

    @property
    def price(self) -> dict[str, float]:
        return {"input": self.input_price, "output": self.output_price}


_SPECS = (
    ModelSpec("haiku", "claude-haiku-4-5-20251001", "anthropic",
              "ANTHROPIC_API_KEY", 1.00, 5.00, supports_caching=True,
              family="claude-haiku-4.5"),
    ModelSpec("sonnet", "claude-sonnet-5", "anthropic",
              "ANTHROPIC_API_KEY", 2.00, 10.00, supports_caching=True,
              family="claude-sonnet-5"),
    ModelSpec("opus", "claude-opus-5", "anthropic",
              "ANTHROPIC_API_KEY", 5.00, 25.00, supports_caching=True,
              family="claude-opus-5"),
    #: STALE PRICE, kept as-is on purpose. $0.14/$0.28 was DeepSeek's flat
    #: rate until 2026-08-16, when it moved to peak/off-peak. The fifteen
    #: archived runs were priced with this table at run time, so their
    #: stored `cost_usd` is understated by 1.6-4.7x. Correcting the table
    #: now would not correct those records — cost is computed once and
    #: stored — and would silently reprice a history it did not produce.
    #: RESULTS.md does not quote them as cost facts; see
    #: docs/threats-to-validity.md.
    ModelSpec("ds-flash", "deepseek-v4-flash", "openai_compatible",
              "DEEPSEEK_API_KEY", 0.14, 0.28,
              base_url="https://api.deepseek.com/v1",
              family="deepseek-v4-flash"),
    ModelSpec("ds-pro", "deepseek-v4-pro", "openai_compatible",
              "DEEPSEEK_API_KEY", 0.55, 2.19,
              base_url="https://api.deepseek.com/v1",
              family="deepseek-v4-pro"),
    ModelSpec("deepseek", "deepseek-chat", "openai_compatible",
              "DEEPSEEK_API_KEY", 0.14, 0.28,
              base_url="https://api.deepseek.com/v1", legacy=True),

    # --- via OpenRouter ---------------------------------------------------
    #
    # The direct-provider entries above are KEPT rather than replaced. The
    # fifteen records in the archive were produced through them, and
    # `spec_for_id` is what prices those runs; deleting the specs would
    # drop every one of them onto the unknown-model fallback rate.
    #
    # An `or-` prefix marks an alias only where the same model is also
    # reachable directly, so there is something to disambiguate. `mimo` and
    # `kimi` exist on one route only and take the plain name.
    #
    # Prices read from OpenRouter's model pages in September 2026. Two of
    # them are BELOW the direct provider's published rate, which is worth
    # re-checking before a sweep rather than assuming — these numbers drive
    # the spend cap, and a wrong one fired a "$0.50" cap after nine
    # iterations in phase 1.
    #
    # Explicit cache control is off for these clients. Upstreams may
    # still cache automatically; estimates do not assume a cache discount.

    #: Pinned to the 0731 snapshot. OpenRouter also serves
    #: `~deepseek/deepseek-v4-flash-latest`, which retargets without
    #: notice — the same trap as the `deepseek-chat` alias above. The
    #: archive's runs used the direct API's floating `deepseek-v4-flash`,
    #: so they are NOT comparable with these and correctly get a different
    #: config_version.
    #:
    #: The route is unpinned. These historical rates are not an upper
    #: bound across providers; existing run costs are estimates at the
    #: recorded registry rates, not verified bills. See threats-to-validity.
    ModelSpec("or-ds-flash", "deepseek/deepseek-v4-flash-0731",
              "openai_compatible", "OPENROUTER_API_KEY", 0.08, 0.18,
              base_url=OPENROUTER_BASE_URL,
              family="deepseek-v4-flash"),

    #: Conservative rates across all endpoints AND context-length tiers.
    #: Checked 2026-09-17 against OpenRouter's model endpoints API:
    #: /api/v1/models/qwen/qwen3-coder-next/endpoints
    #: Alibaba's base $0.30/$1.50 rises to $0.50/$2.50 at 32K prompt
    #: tokens and $0.80/$4.00 at 128K. Do not use only the base rates.
    ModelSpec("qwen3-coder-next", "qwen/qwen3-coder-next",
              "openai_compatible", "OPENROUTER_API_KEY", 0.80, 4.00,
              base_url=OPENROUTER_BASE_URL, family="qwen3-coder-next"),

    #: Registered judge (judge-protocol.md §9, amendment A1). ~1.05M context.
    #:
    #: PRICED AT THE DEAREST ENDPOINT. Unpinned, OpenRouter serves this from
    #: any of 7 providers, $0.304-0.522 in and $0.609-1.500 out per M
    #: (read from /api/v1/models/xiaomi/mimo-v2.5-pro/endpoints on
    #: 2026-09-14). It was priced at the cheapest, $0.3045/$0.609, which
    #: under-counted every call routed elsewhere — and a spend cap reading
    #: an under-count is not a cap. The registry's rule for an unknown
    #: price applies to an unknown route: take the most expensive.
    #: The 2026-09-14 preflight also found DigitalOcean at $1.50 out,
    #: above the previous $1.044 maximum. No MiMo verdict had run yet.
    ModelSpec("mimo", "xiaomi/mimo-v2.5-pro",
              "openai_compatible", "OPENROUTER_API_KEY", 0.522, 1.500,
              base_url=OPENROUTER_BASE_URL, family="mimo-v2.5-pro"),

    #: Registered judge (§9, A1). Never run as an agent, so it never grades
    #: its own output.
    #:
    #: PRICED AT THE DEAREST ENDPOINT, as above: 21 providers, $0.58-1.09 in
    #: and $2.44-4.60 out per M on 2026-09-13. The previous $0.5187/$2.184
    #: was below even the cheapest of them, so every call was under-counted
    #: by 11-111% on input — the judge run's cap would have let it spend
    #: roughly twice what it reported.
    ModelSpec("kimi", "moonshotai/kimi-k2.6",
              "openai_compatible", "OPENROUTER_API_KEY", 1.09, 4.60,
              base_url=OPENROUTER_BASE_URL, family="kimi-k2.6"),

    #: Previous judge option retained for historical configurations.
    ModelSpec("or-haiku", "anthropic/claude-haiku-4.5",
              "openai_compatible", "OPENROUTER_API_KEY", 1.00, 5.00,
              base_url=OPENROUTER_BASE_URL, family="claude-haiku-4.5"),
)

MODELS: dict[str, ModelSpec] = {s.alias: s for s in _SPECS}

#: USD per million tokens, keyed by model ID (not alias). Drives both the
#: reported cost and the spend CAP, so an unknown model is charged at the
#: most expensive rate here: a cap that under-counts is not a cap.
#:
#: Verify against provider pricing pages before a sweep — these move.
PRICES: dict[str, dict[str, float]] = {s.id: s.price for s in _SPECS}

FALLBACK_PRICE = max(PRICES.values(), key=lambda p: p["output"])


#: Model ids already reported as missing, so the warning prints once per
#: process rather than once per API call.
_WARNED: set[str] = set()


def estimate_cost(usage: Any, model: str = "") -> float:
    """Estimate the cost of one call.

    A missing price entry falls back to the most expensive model, which
    makes the spend cap conservative — but it also silently distorts every
    cost figure, and a run can hit its cap in nine iterations while
    appearing to have spent fifty cents. That happened, so the fallback
    now announces itself instead of failing quietly.

    Only clients with explicit Anthropic cache pricing use its discounts.
    Other routes use the full input rate: upstream cache prices vary and
    an assumed discount would under-count the spend cap.
    """
    if not usage:
        return 0.0
    if "input_tokens" not in usage or "output_tokens" not in usage:
        # A usage dict that is present but incomplete is worse than an
        # absent one: absent is caught by the caller and flips cost_known,
        # while a partial dict defaulted to zero under-counts silently and
        # takes the spend cap down with it.
        raise ValueError(
            f"incomplete token usage {sorted(usage)} — the cost of this "
            f"call cannot be computed, and defaulting it to zero would "
            f"disable the spend cap")
    price = PRICES.get(model)
    if price is None:
        if model not in _WARNED:
            _WARNED.add(model)
            print(f"WARNING: no price entry for {model!r} — billing at the "
                  f"most expensive rate in the table. Costs and the spend "
                  f"cap will both be wrong. Add it to PRICES.",
                  file=sys.stderr)
        price = FALLBACK_PRICE

    spec = spec_for_id(model)
    if spec is None or not spec.supports_caching:
        return (int(usage["input_tokens"]) / 1e6 * price["input"]
                + int(usage["output_tokens"]) / 1e6 * price["output"])

    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = int(details.get("cache_creation") or 0)
    # LangChain reports input_tokens as the TOTAL, cached portions included.
    fresh = max(0, int(usage["input_tokens"]) - cache_read - cache_write)

    return (fresh / 1e6 * price["input"]
            + cache_write / 1e6 * price["input"] * 1.25   # writing the cache
            + cache_read / 1e6 * price["input"] * 0.10    # reading it back
            + int(usage["output_tokens"]) / 1e6 * price["output"])


def spec_for_id(model_id: str) -> ModelSpec | None:
    for s in _SPECS:
        if s.id == model_id:
            return s
    return None


def family_of(model_id: str) -> str:
    """The model behind a recorded id, for grouping in a report.

    An id this registry does not know is returned UNCHANGED rather than
    parsed. `deepseek/deepseek-v4-flash-0731` looks like it could be split
    on "/" and stripped of its date, and doing that would merge a pinned
    snapshot with a floating alias on the strength of a naming convention
    the vendor never promised. An unknown id stands alone; that is visible
    in the report and correctable by adding the spec.
    """
    spec = spec_for_id(model_id)
    return spec.model_family if spec else model_id


def model_id(llm: Any) -> str:
    """The model id a LangChain chat model was constructed with.

    `model` on ChatAnthropic, `model_name` on ChatOpenAI. Reading only one
    of them returns "" for the other provider, which lands every run of
    that provider on the unknown-model fallback price.
    """
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def build_model(alias: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                api_key: str | None = None) -> Any:
    if alias not in MODELS:
        raise SystemExit(f"unknown model {alias!r}; "
                         f"choose from {', '.join(sorted(MODELS))}")
    spec = MODELS[alias]

    key = api_key or os.environ.get(spec.env)
    if not key:
        raise SystemExit(f"{spec.env} is not set")

    if spec.legacy:
        print(f"WARNING: {spec.id} is a migration alias whose target can "
              f"change. Use a pinned id for anything you intend to report.",
              file=sys.stderr)

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # The key is wrapped rather than passed bare. `SecretStr` is what
        # both chat models actually declare, and it keeps the key out of
        # reprs and tracebacks — which matters in a harness that prints
        # its own configuration and ships traces into a run record.
        #
        # `max_tokens` and `model` still need silencing: these are pydantic
        # classes whose stubs describe the validated FIELDS, not the
        # aliases the constructor accepts. Silenced at the two call sites
        # with the reason, rather than by relaxing the type gate for the
        # whole package to accommodate one third-party stub.
        return ChatAnthropic(  # type: ignore[call-arg]
            model=spec.id, max_tokens=max_tokens,
            api_key=SecretStr(key), max_retries=MAX_HTTP_RETRIES)

    from langchain_openai import ChatOpenAI
    # No temperature set on purpose. DeepSeek V4 runs thinking mode by
    # default, where temperature and top_p have no effect — the request is
    # still accepted, so a tuned value silently does nothing. DeepSeek's own
    # Code Agent benchmarks used temperature=1.0, not the 0.0 that generic
    # coding guides recommend, and lowering it can collapse the reasoning
    # trace. Length is controlled with max_tokens instead.
    kwargs: dict[str, Any] = {}
    if spec.provider_pin:
        # Route pinning, with fallbacks OFF. Letting OpenRouter pick means
        # two runs of "the same model" can land on different weights and
        # different hardware, and nothing in the record would say so.
        #
        # allow_fallbacks=False makes an unavailable provider fail the
        # attempt rather than silently substitute another. The sweep
        # already classifies a failed attempt as a harness error that
        # produced no sample, which is the correct reading: a run that
        # never happened is not a result.
        kwargs["extra_body"] = {
            "provider": {"order": [spec.provider_pin],
                         "allow_fallbacks": False},
        }

    return ChatOpenAI(  # type: ignore[call-arg]
        model=spec.id, base_url=spec.base_url,
        api_key=SecretStr(key), max_tokens=max_tokens,
        max_retries=MAX_HTTP_RETRIES, **kwargs)
