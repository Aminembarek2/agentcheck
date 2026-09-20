#!/usr/bin/env python3
"""Check a model route with a small text request before running a sweep.

    .venv/bin/python scripts/check_route.py or-ds-flash
    .venv/bin/python scripts/check_route.py            # every routed model

A pinned provider whose slug is wrong does not degrade — it 404s, because
`allow_fallbacks: false` refuses to substitute a different backend. That is
the behaviour we want, and it is why the first sweep attempt died in three
minutes for $0.00 instead of producing four hours of runs served by an
unknown host.

This checks connectivity, not tool compatibility. A task pilot must verify
tool execution before a full sweep. The request allows 16 output tokens;
its cost depends on the model and the serving endpoint.

The reply's provider is READ BACK, not assumed. OpenRouter names the
serving endpoint in the response, so this verifies the pin held rather
than merely that the request did not fail — those are different facts, and
only one of them is what config_version claims.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcheck.models import MODELS, OPENROUTER_BASE_URL, build_model

#: A minimal text response is sufficient for this connectivity check.
PROBE = "Reply with the single word: ok"


#: Where OpenRouter's serving-provider name has been observed to land
#: once LangChain has wrapped the response. Searched in order rather than
#: indexed, because the depth moves between client versions and a hard
#: index would read "" on the next upgrade — silently, which is the
#: failure mode that matters here.
_PROVIDER_KEYS = ("provider", "provider_name", "openrouter_provider")


def _walk(value: object, depth: int = 0) -> str:
    """Depth-first search for a provider name in nested response metadata."""
    if depth > 4 or not isinstance(value, dict):
        return ""
    for key in _PROVIDER_KEYS:
        found = value.get(key)
        if isinstance(found, str) and found:
            return found
    for nested in value.values():
        found = _walk(nested, depth + 1)
        if found:
            return found
    return ""


#: OpenRouter's per-generation record. The serving provider is reported
#: HERE and not in the chat completion itself: LangChain's ChatOpenAI
#: parses the reply into its own schema and drops the top-level `provider`
#: field on the way through (`model_provider` in its metadata is
#: "openai", meaning the CLIENT type, not the endpoint that answered).
GENERATION_URL = "https://openrouter.ai/api/v1/generation"

#: The generation record is written asynchronously, so it can lag the
#: reply by a moment. A few short retries rather than one hopeful call.
_LOOKUP_ATTEMPTS = 4
_LOOKUP_DELAY = 1.5


def lookup_provider(generation_id: str, api_key: str
                    ) -> tuple[str, str]:
    """Ask OpenRouter which endpoint served a generation.

    This is what turns `provider_route` in `config_version` from a CLAIM
    about the pin into a MEASUREMENT of what ran. Those are different
    facts: a pin that silently stopped matching would leave every run
    stamped with a backend that did not do the work, and nothing in the
    record would contradict it.

    Returns (provider, reason-it-failed). Never a guess: an unreadable
    record yields an empty provider AND a reason, so the caller can say
    which of "could not ask" and "asked, got nothing" happened.
    """
    import json as _json
    import ssl
    import urllib.error
    import urllib.parse
    import urllib.request

    # macOS Python does not read the system keychain, so a bare urlopen
    # fails with CERTIFICATE_VERIFY_FAILED while the provider SDK in the
    # same process succeeds — it goes through httpx, which uses certifi.
    # Use the same trust store rather than the one urllib guesses at.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    request = urllib.request.Request(
        f"{GENERATION_URL}?id={urllib.parse.quote(generation_id)}",
        headers={"Authorization": f"Bearer {api_key}"})

    for attempt in range(_LOOKUP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=15,
                                        context=context) as reply:
                body = _json.loads(reply.read().decode())
        except urllib.error.HTTPError as e:
            # Generation records appear asynchronously. A recent id may
            # return 404 until it is indexed; other HTTP errors are final.
            if e.code == 404 and attempt < _LOOKUP_ATTEMPTS - 1:
                e.close()
                time.sleep(_LOOKUP_DELAY)
                continue
            return "", f"HTTP {e.code} from {GENERATION_URL}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return "", f"{type(e).__name__}: {e}"

        data = body.get("data") or {}
        provider = data.get("provider_name")
        if isinstance(provider, str) and provider:
            return provider, ""
        if attempt < _LOOKUP_ATTEMPTS - 1:
            time.sleep(_LOOKUP_DELAY)

    return "", (f"the record exists but names no provider_name; "
                f"keys present: {sorted(data)[:12]}")


def served_by(response: object) -> str:
    """The provider OpenRouter says answered, or "" if it did not say.

    An absent value is reported as unknown, never guessed at — this
    function exists to check a claim, so inventing an answer would defeat
    it entirely.
    """
    for attr in ("response_metadata", "additional_kwargs"):
        found = _walk(getattr(response, attr, None) or {})
        if found:
            return found
    return ""


def describe_metadata(response: object) -> str:
    """What the reply actually carried, for when the search comes up empty.

    Printed instead of shrugging: if the provider name has moved, the next
    run should say where to look rather than requiring a separate debugging
    session against a paid endpoint.
    """
    lines = []
    for attr in ("response_metadata", "additional_kwargs"):
        meta = getattr(response, attr, None)
        if isinstance(meta, dict) and meta:
            lines.append(f"    {attr}:")
            for key, value in sorted(meta.items()):
                rendered = str(value)
                if len(rendered) > 110:
                    rendered = rendered[:110] + "..."
                lines.append(f"      {key} = {rendered}")
    return "\n".join(lines) or "    (the reply carried no metadata at all)"


def check(alias: str) -> bool:
    spec = MODELS[alias]
    pin = spec.provider_pin or "(unpinned)"
    print(f"{alias:<14} {spec.id:<34} pin={pin}")

    try:
        model = build_model(alias, max_tokens=16)
        reply = model.invoke(PROBE)
    except Exception as e:
        # Broad on purpose: the point is to surface the provider's own
        # message, whatever shape the SDK wrapped it in.
        message = str(e)
        print(f"  FAILED: {type(e).__name__}")
        print(f"  {message[:400]}")
        if "No endpoints found" in message:
            print(f"\n  The model slug is fine — the routing funnel found "
                  f"endpoints and then\n  filtered them all out. That is the "
                  f"pin {spec.provider_pin!r} matching none of\n  them. Open "
                  f"https://openrouter.ai/{spec.id} and use the copy button\n"
                  f"  beside a provider name for its exact slug.")
        return False

    provider = served_by(reply)

    # Not in the reply — ask OpenRouter directly, using the generation id
    # it did return.
    if not provider:
        generation_id = (getattr(reply, "response_metadata", None)
                         or {}).get("id", "")
        key = os.environ.get(spec.env, "")
        why = "no generation id in the reply"
        if generation_id and key:
            provider, why = lookup_provider(str(generation_id), key)

    if not provider:
        # The call SUCCEEDED, which is itself most of the evidence: a pin
        # matching no endpoint returns 404 with `Filter by Fallback -> 0`,
        # so a 200 means the filter kept at least one. What is missing is
        # only which one. Report that honestly and show what came back, so
        # the field can be found rather than guessed at.
        if spec.provider_pin:
            print("  ROUTE OK — the call succeeded, so the pin matched at "
                  "least one endpoint\n  (a pin matching none returns 404). "
                  "But WHICH endpoint served it is\n  unconfirmed:")
        else:
            print("  ROUTE OK — the model answered. It is UNPINNED, so "
                  "OpenRouter chose the\n  endpoint and which one it chose "
                  "is unknown:")
        print(f"    lookup: {why}")
        print(describe_metadata(reply))
        if spec.provider_pin:
            print("\n  The pin is enforced either way — allow_fallbacks is "
                  "off, so a pin that\n  stopped matching would 404 rather "
                  "than substitute.")
        else:
            print("\n  Note `system_fingerprint` above if it is present: it "
                  "does not name the\n  provider, but two runs showing "
                  "different fingerprints were served by\n  different "
                  "backends, which is the part that threatens "
                  "comparability.")
        print("  docs/threats-to-validity.md records this limitation.")
        return True
    if spec.provider_pin and provider.lower() != spec.provider_pin.lower():
        # Not a failure of the call, but a failure of the CLAIM. The
        # config_version says which backend produced a run; if the pin does
        # not hold, that field is wrong and every run under it is mislabelled.
        print(f"  MISMATCH: pinned to {spec.provider_pin!r} but served by "
              f"{provider!r}.\n  config_version would record a backend that "
              f"did not run the work.")
        return False
    print(f"  OK — served by {provider}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("aliases", nargs="*",
                   help="model aliases; default is every routed model")
    args = p.parse_args()

    aliases = args.aliases or [a for a, s in MODELS.items()
                               if s.base_url == OPENROUTER_BASE_URL]
    unknown = [a for a in aliases if a not in MODELS]
    if unknown:
        print(f"unknown model(s) {unknown}; choose from "
              f"{', '.join(sorted(MODELS))}", file=sys.stderr)
        return 1

    ok = True
    for alias in aliases:
        ok &= check(alias)
        print()
    if not ok:
        print("Fix the failing pin(s) before running a sweep.",
              file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
