"""Tests for the pairwise judge, using a scripted fake.

No API calls, no cost, deterministic. The judge is the least trustworthy
component in the harness — it is a language model grading a language model
— so its failure modes are tested harder than its successes.

    pytest tests/test_judge.py -v
"""

import json

import pytest

from agentcheck.judge import (
    _MIRROR,
    AGENT_VERDICTS,
    NEUTRAL_VERDICTS,
    JudgeCall,
    aggregate,
    judge_patches,
    parse_reply,
    to_agent_relative,
)


@pytest.mark.parametrize("normalized", [True, False])
def test_empty_reply_keeps_generation_diagnostics(normalized):
    from types import SimpleNamespace

    class Limited:
        def invoke(self, messages):
            usage = {"input_tokens": 20, "output_tokens": 8192}
            meta = {"finish_reason": "length", "id": "gen-test"}
            if normalized:
                usage["output_token_details"] = {"reasoning": 8192}
            else:
                meta["token_usage"] = {
                    "completion_tokens_details": {"reasoning_tokens": 8192}}
            return SimpleNamespace(content="", usage_metadata=usage,
                                   response_metadata=meta)

    result = judge_patches(Limited(), "agent patch", "reference patch",
                           repeats=1, model_name="moonshotai/kimi-k2.6")
    assert result.n_unparseable == 2 and result.cost_known
    assert all(c["finish_reason"] == "length" and
               c["reasoning_tokens"] == 8192 and
               c["generation_id"] == "gen-test" for c in result.raw)

AGENT = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
REFERENCE = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+c\n"


class FakeJudge:
    """Replays scripted replies, and records what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        reply = self.replies.pop(0) if self.replies else self.replies
        class R:
            content = reply
        return R()


def reply(verdict, reasoning="because", confidence="high"):
    return json.dumps({"reasoning": reasoning, "verdict": verdict,
                       "confidence": confidence})


class Positional:
    """A judge that always prefers whichever patch it reads first.

    The exact artifact position swapping exists to detect: it is perfectly
    self-consistent within each order, so repeats alone would call it
    stable.
    """
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        class R:
            content = reply("b_wrong")      # "the second one is wrong"
        return R()


# --- the mirror -------------------------------------------------------------

def test_the_mirror_covers_every_verdict():
    assert set(_MIRROR) == NEUTRAL_VERDICTS
    assert set(_MIRROR.values()) == NEUTRAL_VERDICTS


def test_mirroring_twice_returns_the_original():
    """Two swaps put the patches back where they started, so the verdict
    must too. An asymmetric mapping would corrupt every reversed call."""
    for verdict in NEUTRAL_VERDICTS:
        assert _MIRROR[_MIRROR[verdict]] == verdict


def test_symmetric_verdicts_are_their_own_mirror():
    for verdict in ("equivalent", "different", "unclear"):
        assert _MIRROR[verdict] == verdict


@pytest.mark.parametrize("neutral,at_a,at_b", [
    ("a_narrower", "agent_narrower", "agent_wider"),
    ("b_narrower", "agent_wider", "agent_narrower"),
    ("a_wrong", "agent_wrong", "reference_wrong"),
    ("b_wrong", "reference_wrong", "agent_wrong"),
    ("equivalent", "equivalent", "equivalent"),
    ("different", "agent_different", "agent_different"),
    ("unclear", "unclear", "unclear"),
])
def test_agent_relative_translation(neutral, at_a, at_b):
    assert to_agent_relative(neutral, "a") == at_a
    assert to_agent_relative(neutral, "b") == at_b


def test_every_agent_relative_verdict_is_in_the_declared_vocabulary():
    produced = {to_agent_relative(n, p)
                for n in NEUTRAL_VERDICTS for p in ("a", "b")}
    assert produced <= set(AGENT_VERDICTS)


# --- parsing ----------------------------------------------------------------

def test_plain_json_parses():
    assert parse_reply(reply("equivalent"))[0] == "equivalent"


def test_fenced_json_parses():
    """Models fence their JSON regardless of instructions."""
    assert parse_reply(f"```json\n{reply('a_narrower')}\n```")[0] == "a_narrower"


def test_prose_wrapped_json_parses():
    text = f"Here is my analysis.\n\n{reply('different')}\n\nHope that helps!"
    assert parse_reply(text)[0] == "different"


def test_reasoning_and_confidence_are_kept():
    _, reasoning, confidence = parse_reply(
        reply("equivalent", "A migrates fetch_val; B does the same.", "medium"))
    assert "fetch_val" in reasoning
    assert confidence == "medium"


def test_an_invented_verdict_is_rejected_not_mapped():
    """Mapping an unknown label onto the nearest known one is guessing on
    the judge's behalf."""
    with pytest.raises(ValueError, match="unknown verdict"):
        parse_reply(reply("mostly_fine"))


def test_a_reply_with_no_verdict_is_rejected():
    with pytest.raises(ValueError):
        parse_reply(json.dumps({"reasoning": "hmm"}))


@pytest.mark.parametrize("text", ["", "   ", "I cannot answer that.",
                                  "{not json at all"])
def test_unreadable_replies_raise(text):
    with pytest.raises(ValueError):
        parse_reply(text)


# --- aggregation ------------------------------------------------------------

def test_consistent_judgments_agree_across_orders():
    """The core of mirroring. "B is narrower" with the patches swapped is
    the SAME judgement as "A is narrower". Counting the raw strings would
    manufacture 0% agreement out of a perfectly consistent judge."""
    judge = FakeJudge([reply("a_narrower"), reply("b_narrower")] * 3)
    result = judge_patches(judge, AGENT, REFERENCE, repeats=3)
    assert result.verdict == "agent_narrower"
    assert result.agreement == 1.0
    assert not result.position_bias
    assert result.is_stable


def test_an_order_dependent_judge_is_flagged():
    """Self-consistent within each order and completely uninformative."""
    result = judge_patches(Positional(), AGENT, REFERENCE, repeats=3)
    assert result.position_bias
    assert not result.is_stable
    assert set(result.by_order.values()) == {"reference_wrong", "agent_wrong"}


def test_within_order_disagreement_lowers_agreement():
    judge = FakeJudge([
        reply("equivalent"), reply("equivalent"),
        reply("a_narrower"), reply("b_narrower"),
        reply("equivalent"), reply("equivalent"),
    ])
    result = judge_patches(judge, AGENT, REFERENCE, repeats=3)
    assert result.verdict == "equivalent"
    assert result.agreement == pytest.approx(4 / 6)
    assert not result.is_stable


def test_both_orders_are_actually_run():
    judge = FakeJudge([reply("equivalent")] * 4)
    judge_patches(judge, AGENT, REFERENCE, repeats=2)
    assert len(judge.prompts) == 4
    first_position = [p.index("+b") < p.index("+c") for p in judge.prompts]
    assert sorted(first_position) == [False, False, True, True]


def test_the_judge_is_never_told_which_patch_is_which():
    """If it knew, the position test would measure nothing."""
    judge = FakeJudge([reply("equivalent")] * 2)
    judge_patches(judge, AGENT, REFERENCE, repeats=1)
    for prompt in judge.prompts:
        assert "maintainer" not in prompt.lower()
        assert "agent" not in prompt.lower()
        assert "reference" not in prompt.lower()


# --- unparseable ------------------------------------------------------------

def test_unparseable_replies_are_flagged_not_coerced():
    """`unclear` means the judge looked and could not decide. Unparseable
    means we never got an answer. Folding the second into the first puts a
    harness failure into a substantive category."""
    judge = FakeJudge([reply("equivalent"), "I refuse.",
                       reply("equivalent"), reply("equivalent")])
    result = judge_patches(judge, AGENT, REFERENCE, repeats=2)
    assert result.n_unparseable == 1
    assert "unclear" not in result.distribution
    assert result.verdict == "equivalent"


def test_unparseable_replies_are_outside_the_agreement_denominator():
    """Otherwise a broken parser masquerades as an inconsistent judge."""
    judge = FakeJudge([reply("equivalent"), "garbage",
                       reply("equivalent"), reply("equivalent")])
    result = judge_patches(judge, AGENT, REFERENCE, repeats=2)
    assert result.agreement == 1.0        # 3 of 3 readable calls agreed
    assert not result.is_stable           # but one call was lost
    assert result.n_parsed == 3


def test_a_wholly_unparseable_comparison_says_so():
    judge = FakeJudge(["nope"] * 4)
    result = judge_patches(judge, AGENT, REFERENCE, repeats=2)
    assert result.verdict == "unparseable"
    assert not result.usable


def test_a_model_that_raises_does_not_kill_the_comparison():
    class Exploding:
        calls = 0
        def invoke(self, messages):
            Exploding.calls += 1
            if Exploding.calls == 1:
                raise RuntimeError("rate limited")
            class R:
                content = reply("equivalent")
            return R()
    result = judge_patches(Exploding(), AGENT, REFERENCE, repeats=2)
    assert result.n_unparseable == 1
    assert result.verdict == "equivalent"


# --- truncation -------------------------------------------------------------

def test_an_oversized_diff_is_truncated_VISIBLY():
    """Passing whole diffs overflows the context silently. Task 002's
    reference diff alone is ~1,800 lines. A judge shown half a patch is
    answering a different question, so the truncation is reported."""
    huge = "+x\n" * 50_000
    judge = FakeJudge([reply("equivalent")] * 2)
    result = judge_patches(judge, huge, REFERENCE, repeats=1,
                           max_diff_chars=1000)
    assert result.truncated and "agent" in result.truncated[0]
    assert "omitted by the harness" in judge.prompts[0]


def test_truncation_keeps_both_ends():
    huge = "HEAD\n" + "+x\n" * 50_000 + "TAIL\n"
    judge = FakeJudge([reply("equivalent")] * 2)
    judge_patches(judge, huge, REFERENCE, repeats=1, max_diff_chars=2000)
    assert "HEAD" in judge.prompts[0] and "TAIL" in judge.prompts[0]


def test_a_small_diff_is_not_truncated():
    judge = FakeJudge([reply("equivalent")] * 2)
    assert not judge_patches(judge, AGENT, REFERENCE, repeats=1).truncated


# --- degenerate input -------------------------------------------------------

def test_an_empty_agent_patch_is_not_sent_to_the_judge():
    """It is the answer already, and asking invites a hallucinated
    comparison of nothing against something."""
    judge = FakeJudge([reply("equivalent")] * 6)
    result = judge_patches(judge, "", REFERENCE, repeats=3)
    assert result.verdict == "agent_wrong"
    assert result.n_calls == 0
    assert judge.prompts == []


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        judge_patches(FakeJudge([]), AGENT, REFERENCE, repeats=0)


# --- ties -------------------------------------------------------------------

def test_a_tie_resolves_deterministically():
    """The same calls must always give the same answer, whatever order the
    dict happened to be built in."""
    calls = [JudgeCall("agent_first", 0, "a_narrower", "a_narrower",
                       "agent_narrower"),
             JudgeCall("reference_first", 0, "equivalent", "equivalent",
                       "equivalent")]
    assert aggregate(calls).verdict == aggregate(list(reversed(calls))).verdict


def test_aggregate_of_nothing_is_unparseable_not_equivalent():
    assert aggregate([]).verdict == "unparseable"


# --- usage, cost and caching ------------------------------------------------

class BilledJudge:
    """Replies `equivalent` and reports usage, like the Anthropic client."""

    def __init__(self, usage=None, text=None):
        self.usage = usage
        self.text = text
        self.bodies = []

    def invoke(self, messages):
        self.bodies.append(messages[-1].content)
        usage, text = self.usage, self.text

        class R:
            content = text if text is not None else reply("equivalent")
            usage_metadata = usage
        return R()


USAGE = {"input_tokens": 12_000, "output_tokens": 300,
         "input_token_details": {"cache_read": 10_000, "cache_creation": 0}}


def test_every_call_records_its_tokens_and_its_cost():
    """judge-protocol.md §9 pre-registered per-call token counts.

    None were stored: a judge run could spend any amount and leave no
    record of it, while the protocol said otherwise.
    """
    result = judge_patches(BilledJudge(USAGE), AGENT, REFERENCE, repeats=2,
                           model_name="claude-haiku-4-5-20251001")
    assert result.n_calls == 4
    for call in result.raw:
        assert call["input_tokens"] == 12_000
        assert call["cache_read_tokens"] == 10_000
        assert call["cost_usd"] is not None and call["cost_usd"] > 0
    assert result.cost_known
    assert result.cost_usd == pytest.approx(
        sum(c["cost_usd"] for c in result.raw))


def test_a_cache_read_is_billed_at_a_tenth_not_at_full_price():
    """The whole reason for the breakpoint, checked against the price table."""
    from agentcheck.models import PRICES

    price = PRICES["claude-haiku-4-5-20251001"]
    result = judge_patches(BilledJudge(USAGE), AGENT, REFERENCE, repeats=1,
                           model_name="claude-haiku-4-5-20251001")
    expected = (2_000 / 1e6 * price["input"]
                + 10_000 / 1e6 * price["input"] * 0.10
                + 300 / 1e6 * price["output"])
    assert result.raw[0]["cost_usd"] == pytest.approx(expected)


def test_no_reported_usage_is_unknown_cost_not_free():
    result = judge_patches(BilledJudge(usage=None), AGENT, REFERENCE,
                           repeats=1)
    assert not result.cost_known
    assert all(c["cost_usd"] is None for c in result.raw)
    assert "usage not reported" in result.summary()


def test_a_reply_that_does_not_parse_is_still_billed():
    """The provider charges for the tokens whether or not we can read them."""
    result = judge_patches(BilledJudge(USAGE, text="not json at all"),
                           AGENT, REFERENCE, repeats=1,
                           model_name="claude-haiku-4-5-20251001")
    assert result.n_unparseable == 2
    assert result.cost_known and result.cost_usd > 0


def test_caching_marks_the_prompt_and_changes_nothing_the_model_reads():
    """A breakpoint changes the bill, never the tokens."""
    plain, cached = BilledJudge(USAGE), BilledJudge(USAGE)
    judge_patches(plain, AGENT, REFERENCE, repeats=1)
    judge_patches(cached, AGENT, REFERENCE, repeats=1, cache=True)

    assert all(isinstance(b, str) for b in plain.bodies)
    for text, block in zip(plain.bodies, cached.bodies, strict=True):
        assert block == [{"type": "text", "text": text,
                          "cache_control": {"type": "ephemeral"}}]
