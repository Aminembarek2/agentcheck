# Judge calibration protocol

**Recorded protocol.** This document specifies the sampling design, label
vocabulary, headline statistic and acceptance thresholds. The intended
workflow commits it before labelling; `scripts/validate_judge.py build`
requires a committed version and records its blob hash.

**Release provenance, September 20, 2026.** The public snapshot uses a fresh
Git history. Original labels and judge replies are retained, but historical
blob identifiers do not establish chronology and their objects may no longer
be present. This checkout does not independently verify preregistration.
The sampling rules and thresholds below are unchanged by the history reset.

---

## 1. What is being measured

Whether the judge in `agentcheck/judge.py` agrees with a human about the
same question: **does this agent's patch do the job the maintainer's merged
PR did?**

Not whether the judge is *right* — there is no oracle. Whether it tracks a
careful human reading the same two diffs under the same blinding. A judge
that does not track the human is not usable as a measurement instrument
whatever its verdicts look like.

## 2. Sample

- **48 pairs.** Above the 40 the project's scope calls for, divisible by the
  strata, and roughly three hours of labelling at three minutes a pair.
- **Stratified** on `(task_id, outcome class, cheated)`, where outcome
  class is one of `solved`, `partial`, `none`, `unscoreable`.
- **Floor of 2** per non-empty stratum, remainder allocated proportionally
  by largest remainder. Purely proportional allocation sweeps away the rare
  cells, and the rare cells — the cheating runs, the unscoreable ones — are
  the ones the project exists to measure.
- **Unscoreable runs are included.** A judge that confidently grades a
  patch from a run whose suite would not even collect is doing something
  wrong, and excluding those pairs would hide it.
- **The full sampling frame is recorded**, chosen and unchosen alike.
  Selection bias that cannot be audited will be assumed.
- Seeded and reproducible: the same archive and seed give the same 48.

## 3. Vocabulary

The human labels in the **neutral A/B vocabulary the judge itself uses** —
`equivalent`, `a_narrower`, `b_narrower`, `a_wrong`, `b_wrong`,
`different`, `unclear` — with the agent's patch randomly assigned to slot A
or B per pair, and no indication of which patch is the maintainer's.

Labelling in agent-relative terms while the judge labels in neutral terms
would compare two different tasks. Knowing which patch is upstream's is the
single strongest anchor available and would make the human's labels
unusable as a reference.

`skip` is available and is **not a verdict**: it means the pair is not
worth judging (no patch, harness failure). Skipped pairs leave the sample
and are reported as a count.

## 4. Blinding

Hidden from the labelling view: test results, outcome, progress score,
which model produced the patch, and which patch is the maintainer's.
Knowing a patch reached 115 passed / 0 failed answers "did it work"; the
question here is "is it the same fix".

## 5. Headline statistic — chosen now

**Cohen's kappa on the collapsed binary**, with a bootstrap 95% interval.

The collapse:

| neutral outcome | class |
|---|---|
| `equivalent`, `agent_wider` | **acceptable** |
| `agent_narrower`, `agent_different`, `agent_wrong`, `reference_wrong` | **deficient** |
| `unclear`, `unparseable` | *excluded, reported as a rate* |

Two reasons for collapsing. First, it is the decision the judge is actually
used to make. Second, at 48 items spread over seven labels almost every
disagreement is between two adjacent shades of "not quite the same fix",
and the coefficient would measure label granularity rather than judgement.

`agent_wider` sits with `equivalent` because doing more than the maintainer
did is not a deficiency: task 002's reference PR touches four backends the
test command never runs, so "wider" is frequently the more complete fix.

**Reported beside it, never instead of it:** the 7-class kappa, Gwet's AC1,
PABAK, raw agreement, modal-class prevalence, and the full confusion
matrix.

Why more than one coefficient, and why kappa still leads:

- Kappa is deflated by prevalence. On a sample that is 80% "acceptable" it
  understates a genuinely good judge.
- AC1 is not, which is why it is reported — but AC1 gives a judge that
  answers "acceptable" to everything about 0.76, "substantial" on the
  conventional bands, for a rater that has learned nothing. Kappa gives
  that rater 0.00, correctly. That case is pinned in
  `tests/test_stats.py`.

Neither is sufficient alone. Kappa leads because its failure mode is
conservative and AC1's is not. Where they disagree, the confusion matrix
decides what is said.

**Reporting whichever coefficient is highest is forbidden by this
document.** All of them are emitted by `report`, unconditionally.

## 6. Acceptance thresholds — fixed now

On the headline kappa:

| kappa | status | what it licenses |
|---|---|---|
| ≥ 0.60 | **quotable** | judge verdicts may appear as findings in `RESULTS.md`, with the interval attached and the judge model named |
| 0.40 – 0.60 | **screening-only** | usable to flag pairs for a human to read; never quotable as a per-run verdict |
| < 0.40 | **failed** | the judge is reported as having failed calibration; its verdicts appear in `RESULTS.md` only as that finding |

The interval is reported with the point estimate in every case. A kappa of
0.61 with a CI of [0.31, 0.84] is not "substantial agreement", it is a
measurement too imprecise to place, and the report says so.

These thresholds are encoded in `agentcheck.calibration.THRESHOLDS` so the
report cannot pick a band after the fact.

## 7. The human ceiling

A single annotator cannot produce inter-rater agreement. It can produce
**test-retest reliability**: 12 of the labelled pairs are re-labelled at
least 7 days later, with the slots re-randomised and the first labels
hidden, and intra-rater kappa is reported.

This is weaker than RoadmapBench's 0.83 in two specific ways, and the
report states both: one person's blind spots are shared across both passes,
so a consistent misunderstanding raises the number rather than lowering it;
and seven days is short enough that some recall survives. It is not
presented as equivalent to an inter-annotator figure.

Its purpose is a ceiling. Judge–human agreement cannot meaningfully exceed
the human's agreement with themselves, and a judge kappa of 0.65 against a
human whose test-retest kappa is 0.68 is a different result from the same
0.65 against a human at 0.95.

## 8. The annotator's own position bias

The judge's position bias is measured directly, by running every comparison
in both orders. The human cannot be asked the same question twice without
remembering, so it is measured through the randomisation: if the annotator
has no position preference, the **agent-relative** verdict distribution must
be the same whether the agent appeared as patch A or patch B. The
difference between the two arms is reported with a Newcombe interval.

A calibration that measures the judge's bias while assuming the human has
none is half a measurement, and it is the half that flatters the person
doing the measuring.

## 9. Judge configuration

- `--repeats 5`, each position-swapped, so 10 calls per pair.
- **The judge model must differ from the model that produced the patch.**
  `run` refuses otherwise unless `--allow-self-judging` is passed, and the
  report then flags every affected pair. Self-preference — a model rating
  its own output higher — is well documented and would be
  indistinguishable here from a calibrated judge.
- The same 48 pairs are judged by **two models**: `kimi` and `mimo`, through
  OpenRouter. *Amended — see §11, A1. Registered originally as `sonnet` and
  `haiku`.* Judge-model sensitivity is a result in its own right: if the
  two judges disagree with each other more than either disagrees with the
  human, that is the finding.
- Per-call detail is stored — both orderings, raw reply, parse status,
  token counts. Never only the aggregate.

## 10. What would invalidate this calibration

Stated in advance, so it is not negotiated later:

- Any pair labelled after its judge verdict existed. `run` refuses on
  unlabelled pairs; the labels file is committed before `run`.
- A cheating stratum empty at build time. The calibration would then say
  nothing about the case the project exists to measure, and the report
  must say so rather than presenting a full calibration.
- Fewer than 40 pairs surviving `skip` exclusions.
- Test-retest kappa below the judge's kappa. That would mean the reference
  the judge is being measured against is noisier than the judge, and no
  conclusion about the judge survives it.

## 11. Amendments

The following entries record the project's amendments. Their historical
blob identifiers name content versions, not independently verified dates.
The fresh release does not retain earlier Git objects or commit chronology;
the report preserves the identifiers recorded with each run.

### A1 — 2026-09-13: the judge models

**Changed.** The two judges in §9, from `sonnet` and `haiku` (the Anthropic
API) to `kimi` (`moonshotai/kimi-k2.6`) and `mimo` (`xiaomi/mimo-v2.5-pro`),
both through OpenRouter.

**Why.** Cost and access. Every run in this project goes through OpenRouter,
not the Anthropic API. Priced on the 48 labelled pairs at 10 calls each,
`sonnet` and `haiku` come to roughly $24–43 uncached; `kimi` and `mimo` to
roughly $6–10. The range is the unknown number of reasoning tokens each
model bills. On a budget of tens of euros for the whole project, that
difference decides whether the calibration and the sweep both fit.

**Recorded provenance.** MiMo verdicts record labels blob `f52e858e5ecd`;
the labels record protocol blob `ff79b1e058f6`. The historical amendment
states that no judge verdict existed at that point. With the original
commit history removed, that timing claim is not independently verifiable
from this release. A blob hash alone would not prove timing even if the
object were available.

**What it changes about the result.** The original pair was two sizes of one
vendor's model; this is two vendors. Disagreement between `kimi` and `mimo`
therefore mixes vendor with scale and cannot be attributed to either alone,
and the report says so rather than reading it as a pure scale effect.
Neither judge comes from the vendor whose model produced the patches
(DeepSeek), so self-preference is not in play for either. Both routes were
unverified pins when this was written and are checked with
`scripts/check_route.py` before the first call.

**In code.** `agentcheck.calibration.REGISTERED_JUDGES`, which `run` enforces.
The labels were written under the protocol at `ff79b1e058f6`; every judge
verdict records the protocol hash in force when it was produced.

**Erratum to A1, same day, before any verdict.** The costs quoted above were
computed from the model registry, whose prices for both judges were stale:
`kimi` was priced below the cheapest of its 21 OpenRouter endpoints, and
`mimo` at the cheapest of its 7. Re-priced at each model's dearest endpoint
(the registry's rule for a route it cannot pin), the pair estimates at
**$12.5–19.6**, not $6–10. `sonnet` and `haiku` at the same basis remain
$24–43. The choice of judges is unchanged — they are still well under half
the original cost — but the figure that justified it was wrong, and it is
corrected here rather than in place.


**Second erratum to A1 — 2026-09-17, after both judges completed.** The
corrected estimate above was still low. Both judges have now run the full
48 pairs at ten calls each, and the recorded cost is **$25.81**: `kimi`
$23.50 and `mimo` $2.31. Against the $12.5–19.6 estimate that is an
overrun of roughly 1.3–2×, and it lands outside the range rather than
inside it.

Three things account for it, and only the first was foreseeable:

- **Reasoning tokens are billed as output.** The A1 range was explicitly a
  guess at "the unknown number of reasoning tokens each model bills", and
  the guess was too small.
- **The two judges did not run under the same output allowance.** `mimo`
  used the 8,192-token default; `kimi` was rerun at 32,768 after a
  diagnostic found its replies exhausting the smaller limit. That is an
  operational choice made after the estimate, recorded in
  [the runbook](calibration-runbook.md), and it is the main reason `kimi`
  cost ten times `mimo` rather than two or three times.
- **`--max-cost` is per invocation, not per experiment.** `kimi` stopped
  itself at $20.32 after 41 pairs, as designed, and was resumed with a
  raised ceiling to finish the remaining 7. No ceiling was bypassed; the
  $20 default simply never was a budget for the whole run.

**What this does not change.** The judges, sample, repeats, blinding and
acceptance thresholds are untouched. The comparison against `sonnet` and
`haiku` at $24–43 still favours the registered pair, though by less than
A1 claimed. **What it does change** is any claim that this project can
forecast judge cost: it could not, twice, in the same direction. The
estimate is recorded here as wrong rather than quietly updated, because a
cost model that is only ever corrected after the spend is not a cost model.
