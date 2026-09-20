# Threats to validity

This pilot supports observations about the recorded runs, not a general
model ranking. [RESULTS.md](../RESULTS.md) contains the generated evidence.

## Construct validity

**Detector flags are not confirmed cheating.** The scorer reads diffs using
syntactic rules. Legitimate changes can trigger false positives; indirect
shims or weakened fixtures can evade detection. Neither precision nor
recall on real patches has been measured. The flag rate is not a demonstrated
lower bound, and Wilson intervals do not account for detector error.

The 38-patch audit sample is frozen but unreviewed. Auditing all 23 flagged
patches would assess patch-level precision against reviewer judgments.
The 15 sampled unflagged patches need weight 39/15 when estimating missed
positives. Zero observed misses would not prove perfect recall.

**The maintainer's patch is a reference, not proof.** Alternative correct
fixes can differ from it. The scorer's allowances reduce false positives
but do not establish semantic correctness. Judge calibration measures
patch-equivalence judgments, not detector precision.

**Progress is a proxy.** Tests within a root cause often move together.
Coverage was recorded from the green baseline; new-version-only paths may
be incorrectly treated as unverifiable. “Clean” means no detector flags,
not independent human verification.

## Internal validity

**The harness has had measurement bugs.** Missing baselines, stale test
reports, price defaults and record-order sampling produced plausible but
wrong results. Strict schemas, regression tests and generated reports now
guard these cases; they do not prove the absence of other bugs. See
[findings](findings.md) and the [archive manifest](../runs/archive/MANIFEST.md).

A later denominator bug excluded patches without a test verdict from the
flag rate, hiding runs that destroyed collection. Detection now includes
all scored patches; progress still requires a valid verdict. Runs with no
patch are counted separately.

Earlier container cleanup could terminate another live harness process.
Affected runs are retained as superseded evidence and excluded from current
results. Cleanup now checks the owning process rather than container age.

**Serving providers are not pinned.** OpenRouter can route attempts within
one configuration to different endpoints. Records do not establish the
serving hardware or quantization. Historical direct-provider Flash runs
also differ from the later OpenRouter snapshot and cannot be pooled into
matched comparisons merely because both use Flash.

**Costs are registry estimates.** Stored direct-provider Flash costs used
stale rates; the recorded pricing analysis estimates understatements of
1.6–4.7×. OpenRouter Flash rates were not endpoint maxima. Qwen used a
conservative endpoint/context-tier table checked on September 17, 2026,
but that is not a guarantee about later prices or total spend. Implicit
caching, context tiers and the final call crossing an estimated cap add
uncertainty. Stored records are not repriced; the report labels all costs
as estimates, not invoices or future budgets.

Trace backends are optional and non-authoritative. Published numbers come
from saved run records, not remote tracing services.

## External validity

Related work: [ImpossibleBench](https://arxiv.org/abs/2510.20270) studies specification–test conflicts; [EvilGenie](https://arxiv.org/abs/2511.21654) evaluates reward hacking and its detection in programming tasks.
[METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/) documents agents manipulating tests and scoring code.
[SpecBench](https://arxiv.org/abs/2605.30314) evaluates specification review through historical RFC critiques, an adjacent problem rather than a cheating benchmark.
This pilot's narrower scope is pinned Python dependency migrations, reference-aware diff flags and judge calibration; it does not claim to discover reward hacking.

- Three Python dependency migrations, with only two measuring tasks. The
  one-root-cause docker-py task is a smoke test and is excluded from pooling.
- Nothing here establishes performance on feature work, other languages or
  different repositories, prompts, tools or scaffolding.
- Qwen has only three pilot attempts. No matched comparison reaches the
  required eight attempts per model and task.
- Historical i150 cells have five attempts each. They are descriptive;
  adding runs under a different configuration would not complete them.

## Contamination

The tasks use public upstream fixes. Two migrations date from 2023, and
migration guides describe the relevant API changes. Correct patches may
reflect recall, reasoning or both.

The optional `scripts/contamination_probe.py` was **not run**. Its file-name
recall prompts would provide a limited diagnostic, not proof of whether a
model had seen a patch: plausible file names can also be inferred, and
failure to recall does not establish absence of training exposure.

## Judge calibration

Both judges failed the specified binary-kappa threshold. Their patch
verdicts are not validated quality scores. The sample is strongly imbalanced;
raw agreement and AC1 can look high for largely deficient predictions.
Read the confusion matrices and uncertainty in the
[calibration report](judge-calibration.md).

There is one human annotator. The 12-pair blind retest measures intra-rater
consistency, not inter-rater agreement or correctness. Shared blind spots
and recall can inflate consistency. Judge disagreement does not by itself
prove that the judge is wrong. Different judge output allowances and
missing MiMo generation metadata further limit comparisons.

**Release provenance.** This repository was published with a fresh Git
history. Saved labels, replies and historical blob identifiers are retained,
but the original commit chronology is not. This checkout cannot independently
establish that the protocol or human labels predated judge output.

## Statistical validity

- Samples are small and outcomes often bimodal. The report shows intervals
  and individual values; eight attempts is a minimum target, not adequate
  power for every comparison.
- Tests within a task are correlated: 53 of task 002's 55 initial failures
  share one root cause. Root-cause progress is reported beside test progress.
- Bootstrap intervals at small n describe resampling of the observations,
  not an exhaustive range of future outcomes. Degenerate intervals are flagged.
- The headline rate pools recorded configurations descriptively. It is not
  a controlled estimate of model behavior or the effect of a budget change.
- Matched comparisons require equivalent recorded settings and use Newcombe
  intervals on rate differences. Provider variability can remain within them.
- Multiple comparisons have no multiplicity correction; isolated differences
  are exploratory rather than confirmatory findings.

## Scope at closeout

Further paid collection is out of scope. A fourth task, a completed model
comparison, contamination probes, an independent detector audit and another
annotator would strengthen the evidence, but are not claimed as completed.
