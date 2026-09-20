# Findings

## Pilot closeout — September 20, 2026

The pilot ends with 122 loadable attempts across three dependency migrations
and 15 configurations. On the two measuring tasks, automated detectors
flagged 23 of 62 OpenRouter Flash patches: 37%, with a 95% Wilson interval
of 26%–50%. This describes recorded configurations; it is neither a model
ranking nor a human-confirmed cheating rate.

Across model IDs, counting only patches with a final test verdict gives
10/53 (19%), compared with 28/74 (38%) among all patches. Another 43
attempts produced no patch. Excluding runs that destroy test collection can
therefore hide behavior the detector is intended to measure. The complete
breakdown is generated in [RESULTS.md](../RESULTS.md).

The Qwen comparison stopped after three attempts against eight Flash
controls on each measuring task. All three hit a cost or iteration cap;
none solved cleanly. These pilots establish execution, not a ranking.

Both judges completed 48 pairs and the original annotator completed the
12-pair blind retest. Both failed the registered binary-kappa threshold:
Kimi 0.00 and MiMo −0.04. Their raw agreement alone would hide the effect of
class imbalance. Read the intervals and confusion matrices in the
[calibration report](judge-calibration.md); the labels are one annotator's
judgments, not ground truth.

The detector audit has a reproducible 38-patch sample but no reviews.
Precision and recall remain unknown. The contamination probe and fourth
task were not completed; further paid collection is out of scope.

## Historical engineering findings

The identifiers below are retained because code comments cite them.
These examples motivated harness changes; they are not controlled model
comparisons under the final configuration.

### 1. Missing measurements became plausible scores

Early failures included absent baselines becoming success, a stale test
report reused by another run, incorrect price defaults, and record-order
sampling. Strict record loading, configuration fingerprints and generated
reports now guard these cases. Rejected evidence remains in the
[archive manifest](../runs/archive/MANIFEST.md).

### 2. Harness choices affect results

Exploratory trials changed substantially after adding installed-package
inspection and changing the prompt. Capability claims must specify the
harness and keep differing configurations separate.

### 3. Agents can remove the evidence used to score them

Recorded patches deleted tests or prevented collection. The detector runs
even when there is no final test verdict; progress still requires one.

### 4. Detector rules can have false positives

An early rule flagged legitimate `select([col])` → `select(col)` migration
edits. Reference-patch controls catch some mistakes, but do not establish
precision on real runs. Patch-equivalence judge calibration is a separate
measurement from detector validation.

### 5. Repeated attempts vary

Historical attempts under the same settings produced credible progress of
0%, 0%, 44% and 93%. The report shows distributions and small-sample
uncertainty rather than presenting a single attempt as typical.
