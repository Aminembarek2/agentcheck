# Reproducing the calibration

The pilot calibration is complete: 48 original human labels, both judges on
all 48 pairs, and a 12-pair blind retest. Both judges failed the registered
headline threshold. No new API calls or labels are needed to reproduce it.

```bash
.venv/bin/python scripts/validate_judge.py status judge-labels.json
.venv/bin/python scripts/validate_judge.py report judge-labels.json --out docs/judge-calibration.md
.venv/bin/python scripts/report.py
.venv/bin/python scripts/report.py --check
```

Preserve the saved human labels, retest, raw judge replies and
[judge protocol](judge-protocol.md). The fresh Git history does not verify
their original ordering; historical blob identifiers remain metadata only.
Later adjudication belongs in a separate analysis. The retest used the original annotator, a seven-day
minimum interval, hidden first labels and re-randomized patch order.

Kimi's completed run used a 32,768-token output allowance. MiMo's 8,192-token
allowance is inferred from the resume guard, not recorded in its replies.
The protocol did not fix these allowances, so judge comparisons vary both
model and output allowance. Retain failed and unparseable replies.

Stored completed-run cost estimates are $23.50 for Kimi and $2.31 for MiMo;
they are not invoices. Interrupted calls and diagnostics are retained under
ignored `.local/judge-diagnostics/`, outside the registered measurement.

The detector audit is separate and **unreviewed**. Its frozen sample contains
all 23 flagged patches and 15 of 39 unflagged patches. Verify it with:

```bash
.venv/bin/python scripts/audit_detector.py --verify runs/audits/detector-sample.json
```

A future review must use the recorded sample and rubric, identify its
reviewer, and account for sampling weights and uncertainty. AI reviews must
not be presented as human validation. The pilot makes no precision or
recall claim from this sample.
