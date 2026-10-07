# TACL 11241 revision evidence

This directory contains compact, read-only evidence mirrored from the formal
server runs.  The submitted V1 paper directory is not modified.

- `locked-gsm8k-qwen-instruct-v3/`: corrected natural-decoding locked GSM8K
  complete-family report and triage decision.
- `ood-math500-qwen-instruct-v3/`: all 500 MATH-500 questions, all eight frozen
  methods, symbolic-grader audit provenance in the manifest.
- `ood-svamp-qwen-instruct-v3/`: all 1,000 SVAMP questions, all eight frozen
  methods, numeric-grader audit provenance in the manifest.
- `label-efficiency-qwen-instruct-v3/`: complete frozen 0/5/10/20/50/100
  target-label curve and family-wise statistics.
- `geovote-length-audit-n8-retrospective/`: same-pool GeoVote length audit,
  including the explicit 1,000-permutation adjacent-length-pair null.
- `per-problem-audit-v1/`: hash-verified, content-free problem-level correctness
  and behavior tables for locked GSM8K, MATH-500, SVAMP, and the target-label
  curve.  Prompt text, gold/predicted answers, completions, and seeds are excluded.

These artifacts are reporting inputs, not parameter-selection inputs.  No OOD
result is used to change a layer, strength, sign, schedule, seed, or subset.

- `long-context-qwen-instruct-v3/`: complete Qwen long-context audit at both
  4,096 and 32,768 budgets for source CrossSteer and target-calibrated
  directions, with all four frozen schedules and behavior telemetry.

The long-context reports are robustness evidence, not a post-hoc search space.
They do not support a source-transfer accuracy claim.

- `signature-stability-retrospective/`: problem-bootstrap nearest-neighbor
  stability audit for the reviewer-requested 1.5B/7B scale sensitivity check.
  It is explicitly retrospective and does not select intervention settings.
