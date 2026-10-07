# Frozen Long-Context Stability Protocol V1

**Status:** Design is frozen before any 4k/32k result inspection.  This is the
revision response to the reported long-generation repetition loops; it is not a
new search budget for rescuing a failed 512-token intervention.

## Entry condition

A method enters this study only if it has a behavior-eligible,
validation-selected configuration from `VALIDATION_GRID_V1.md`.  Its direction,
layer and alpha are fixed.  Eligibility is a safety gate rather than a
post-hoc requirement for a favorable score: the complete selected result is
reported.  No failed 512-token method can enter by changing its layer, sign or
alpha here.

## Common conditions

- Same target, prompt, decoder, evaluator, source-label provenance and locked
  problem IDs as the corresponding selected in-domain intervention.  The Qwen
  short-envelope cell reuses its selected 512-token greedy decoder; the R1
  official-context cell reuses its selected 32,768-token sampled decoder and
  paired problem-indexed seed rule.
- `decode_last` only: the prompt prefill is never modified.
- Every vector retains RMS=1 normalization before its frozen alpha is applied.
- Report accuracy, paired repair/break counts, output token count, repeated
  4-gram fraction, distinct 4-gram ratio, answer-marker position, truncation
  and stop reason.

## Schedules

For the Qwen short-envelope cell, report the selected 512-token constant
intervention as its short-budget reference.  For the R1 official-context cell,
the selected 32,768-token constant intervention is the model-valid reference.
At 4096 and 32768 maximum new tokens, compare the following **same-alpha**
policies for every entered cell:

1. `constant`: injection RMS is the fixed normalized-vector budget.
2. `prefix-256`: injection is applied only through the first 256 decode steps.
3. `exponential-1024`: strength is multiplied by `exp(-t/1024)`.
4. `relative-hidden-rms`: injection coordinate RMS is
   `alpha * schedule(t) * residual-coordinate-RMS` at the changed decode
   position.  This is a scale-adaptive safety policy, not a new direction.

The first three policies are deterministic time schedules.  The fourth only
rescales the already frozen direction by the target residual norm, making its
relative perturbation bounded across depth and long generation.  It cannot
change direction, layer, sign or source data.

## Interpretation gate

A long-context policy is a mitigation only if it reduces repetition/truncation
without reducing locked accuracy below the unsteered baseline by more than the
predeclared paired tolerance.  If all policies loop or degrade accuracy, the
paper reports this as a CrossSteer limitation rather than claiming long-context
robustness.

## Method entry set and complete schedule reporting

The long-context study is predeclared for the two trajectory-direction variants:
`crosssteer_source` and `target_calibrated`.  Each enters only when it is
validation-eligible; CAA/ActAdd/sparse remain matched 512-token baselines in the
main comparison rather than being selectively promoted to a long-context claim.
For every entered variant, both budgets and all four schedules are run and
reported.  The R1 implementation waits for its separate 32k frozen formal
comparison, then reuses its immutable sampled selection; it cannot inherit a
Qwen selection or return to the invalid 512-token R1 envelope.  If neither
variant is eligible, the chain writes an explicit no-eligible-method marker
instead of changing direction, layer, alpha, or method.
