# Paper Revision Blueprint for TACL 11241

This is the writing plan for the conditional-acceptance revision. It is a
blueprint, not a license to preserve an unsupported result. Every numerical
claim must be replaced by the latest audited artifact before submission.

## 1. Revised central question

The paper should answer two separate questions rather than treating them as a
single causal claim:

1. **Diagnosis:** Do post-training variants exhibit reproducible differences in
   hidden-state trajectory statistics under a fixed extraction and evaluation
   protocol?
2. **Intervention:** When a correctness-associated direction is injected at
   inference time, under what target/source compatibility conditions does it
   change task accuracy rather than only output length or style?

The first claim is descriptive. The second is conditional and must be tested
against target-calibrated, CAA/ActAdd, logprob, length, random and sparse
controls.

## 2. Section-by-section rewrite

### Abstract

- Remove any absolute claim that GeoVote is a geometry-based confidence method
  unless fresh locked data show incremental utility over length and majority.
- State the final CrossSteer operating regime and negative cases, not only the
  strongest same-target cell.
- Report paired uncertainty and data split in one sentence.
- If the new baselines match CrossSteer, describe the contribution as a
  controlled study of trajectory diagnostics and transfer boundaries, not a
  superior steering algorithm.

### Introduction

- Replace the current single pipeline story with the diagnosis/intervention
  split above.
- Define “signature” as a vector/matrix of measured associations, not a causal
  mechanism or model fingerprint by default.
- Add one paragraph explicitly saying what the paper does not establish:
  geometry is not automatically independent of length, position, confidence,
  text or answer frequency.
- Move the strongest contribution sentence until after the reviewer-requested
  controls are summarized.

### Related work

Organize by problem rather than by method name:

1. hidden-state correctness/trajectory diagnostics;
2. best-of-N, majority and logprob verifiers;
3. activation steering, CAA/ActAdd and sparse steering;
4. long-context intervention stability and behavioral side effects.

For each group, state the exact gap tested here: common architecture, dual
math/code evaluation where available, source-to-target transfer, and paired
length/style controls.

### Framework / Methods

Consolidate all definitions into one section in this order:

1. trajectory extraction and prompt/template;
2. correctness labels and candidate-pool construction;
3. geometric metrics and signed orientation;
4. GeoVote algorithm and its length-controlled variants;
5. CrossSteer source direction, target calibration and injection schedule;
6. CAA/ActAdd/sparse baselines;
7. train/validation/locked-test split;
8. statistical tests and artifact provenance.

Add pseudocode for GeoVote and CrossSteer. Define every symbol once; do not
reuse `m` for both a metric and a model, and replace ambiguous `y_i` with
`answer_i` / `correct_i` where possible.

### Diagnostic results

Report:

- per-model accuracy and generation-length distributions;
- signature matrices and bootstrap/rank stability;
- correctness association with confidence, length and text-only controls;
- model-role differences separately from correctness prediction;
- a clear distinction between “different checkpoint statistics” and
  “post-training-specific mechanism”.

Any nearest-neighbor topology that is unstable across scale or bootstrap must
be moved to an exploratory appendix and described as scale-sensitive.

### GeoVote results

Use a decision gate:

- If residualized GeoVote beats majority/logprob on fresh locked data with
  paired uncertainty, retain it as a secondary utility result.
- Otherwise, report the length correlation, controlled failure and demote
  GeoVote to a diagnostic/negative control. Do not scan extra metrics or layers
  after seeing locked labels.

The existing retrospective audit already indicates the second branch is
plausible and must be treated as the default writing plan.

### CrossSteer results

Use one table with identical columns for:

- no steering;
- CrossSteer source direction;
- target-calibrated direction;
- CAA;
- ActAdd;
- sparse direction;
- matched-norm random direction;
- logprob/length controls.

Every row must include layer, norm, schedule, alpha, source-label count,
validation-selection rule, test accuracy, delta, paired CI and exact p-value.

Then add:

- OOD math and code results;
- 512/4096/32768 stability;
- length/repetition/answer-marker behavior;
- repair/break examples;
- explicit failure boundary for incompatible source/target pairs.

### Discussion and limitations

Replace “geometry injects correctness” with the strongest supported language:

> The intervention changes hidden states along a direction associated with
> correctness in the calibration protocol; whether the direction represents a
> reasoning mechanism or a correlated behavioral variable is unresolved.

Explain why target calibration remains a strong baseline and why source transfer
is useful only when target labels are unavailable. State that a positive result
on one model family is not universal evidence.

## 3. Response-letter structure

For every reviewer point use four sentences:

1. acknowledge the concern;
2. state the exact new analysis or manuscript change;
3. report the result with split and uncertainty;
4. state the resulting claim boundary.

Do not answer a failed result by saying “we tuned another layer/alpha.” If a
requested control invalidates a submitted claim, explicitly say that the claim
was narrowed or removed.

## 4. Submission gate

Before the revised manuscript is finalized, verify:

- no submitted numerical result is copied into a revised main table without a
  provenance tag;
- no locked-test label was used for feature/layer/sign/alpha selection;
- every reviewer-requested baseline is either a faithful implementation or is
  explicitly marked unavailable with a technical reason;
- the conclusion matches the locked evidence, including negative results;
- the submitted V1 snapshot remains byte/source immutable in its separate path.
