# R1 Official-Context Formal-Comparison Protocol

This document predeclares the next R1 stage **before** any official-context
calibration or intervention score is inspected.  It supplements, rather than
replaces, `R1_OFFICIAL_CONTEXT_PROTOCOL.md`.

## Scope

R1-Distill is the submitted paper's historical repair target.  It is not merged
with the Qwen-Instruct complete-family table because their decoding envelopes
differ.  The Qwen table supplies the primary fair CAA/ActAdd/SAE/sparse
comparison requested by the editor.  An R1 result can support the historical
repair premise only under this separate model-valid protocol.

## Non-negotiable preconditions

1. The 20-problem official-context readiness run must pass its correctness-blind
   representation, saturation and severe-repetition gate.
2. The 100-problem R1 calibration must pass the frozen structural calibration
   audit together with the fixed Qwen-Math source calibration.
3. The compact one-token replay parity artifact must be present.
4. The historical 512-token R1 pool/configuration is inadmissible and cannot be
   retried, mixed, or used as a fallback.

A failure at any precondition removes the historical R1 repair claim.  It does
not authorize a new prompt, maximum length, temperature, top-p, seed, layer,
alpha, sign, or source subset.

## Frozen decoder and paired randomness

Every R1 baseline/intervention completion uses:

```text
max_new_tokens = 32768
do_sample      = true
temperature    = 0.6
top_p          = 0.95
seed(problem)  = 11241 * 100003 + problem_id * 1009
```

For a same-question CAA/ActAdd contrast pool, the fixed seed is
`11241 * 100003 + problem_id * 1009 + sample_idx`.  The generator is locally
forked, so resuming, reordering, or another method cannot alter a completion.
The resolved manifest records the decoder and each completion seed.

## Direction and baseline construction

- **CrossSteer source:** Qwen-Math correct-minus-incorrect generated-state
  trajectory mean on GSM8K IDs 0--99.
- **Target-calibrated:** R1 correct-minus-incorrect compact generated-state mean
  on the same IDs.
- **Negative direction:** the exact sign reversal of the source direction.
- **Matched-norm random:** one fixed seed and an RMS-matched random direction.
- **CAA / ActAdd / coordinate-sparse CAA:** R1 prompt-final states replayed from
  the frozen official-context K=4 same-question completion pool in
  `revision_gsm8k_r1_official32k_caa_source_k4.yaml`.

The target-local SAE baseline is part of the primary Qwen complete-family
comparison.  It is not silently reused for R1: the current SAE implementation
requires time-indexed trajectories, while the R1 protocol deliberately retains
only exact pooled means to make 32k extraction feasible.  Any R1 SAE extension
would require a separately frozen representation and implementation audit; it
cannot be added after seeing R1 outcomes.

## Evaluation and claim boundary

If the preconditions pass, the R1 validation grid has the same predeclared
relative layers (8, 14, 20, 24), RMS alphas (0.025, 0.05, 0.10, 0.20),
`decode_last` hook, behavior guard, and disjoint GSM8K role IDs as the Qwen
comparison, but uses the R1 decoder above.  Baseline and all interventions share
the problem seed.  Every method receives the same grid; no positive cell is
carried from the old 512-token sweep.

Any R1 locked/OOD/long-context conclusion requires its own completed paired
report, with accuracy, paired interval, exact test, repair/break counts, token
length, repetition, truncation and stop telemetry.  A Qwen advantage cannot be
reported as R1 evidence, and an R1 result cannot be called a universal
cross-model steering result.

## Decoder-preservation implementation guard

The formal R1 path uses the same immutable selection-to-locked-launch interface
as the short-output target, but it does **not** inherit its 512-token greedy
default.  The locked specification reads `max_new_tokens`, `do_sample`,
`temperature`, `top_p`, and the base seed from the completed R1 validation
manifest and writes them into the locked-launch manifest.  A sampled validation
artifact without all four decoder/seed fields is rejected.  The cross-method
selector also compares normalized generation metadata before it compares a
method's validation result; legacy Qwen greedy artifacts are canonically
backfilled only to the declared greedy defaults and retain their original
hashes.

This guard is protocol enforcement, not a source of additional selection: it
cannot modify a layer, alpha, source subset, prompt, model, seed, or decoder
once validation generation has begun.
