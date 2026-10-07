# Formal Steering Baseline Protocol (Draft, Frozen Before Stage 2)

This document resolves reviewer concern **E3a / R-B3 / R-C**: the submitted
CrossSteer comparison omitted standard activation-steering baselines.  The
implementation is deliberately separated from the old exploratory scripts.

## Shared constraints for every method

- Same target model, prompt template, tokenizer, generation configuration,
  layer grid, validation protocol, and locked test IDs.
- No vector may use validation, locked-test, or OOD labels.
- Every direction is normalized to RMS 1 at its selected layer before alpha
  selection, so raw vector scale cannot buy a larger perturbation budget. Vector
  norms are reported, and a matched-norm random direction is always included as
  a perturbation control.
- All methods use the revision hook's `position_mode=decode_last`, which skips
  prompt prefill and only injects on decode positions; the historical all-position
  hook is not formal evidence.
- The winning `(layer, alpha, schedule)` is selected once on validation and
  frozen before locked testing.  Each method receives the same selection
  budget.

## Methods

| Method | Direction source | What it tests | Revision implementation status |
|---|---|---|---|
| CrossSteer | Source model's correct-minus-incorrect **trajectory means** from source-training IDs only. | Whether a full trajectory direction transfers across checkpoints. | Implemented with source-ID audit. |
| Negative direction | Exact sign reversal of the tracked CrossSteer source direction, with no new labels or fitting. | Whether the claimed correct-minus-incorrect orientation itself is meaningful rather than any arbitrary sign/perturbation. | Implemented as a fingerprinted sign-falsification control; it receives the identical layer/alpha grid. |
| Matched-norm random | One fixed, seeded (`11241`) isotropic Gaussian direction matched to the raw L2 norm and shape of the CrossSteer source vector; it inherits the same source IDs but uses no labels beyond the reference artifact. | Whether an arbitrary perturbation at the same perturbation budget can explain an apparent CrossSteer effect. | Implemented as a fingerprinted registry and included in the full validation/locked family. No seed search or best-seed reporting. |
| SAE-space sparse activation | A target-model ReLU sparse autoencoder is trained without correctness labels on fixed relative-progress source-training states. Same-question correct/incorrect completions select a fixed sparse feature set, whose decoder vectors form the intervention. | A genuine sparse-representation activation baseline, rather than a coordinate mask relabelled as sparse activation steering. | Implemented as a target-local SAE-space baseline. It is not described as an exact reproduction of a third-party pretrained-SAE release. |
| CAA | Target model prompt-final activations after source-training completions. Multiple sampled completions of the same source-training question are paired only when one is correct and one is incorrect; among such candidates, the closest-length pair is selected, then averaged as positive-minus-negative residual states. | Standard multi-pair contrastive activation addition at a true prompt-final position. | Implemented as an audited registry; source IDs 0--99 only. |
| ActAdd | One deterministic correct/incorrect pair selected from that frozen, same-question contrast set, captured at the same prompt-final position. | Single-pair activation addition baseline. | Implemented as an audited registry; the selected pair is SHA-256 determined before validation. |
| Sparse steering | The top 10% coordinate-sparse version of the same CAA direction, with the same per-layer RMS normalization before alpha selection. | Whether a sparse coordinate intervention, rather than dense prompt-final contrast information, explains effects. | Implemented as a predeclared coordinate-sparse ablation. It is **not** called SAE-based Sparse Activation Steering. |
| Target-calibrated direction | Same trajectory construction as CrossSteer, but from target training IDs. | Label-efficiency and the reviewer question of why source transfer is needed. | Requires fresh target calibration generation. |

## What will *not* be claimed

1. A mean-difference direction from generated tokens is not called CAA.
2. A coordinate top-k ablation is not called SAE-based Sparse Activation
   Steering.
3. A favorable setting selected on the smoke run is not called validation or
   locked-test performance.
4. If CAA/ActAdd/sparse methods match or beat CrossSteer, the manuscript
   narrows its novelty to the diagnostic analysis rather than claiming a
   superior steering method.
5. The first sparse comparison is a 10% coordinate-sparse ablation, frozen
   before validation; it is not presented as an SAE reproduction.
6. The matched-norm random control is one predeclared deterministic seed, not a
   distributional claim and not a seed sweep from which the largest effect may be selected.
7. The SAE-space sparse activation baseline is a target-local, auditable baseline
   with frozen small-SAE construction settings. It is not mislabelled as a direct
   reproduction of a released SAE trained on a different model family.
