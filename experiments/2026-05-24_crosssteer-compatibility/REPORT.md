# CrossSteer compatibility diagnostic

- Date: 2026-05-24
- Status: completed as a zero-GPU re-analysis
- Script: `scripts/crosssteer_compatibility_analysis.py`
- Outputs: `results/2026-05-24_crosssteer_compatibility/`

## Purpose

The paper needs a stronger innovation statement than "a steering direction
improves one target."  The more defensible claim is that the geometric signature
also predicts when a source direction should help and when it should hurt.

## Diagnostic

For each model, take the `curvature_mean` signature row and choose the layer with
the largest distance from chance.  The sign of `AUC - 0.5` gives a binary
orientation:

- `AUC < 0.5`: higher curvature mean is associated with correct generations.
- `AUC > 0.5`: higher curvature mean is associated with wrong generations.

Qwen Base, Instruct, and Math share the first orientation.  R1-Distill has the
opposite orientation.

## Results

The orientation rule predicts the sign of all currently tested CrossSteer cells.

| Setting | Prediction | Observed delta |
|---|---|---:|
| Qwen Base -> R1, GSM8K | repair candidate | +10pp |
| Qwen Instruct -> R1, GSM8K | repair candidate | +13pp |
| Qwen Math -> R1, GSM8K | repair candidate | +15pp |
| Qwen Math/GSM8K -> R1, MATH-500 | repair candidate | +5pp |
| Qwen Instruct/MATH-500 -> R1, MATH-500 | repair candidate | +3pp |
| Qwen Math/MATH-500 -> R1, MATH-500 | repair candidate | +7pp |
| Qwen Math/GSM8K -> Qwen-Instruct, MATH-500 | overshoot risk | -19pp |

Summary: 7/7 sign matches.  On the GSM8K source-quality sweep, source accuracy
and CrossSteer delta are almost perfectly monotone (`r = 0.999`, `n = 3`), which
should be reported as descriptive rather than conclusive.

## Paper implication

This should be written as an applicability diagnostic, not as a universal theory.
The claim is:

> The signature is not just a fingerprint and not just a source of steering
> vectors.  It also gives a pre-intervention warning about compatible and
> over-aligned targets.

## Follow-up experiment

The clean next baseline is:

1. Calibrate both Qwen-Math and R1-Distill directions on GSM8K ids 0-99.
2. Evaluate both directions on R1-Distill ids 100-199.
3. Compare cross-model source direction against same-target direction.

The runner is `scripts/run_c4_holdout_baselines.sh`.  It uses
`scripts/c4_steer_r1.py --eval-start 100` to keep calibration and held-out
evaluation disjoint.
