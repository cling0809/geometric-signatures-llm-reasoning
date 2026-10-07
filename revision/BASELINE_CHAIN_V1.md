# Predeclared Steering-Baseline Chain V1

**Status:** Historical execution protocol.  The chain has completed; do not interpret the preconditions below as a current pending state.  Consult `revision/evidence/` for the final disposition.

This document freezes the automatic post-validation execution order. It exists
to ensure that CAA, ActAdd, and the coordinate-sparse ablation receive the same
validation budget as `crosssteer_source` and `target_calibrated`, without
inspecting their validation scores first.

## Preconditions

Both V1 grids must finish without `FAILURE_REASON`:

- `validation-grid-v1-crosssteer_source`;
- `validation-grid-v1-target_calibrated`.

If either fails, the chain stops. It does not silently alter a prompt, layer,
alpha, seed, schedule or max-token budget.

## Frozen sequence

1. Generate the K=4 source-training-only Qwen-Instruct completion pool on GSM8K
   IDs 0--99 using `revision_gsm8k_qwen_instruct_caa_source_k4.yaml`.
2. Build prompt-final CAA, ActAdd and 10% coordinate-sparse CAA vectors.
3. Evaluate CAA and ActAdd concurrently on validation IDs 100--199.
4. Evaluate sparse CAA on the same validation IDs after those two jobs finish.

Every method uses exactly:

- Qwen2.5-1.5B-Instruct target;
- greedy decoding, `max_new_tokens=512`;
- `decode_last`, absolute injection, constant schedule;
- RMS-normalized direction;
- layers 8/14/20/24 and alphas 0.025/0.05/0.10/0.20;
- the behavior guard and tie-break rule in `VALIDATION_GRID_V1.md`.

## Explicit non-actions

The chain does **not** read/choose a winning validation cell, run GSM8K IDs
200--299, run OOD, change method budget, or label the sparse coordinate ablation
as SAE steering. Selection and locked evaluation remain separate, fingerprinted
steps.
