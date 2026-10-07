# Clean Held-out CrossSteer Baselines

## Purpose

This run addresses the main remaining steering baseline question: is the
cross-model Qwen-Math direction useful beyond a direction extracted from the
target family itself?

The design keeps calibration and evaluation separated. Source directions are
computed from existing source runs on ids 0--99. Target evaluation uses GSM8K ids
100--199, so target labels from the evaluation questions are not used to form the
direction.

## Runs

Default command:

```bash
bash scripts/run_c4_holdout_baselines.sh
```

The runner launches two jobs in parallel when two GPUs are available:

| run | source direction | target | device |
| --- | --- | --- | --- |
| `cross_source_qwen_math_to_r1` | Qwen-Math, GSM8K ids 0--99 | R1-Distill, GSM8K ids 100--199 | `cuda:0` |
| `same_target_r1_to_r1` | R1-Distill, GSM8K ids 0--99 | R1-Distill, GSM8K ids 100--199 | `cuda:1` |

Both use `layer=14`, `alphas={0,1,2}`, `max_new_tokens=512`, and greedy decoding.

## Interpretation

If the cross-model source matches or beats the same-target direction on held-out
ids, it strengthens the paper's cross-model-transfer claim. If same-target wins,
the result still helps the paper: CrossSteer should be framed as evidence that
signature-derived directions are causal, while cross-model transfer becomes a
useful but not dominant variant.

## Result

Server output directory:

```text
/root/AI/runs/2026-05-24_c4-holdout-baselines
```

Mirrored local output directory:

```text
results/server_2026-05-24_c4-holdout-baselines
```

| run | alpha=0 | alpha=1 | alpha=2 | best delta |
| --- | ---: | ---: | ---: | ---: |
| Qwen-Math -> R1-Distill | 0.33 | 0.43 | 0.43 | +10pp |
| R1-Distill -> R1-Distill | 0.33 | 0.38 | 0.53 | +20pp |

## Paper implication

The result is useful but changes the claim boundary. Cross-model transfer remains
positive on a clean held-out split, so the source direction is not just exploiting
the original 100 evaluation questions. However, a same-target direction is stronger
when it is available. The paper should therefore claim that signature-derived
directions are causal and transferable, not that cross-model transfer dominates
same-target activation addition.
