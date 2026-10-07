# R1 Official-Context Protocol Amendment

## Why the 512-token R1 control cell is not an admissible capability test

The shared 512-new-token steering grid is appropriate for the short-output Qwen
Instruct target, but it is not an official-like operating configuration for
`DeepSeek-R1-Distill-Qwen-1.5B`.  This was known before the current revision
run: the repository's historical length audit records 93 of 100 R1 GSM8K
trajectories hitting the 512-token budget, and the pre-existing locked model
configuration records an official-like 32,768-token, `temperature=0.6`,
`top_p=0.95` MATH-500 sanity evaluation.

The completed 512-token R1 calibration audit is therefore retained as a
**budget-saturation diagnostic**, not as evidence that the R1 model, its
post-training, or CrossSteer is ineligible.  It must not appear as a capability
or intervention failure in the manuscript.

## Amendment scope

This amendment is a protocol-validity correction based on pre-existing model
configuration evidence, not a change selected after reading intervention scores.
It changes only the R1 target's generation envelope:

| Field | Short-target protocol | R1 official-context protocol |
|---|---:|---:|
| Target | Qwen2.5-1.5B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B |
| `max_new_tokens` | 512 | 32768 |
| Decoding | greedy | fixed sampling: `temperature=0.6`, `top_p=0.95` |
| Sampling randomness | n/a | one deterministic problem-indexed seed schedule |
| Prompt, evaluator, source IDs | frozen | same frozen GSM8K protocol |
| Steering direction | full trajectories | exact generated-state mean via compact replay pooling |

No task, prompt, model, source IDs, validation IDs, locked IDs, layer, sign,
strength family, behavior guard, or evaluator is changed by this amendment.
The R1 seed schedule is fixed before any 32k validation result is inspected and
is shared by paired baseline/intervention generations for each problem.  GPU
serialization is a physical-resource gate only: the R1 runner waits until
physical GPU~1 is free, but it has no Qwen-chain prerequisite.  In particular,
a Qwen decoding/configuration failure cannot suppress this separately frozen R1
experiment.  The R1 chain exposes only physical GPU~1 (as logical
\texttt{cuda:0}); Qwen work may use physical GPU~0 separately once it has its
own valid frozen selection.

## Compact pooled-trajectory extraction

For CrossSteer, each source example already contributes only its generated-state
mean to the direction.  Saving all `[T, L, H]` states for an R1 response of up
to 32k tokens is unnecessary and unsafe.  The amended extractor therefore:

1. generates the exact completion without retaining all hidden states;
2. replays the prompt and generated token IDs with KV caching one token at a
   time; a parity diagnosis showed multi-token replay chunks deviate from the
   legacy `generate` states for this checkpoint;
3. accumulates only the per-layer sum and count of the same generated-state
   positions used by the historical extractor; and
4. saves the resulting `[L, H]` mean, token IDs, completion, generation
   telemetry, config and manifest.

A short-sequence parity test must show that the compact mean matches the legacy
full-trajectory mean with one-token KV replay (maximum absolute error at most
`1e-4`) before any R1 32k calibration is admitted.

## Observed readiness result and stop decision (2026-08-04)

The fixed 20-problem official-context readiness pass completed with a 10%
budget-hit rate (2/20; within the 25% ceiling) but a 10% severe repetition
rate (2/20), exceeding the preregistered 5% ceiling.  The two looped outputs
reached the 32,768-token cap and contained repeated answer-like text.  This
correctness-blind gate therefore stopped the R1 branch before 100-problem
calibration, direction fitting, validation, locked evaluation, or 4,096/32,768
schedule mitigation.  No prompt, seed, layer, alpha, sign, schedule or decoder
sweep is permitted to rescue this branch.

## Gates and claim boundary

1. Run a 20-problem R1 official-context capability sanity pass first.  Its only
   purpose is to verify completion, evaluator, token-budget and compact-replay
   parity; it is not used to select a method, layer, alpha or seed.  The frozen
   `revision_audit_pooled_readiness.py` gate requires exact IDs 0--19, one
   compact pooled completion per ID, no more than 25% budget hits, and no more
   than 5% severe whitespace 4-gram repetition loops (a repeated-4-gram
   fraction of at least 0.95).  It does not inspect correctness or compare a
   score.
2. If that pass is technically valid and no longer budget-saturated, run the
   frozen 100-problem source/target calibration and formal comparison matrix
   under this protocol.  The formal runner must use `--do-sample`,
   `--temperature 0.6`, `--top-p 0.95`, `--max-new-tokens 32768`, and the same
   problem-indexed seed (`seed * 100003 + sample_id * 1009`) for baseline and
   every intervention cell.  Its resolved manifest records those values; it
   may not reuse the short-target greedy manifest.
3. If it remains invalid or enters unrecoverable repetition behavior, retain
   the failure and remove the historical R1 repair claim.  Do not revert to a
   selected short-budget score or substitute Qwen-Instruct as an R1 repair.
4. Any final R1 result is reported separately from Qwen because their decoding
   envelopes differ; the paper may not call the two budgets a single matched
   benchmark comparison.
