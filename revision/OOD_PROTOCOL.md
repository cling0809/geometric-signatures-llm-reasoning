# Frozen OOD Protocol V2: MATH-500 + SVAMP

This protocol addresses the TACL request to test whether a GSM8K-selected
CrossSteer configuration transfers beyond the in-domain split.  It forbids
selecting an OOD task, layer, alpha, schedule, sign, source subset, prompt, or
generation budget after inspecting an OOD steering outcome.

## Prerequisites

1. A method must first be eligible under the frozen GSM8K validation rule.
2. Its locked in-domain evaluation must be generated from the immutable
   selection artifact.
3. Both OOD grader audits must pass before any OOD generation:
   - MATH-500: `math-verify==0.9.0` symbolic equivalence on all 500 gold
     answers, with positive/negative symbolic fixtures;
   - SVAMP: the complete 1,000-item official JSON, fixed source hash
     `5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e`,
     numerical gold-label parsing and fixed positive/negative fixtures.

Failure of a prerequisite means **no OOD steering claim** for that method. It
is not a reason to substitute a different vector, method, or task.

## Frozen OOD datasets

| Dataset | Size | Evaluator | Purpose |
|---|---:|---|---|
| MATH-500 | 500 | `math-verify==0.9.0` symbolic equivalence over complete decoded text | Hard symbolic-math transfer |
| SVAMP | 1,000 | boxed-answer/last-number numeric comparison, tolerance `1e-4`, pinned official JSON hash | Fresh arithmetic-word-problem transfer |

The prompt, greedy decoder, selected maximum-token budget, vector, layer,
alpha, schedule, injection mode and RMS normalization are copied exactly from
the GSM8K selection.  OOD scripts expose no layer/alpha/sign/schedule sweep.

## Complete-family execution and reporting

For each target whose validation selection contains eligible methods, **every
eligible method** runs on all 500 MATH-500 items and all 1,000 SVAMP items.
Each run independently regenerates its unsteered baseline and retains full
completion, correctness, token count, stop reason, truncation, repetition,
answer-marker position, repair/break, paired bootstrap interval and exact test.

The report builder refuses a partial family, a missing question, a mismatched
baseline, an evaluator mismatch, a different frozen selection hash, or a
cross-dataset run that does not use the exact selected target specification.
MATH-500 and SVAMP are reported as separate paired tables; neither result is
pooled with GSM8K or used to retune the intervention.  A negative result remains
in the table.
