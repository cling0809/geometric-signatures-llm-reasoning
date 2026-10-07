# Diagnostic Generalization Protocol V1

## Purpose

The submitted paper's first claim is valuable only if a signature is more than a
single heatmap produced on one slice of one model family.  This protocol asks a
strict, limited question:

> Under a fixed extractor and a fixed set of trajectory measurements, can a
> held-out signature be matched back to the post-training variant that produced
> it more often than chance, after controlling the most obvious generation
> confounds?

This is a **diagnostic reproducibility** study.  It does not claim that a
signature identifies an individual training operation, reveals a causal
mechanism, or is universal across all LLM architectures.

## Non-negotiable anti-selection rule

- Reuse the seven already-defined trajectory functionals and the existing
  length-residualized AUC signature; do not add a metric after inspecting a
  held-out result.
- Use all relative-depth positions in the comparison; do not select one layer,
  cell, sign, or visual region.
- Freeze task slices, prompt templates, token budgets, model revisions,
  evaluator versions, and split hashes before generation.
- Report every registered model/task/budget cell, including failed retrievals,
  unstable pairs, and capability-quarantined cells.
- No random seed, model subset, prompt, generation budget, or metric family is
  selected because it improves model identification.

## Input-eligibility gates (amended before interpreting any retrospective cell)

A retrieval number is eligible for a paper claim only when **both** gates pass:

1. **Auditable extraction provenance.** The source run must retain the resolved
   model/tokenizer revision, prompt or chat template, complete generation
   envelope (including all EOS token IDs), evaluator/grader version, generated
   labels and compact extracted trajectories.  A legacy directory containing
   only a short config plus aggregate labels/metrics is retrospective context,
   not confirmation evidence.  This gate was added after the revision uncovered
   an EOS-enveloping defect in a separate legacy Qwen steering workflow; no
   historical run is presumed comparable merely from its model name and token
   budget.
2. **Per-partition outcome support.** After common-ID alignment, every model
   must provide at least **10 correct and 10 incorrect** trajectories in **each**
   stable SHA-256 partition.  This floor applies before class matching and
   length residualization.  A cell below the floor is reported as
   *underpowered/unqualified*; its partitions may not be merged, resampled into
   existence, or used to support a positive retrieval claim.

The gates are provenance and sample-support requirements, not feature, layer,
model, prompt, seed, or result-selection rules.  They apply equally to positive
and negative cells and do not change the frozen signature construction below.

## What is tested

### Primary within-lineage test

Use matched-width Qwen-derived roles that have a verified, non-pathological
capability configuration for the task:

- Base;
- Instruct;
- Math-Instruct;
- Coder-Instruct;
- R1-Distill is capability-quarantined for any new model-valid diagnostic:
  its completed 32,768-token readiness audit had severe repetition in 2/20
  samples (10% > the frozen 5% ceiling).  Its historical short-envelope
  diagnostic remains descriptive only and cannot support a reproducibility or
  transfer claim.

The experiment creates independent signature estimates from disjoint problem
splits.  A signature estimated from one split is a *query*; signatures estimated
from another split form a *gallery*.  The test asks whether the nearest gallery
signature belongs to the same checkpoint, rather than whether a visually chosen
cell looks different.

### Cross-condition replication

The same fixed signature definition is then tested across:

1. two fixed extraction budgets (512 and 2048) on the common diagnostic
   envelope;
2. a math task and a code task, but only for model/task cells whose official or
   reference configuration passes the repository's capability-comparability
   check; and
3. one independent matched base/instruct lineage selected **before** loading its
   trajectories.  A cross-lineage result is a replication, not a calibration
   source for Qwen.

A role that is capability-quarantined on a task remains reported as a stress
cell; it cannot manufacture a high diagnostic score through a near-one-class
correctness distribution.

## Fixed signature construction

For every `(model, task, budget, split)` cell:

1. Generate with the resolved, archived configuration.
2. Separate correct from incorrect trajectories using the audited evaluator.
3. Regress the predeclared generation-length terms from each trajectory score.
4. Construct the existing `7 × relative-depth` signed AUC matrix using a fixed
   relative-depth interpolation rule.
5. In every bootstrap replicate, match correct/incorrect trajectory counts so a
   model's raw accuracy cannot create a larger signature merely through class
   imbalance.
6. Center by the no-signal value `0.5` before computing Frobenius distances.

No trajectory cell is used as a steering choice in this protocol.

## Data splits and tests

For each registered cell, divide problem IDs into three stable hash partitions:
`A`, `B`, and `C`.  The hash salt and IDs are recorded before extraction.

### Primary statistic: held-out model retrieval

For each pair of distinct partitions:

- build a query signature for every checkpoint on partition `A`;
- build a gallery signature for every checkpoint on partition `B`;
- retrieve the nearest centered signature by Frobenius distance;
- repeat for `A→C`, `B→A`, `B→C`, `C→A`, and `C→B`.

Report macro top-1 checkpoint retrieval, its bootstrap interval over problem
IDs, and a permutation p-value obtained by permuting checkpoint names in the
gallery.  A single successful model or pair does not qualify as a result.

### Stability statistic

For every task/budget condition, compare the full pairwise distance vector from
one partition/budget with the corresponding vector from another condition.
Report the rank correlation and its bootstrap interval.  This is descriptive:
it can support a stability statement only when the interval remains positive;
it cannot be replaced by a selected nearest-neighbor diagram.

## Claim gates

| Evidence outcome | Allowed wording |
|---|---|
| Within-lineage retrieval does not beat the permutation null | Do not claim a reproducible diagnostic signature; retain only the descriptive heatmap. |
| It beats the null for registered Qwen conditions, but cross-budget/task stability fails | “Post-training variants are distinguishable under the stated Qwen diagnostic protocol.” Do not use *universal*, *stable fingerprint*, or a model taxonomy claim. |
| It beats the null across registered Qwen splits and fixed budgets, with positive full-distance stability | “The diagnostic pattern is reproducible across the registered extraction conditions.” Still do not infer a causal training mechanism. |
| The same preregistered test succeeds in an independent matched lineage | “The diagnostic protocol replicates across the registered matched lineages.” This is the strongest permitted generalization claim; it is still not a claim about every LLM. |

For any positive diagnostic wording, all registered cells, paired distance
matrices, bootstrap intervals, permutation tests, extraction manifests, and
capability quarantines must be released together.

## Relationship to the steering revision

This study cannot rescue a failed intervention by definition.  It is deliberately
separate from the Qwen steering validation/locked pipeline:

- its model-retrieval result is never used to select a CrossSteer layer or
  strength;
- the active eight-method steering family remains the first GPU priority;
- no diagnostic result triggers a new steering alpha, prompt, seed, or method
  search;
- it begins only after the active steering control family has released the GPU,
  unless a separate non-contending resource is available.

## Decision

The purpose is to raise the evidential standard for the submitted paper's first
claim, not to change its wording after seeing a favorable plot.  If this protocol
fails, the paper must not call the phenomenon universal.  If it passes, the
paper may strengthen the first contribution using the exact claim gate above.
