# Action Editor Requirements and Acceptance Criteria

## Editor decision

The Action Editor conditionally accepted the paper subject to specific revisions within two months and stated that, if all requested revisions are made, the next decision will be final acceptance.  The exact due date must be confirmed in the TACL system; this repository uses an internal deadline one week before the official deadline.

## Mandatory revision E1 — consolidate methods

**Editor request:** Consolidate the fragmented methods section for clarity.

**Required evidence:**
- One unified Method section before experiments.
- Trajectory extraction, metric definition, signature construction, GeoVote, CrossSteer, controls, and statistical protocol each introduced once.
- No experimental section redefines the main method.

**Acceptance artifact:** revised manuscript structure and a response-letter mapping.

## Mandatory revision E2 — GeoVote length bias and statistics

**Editor request:** Run controlled experiments that disentangle GeoVote from length bias and report statistical significance for all comparisons.

**Required evidence:**
- Same candidate pool for every voting method.
- Majority, token-count, length-weighted, log-probability, text-only, no-trajectory-length, residualized-geometry, and length-matched controls.
- Frozen calibration/validation/locked-test split at question level.
- Paired bootstrap confidence intervals and exact paired tests for every primary comparison.
- An explicit conclusion if geometry has no independent benefit beyond length.

**Acceptance artifact:** GeoVote control table, statistical appendix, source data manifest, revised claim language.

## Mandatory revision E3 — CrossSteer baselines, OOD, long context

**Editor request:** Benchmark CrossSteer against recent steering baselines (CAA, ActAdd, sparse activation methods), and expand OOD generalization and long-context stability analysis.

**Required evidence:**
- Fair protocol with no steering, matched-norm random, negative-direction, CAA, ActAdd, and sparse activation steering baselines.
- Same source-label budget, layer/strength validation budget, model, prompt, evaluator, and locked test for all methods.
- At least MATH-500 plus at least one additional fresh math OOD task; all OOD hyperparameters frozen from GSM8K validation.
- 512/4096/32768-token stability report and mitigation tests for repetition loops.
- Target-label-efficiency analysis that clarifies the practical use case relative to target-calibrated steering.
- Text/length behavior analysis to test whether the intervention is merely a generation-style or length shift.

**Acceptance artifact:** main CrossSteer comparison, OOD table, long-context stability figure, behavior-control appendix, response mapping.

## Mandatory revision E4 — readability

**Editor request:** Improve readability with intuitive spatial analogies, pseudocode, and streamlined notation.

**Required evidence:**
- One trajectory/geometry overview figure.
- Curvature-angle visual explanation.
- GeoVote and CrossSteer pseudocode.
- Symbol table; all variables defined at first use.
- Expanded method names at first mention.
- Rewrite dense prose in direct, declarative language.

**Acceptance artifact:** revised manuscript and reviewer response with page/line references.
