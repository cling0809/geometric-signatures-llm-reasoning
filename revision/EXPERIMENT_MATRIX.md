# Frozen Revision Experiment Matrix

**Status:** Historical preregistered matrix.  All required downstream cells have now been dispositioned; the current evidence is recorded in `revision/evidence/`.

## Stage 0 — infrastructure and audit (CPU first)

| ID | Task | GPU | Gate | Reviewer mapping |
|---|---|---|---|---|
| S0.1 | Freeze submission tag/worktree; record software/model revisions. | No | Clean branch and manifests exist. | R-C2, R-C3 |
| S0.2 | Audit existing GeoVote candidates and existing steering outputs. | No | Every reuse artifact has IDs, labels, evaluator and config provenance. | E2, E3 |
| S0.3 | Implement unit tests for split isolation, direction construction, norm matching, paired statistics, repetition metrics. | No | Tests pass before GPU launch. | E2, E3, R-C2 |
| S0.4 | Build formal CAA/ActAdd/Sparse adapters from audited reference definitions. | No | Tiny deterministic smoke run agrees with expected vector shape/hook semantics. | E3a |

## Stage 1 — R1 official-context technical readiness

| Cell | Target | Decoder | Purpose | Forbidden use |
|---|---|---|---|---|
| GSM8K readiness | R1-Distill | 32,768 tokens; temperature 0.6; top-p 0.95; fixed problem-index seed | Validate completion, evaluator, exact compact replay, budget saturation and severe repetition. | No method, layer, alpha, sign, seed, score, or direction selection. |

**Hard gate:** the 20-problem readiness pass must be technically complete,
not budget-saturated, and free of severe loop failure before it releases the
100-problem R1 calibration.  The historical R1 512-token artifact is a retained
saturation diagnostic and cannot reopen the shared short-budget protocol.

## Stage 2 — formal in-domain baseline tests

| Cell | Dataset / IDs | Decoder | Direction/control family |
|---|---|---|---|
| A: Qwen-Instruct complete family | GSM8K 0–99 / 100–199 / 200–299 | greedy, 512 new tokens | no steering, negative, matched-norm random, CrossSteer, target-calibrated, CAA, ActAdd, SAE sparse activation, coordinate-sparse CAA |
| B: R1 official context | GSM8K 0–99 / 100–199 / 200–299 | 32,768; temperature 0.6; top-p 0.95; paired fixed seed | released only after Stage 1 and 100-problem structural audit; a separately frozen R1 comparison matrix is required |

**Outputs:** per-problem generations, correct flag, answer, token count, stop
reason, method config, paired statistics, and behavior fields.

**Randomness rule:** Cell A uses greedy decoding.  Cell B uses a single
problem-indexed seed shared by baseline and every intervention cell; the seed is
recorded in the resolved manifest and no seed is searched or selected.  Any
multi-seed auxiliary study reports its declared complete set rather than its best
seed.

**Capability gate:** a target whose calibration fails class-balance,
decoder-saturation, evaluator, or representation-integrity gates receives no
direction, locked, OOD, or long-context claim under that envelope.  Its evidence
is retained, it does not block an independent cell, and it never authorizes a
more favorable prompt, seed, layer, sign, or decoding fallback.

## Stage 3 — target-label efficiency

| Target-label budget | 0 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| CrossSteer orientation / target-calibrated baseline | run | run | run | run | run | run |

**Purpose:** distinguish 'no deployment-question labels' from 'no target calibration at all'.

## Stage 4 — frozen OOD test

| Dataset | Hyperparameters | Required result |
|---|---|---|
| MATH-500 (all 500) | Frozen from GSM8K validation | Full paired symbolic-grader table. |
| SVAMP (all 1,000) | Frozen from GSM8K validation | Full paired numerical-grader table. |

Both datasets and their evaluator audits are fixed before validation outcomes are read. No task, subset, prompt, layer, sign, alpha or schedule is chosen because it produces a favorable steering result.

## Stage 5 — long-context stability

| Budget | Method schedules |
|---:|---|
| 512 | constant steering control |
| 4096 | constant, decay, prefix-only, norm-adaptive |
| 32768 | constant, decay, prefix-only, norm-adaptive |

**Primary safety outputs:** accuracy, repetition-loop rate, truncation, stop reason, length, distinct-n.

## Stage 6 — behavior and length controls

- Paired steered/unsteered length distributions.
- Length-matched correctness comparison.
- Explicit long-minus-short direction control.
- Repair/break analysis.
- Text style / n-gram / answer-position / repetition analyses.

## Stage 7 — GeoVote controlled reanalysis

Use existing complete candidate pools where provenance passes audit.  Do not regenerate solely to search for a positive GeoVote cell.  Run same-pool voting controls and locked statistics as defined in `STATISTICAL_ANALYSIS_PLAN.md`.

## Stage 8 — diagnostic stability reanalysis

Bootstrap existing 1.5B/7B and slice-size signature matrices; revise taxonomy claims based on the result.  No claim of scale-invariant nearest-neighbor topology is allowed without direct evidence.
