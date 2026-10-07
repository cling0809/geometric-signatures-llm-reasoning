# TACL 11241 Major Revision Control Center

**Decision:** Conditional acceptance / B decision.  
**Manuscript:** *Geometric Signatures of Post-Training in LLM Reasoning Trajectories: Separating Observability from Transferability*
**Immutable submitted baseline:** Git commit `3aa041b` (`2026-06-01`).  
**Revision branch:** `codex/tacl-11241-major-revision`.

## Current submission state

The figure rebuild is complete.  The author-supplied Fig. 1 overview and the
audited quantitative figures are integrated into the current 17-page manuscript
(Fig. 1 print-prepared on 2026-08-19).
The manuscript, response letter, tests, and anonymous-release preflight have been
re-run against this final figure set.  Any anonymous archive generated before
this state must be rebuilt before upload.

## Objective

Satisfy every mandatory Action Editor revision without reopening open-ended geometry metric searches.  All new empirical work must map to a reviewer comment, use a predeclared protocol, and preserve a locked final test.

## Non-negotiable principles

1. Do not modify or overwrite the submitted artifact or its source snapshot.
2. Do not reuse post-submission exploratory results as formal evidence until they pass the revision audit.
3. Do not select a metric, layer, sign, alpha, or task after looking at its locked-test labels.
4. If GeoVote has no incremental value beyond length controls, state that result and demote it; do not search for a favorable metric.
5. If CrossSteer does not outperform a fair baseline, narrow the claim to the evidence actually obtained.
6. Every main result needs the exact config, data split, paired uncertainty, and completion manifest.
7. Do not choose a favorable random seed.  Greedy primary runs use paired problem-level uncertainty; any stochastic robustness run declares its seed set in advance and reports every seed.

## Documents

| File | Purpose |
|---|---|
| `EDITOR_REQUIREMENTS.md` | Exact editor-level acceptance conditions. |
| `REVIEWER_COMMENT_TRACKER.md` | One-to-one mapping from each comment to evidence and manuscript changes. |
| `EXPERIMENT_MATRIX.md` | Frozen revision experiments, stages, gates, and artifacts. |
| `STATISTICAL_ANALYSIS_PLAN.md` | Predeclared splits, tests, and reporting rules. |
| `CLAIM_REVISION_MAP.md` | Old claims, revised claims, and evidence thresholds. |
| `CONTRIBUTION_STRENGTHENING_PLAN.md` | Frozen strong-claim strategy, three evidence pillars, and the result-dependent writing decision. |
| `DIAGNOSTIC_GENERALIZATION_PROTOCOL.md` | Frozen diagnostic-reproducibility plan for strengthening or rejecting the first claim without metric selection. |
| `R1_FORMAL_EXECUTION_CHAIN.md` | R1 32k sampled validation-to-locked execution contract, isolated from the Qwen short-context table. |
| `evidence/validation-family-qwen-instruct-v1/` | Exact 128-cell, hash-linked Qwen frozen-validation evidence and selected-cell appendix rows. |
| `scripts/revision_build_validation_family_report.py` | Hash-linked all-cell validation report; prevents selected validation peaks from replacing the complete frozen landscape. |
| `RESULTS_REPLACEMENT_LEDGER.md` | Legacy tables/claims that must be replaced, demoted, or removed before resubmission. |
| `SUBMITTED_ARTIFACT_INTEGRITY.md` | Hash record for the immutable June 1 submitted artifact. |
| `SERVER_AUDIT_2026-08-03.md` | Current remote compute, storage, and asset audit. |

## Final author handoff

The page-verified submission checklist is `FINAL_AUTHOR_CHECKLIST.md`. It maps
the response letter to the current 17-page PDF and records the final official
B-decision email procedure without reopening experiment selection.

Build the complete official bundle with:

```bash
python scripts/build_tacl_resubmission_bundle.py
```

The builder regenerates the anonymous response letter and anonymized original
decision/reviews, then inserts those files together with the revised manuscript
into `revision/RESUBMISSION_BUNDLE.pdf` using the official TACL bundle structure.
The canonical bundle is emailed to the Editors-in-Chief; it is not uploaded as a
new submission. The separate source/evidence archive is an internal
post-acceptance release candidate and is not supplementary review material.

## Compute status

All formal GPU experiments required by the revision are complete.  The release
contains bounded evidence artifacts and does not require access to the authors' current
server state for verification; the CPU smoke test and report-recomputation tests run
from the audited internal release candidate.

## Latest evidence closure — 2026-08-15

The frozen downstream revision chain is complete.  The strongest positive result
is the predeclared five-label target-local GSM8K point (73% versus a shared 62%
baseline, +11 pp, 95% CI [+4,+18], Holm-adjusted p=0.0443); the remaining label
budgets are non-monotone and inconclusive.  The primary eight-method locked
family, MATH-500, SVAMP, and all Qwen long-context schedules do not support a
reliable source-direction gain.  The R1 official-context readiness gate stopped
that branch at 2/20 severe repetition loops (10%, above the 5% ceiling).

The paper and response now present this as a protocol-bounded measurement and
controlled intervention audit, not as a universally effective steering method.
All four long-context reports and the retrospective signature-stability audit
are mirrored under `evidence/`.  Local verification: 17-page Tectonic build
with no overfull boxes or unresolved references, 149 tests passed, and the
anonymous release preflight has zero missing required items and zero identity
hits; the clean archive has been extracted and rechecked with 149 tests
and the CPU smoke test.  The source and generated PDFs are committed on the final revision branch.

## Final handoff status — 2026-08-19

The manuscript, response letter, anonymized decision/reviews copy, and official
single-PDF resubmission bundle were rebuilt together on 2026-08-19 from the
17-page manuscript. The anonymous-release preflight remains an
internal reproducibility check; the source/evidence archive must not be attached
as supplementary review material. The canonical email attachment is only
`TACL-11241-major-revision-resubmission-bundle-20260819.pdf`.
