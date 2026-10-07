# TACL 11241: Final Revision Narrative Plan

## Objective

Maximize the probability of final acceptance through a complete, auditable
revision.  The revised paper must make the strongest claims that the frozen
evidence supports, but must not turn point estimates, failed eligibility cells,
or exploratory runs into positive evidence.

## Editorial strategy

The action editor's decision is conditional: the revision must answer four
specific requests.  The paper should therefore be written as a direct answer
to those requests, not as a defense of every sentence in the submitted version.
The positive contribution is the trajectory-level measurement question and the
measurement/evaluation protocol.  GeoVote and CrossSteer are inference-time
hypotheses that are tested, not assumed to be successful methods.

## Claim hierarchy

### Claim 1: controlled measurement

Under the original fixed Qwen extraction protocol, metric--layer correctness
associations differ across the matched post-training variants.  This is a
protocol-bounded descriptive result.  It is not a universal fingerprint, a
mechanism explanation, or evidence that one layer should be used for steering.

The later score-blind confirmation chain is reported as an eligibility boundary:
Qwen 512, Qwen 2048, Gemma 512, and Gemma 2048 all exceeded the predeclared
25% truncation ceiling for at least one required model.  These cells are not
used as positive signature evidence.  The wording must say that the fresh
confirmation was not qualified, not that it proved the absence of geometry.

### Claim 2: GeoVote

GeoVote is retained as a controlled audit.  Same-pool comparisons show that its
apparent advantage is explained by completion length and does not reliably beat
majority, token-count, or log-probability controls.  The revision must remove
any claim that GeoVote is a reliable performance method.

### Claim 3: CrossSteer and intervention

The locked GSM8K family does not support source-direction superiority.  Source
CrossSteer is -1 pp; CAA and SAE have larger positive point estimates but their
paired confidence intervals cross zero and their Holm-adjusted tests are not
significant; sign-reversed CrossSteer is also positive at the point-estimate
level.  OOD results must be reported for every frozen method, including all
failures.  No post-hoc method, layer, alpha, seed, or subset selection is
permitted.

If the complete OOD chain remains negative, the paper should say that the
baseline-complete audit finds no reliable inference-time gain under the tested
protocol.  This is a boundary result that directly answers the reviewers'
request about whether full trajectories buy practical steering utility.

## What to emphasize

1. The manuscript now separates three questions that were conflated in V1:
   (a) is there a measurable association, (b) does it improve candidate choice,
   and (c) does it causally improve target generation?
2. Every method is evaluated on paired problems with frozen configurations,
   behavior telemetry, and complete baselines.
3. The revised result is more informative than a single positive intervention
   table because it identifies length, sign, target calibration, OOD, and
   long-context boundaries.
4. The code/reproducibility package is itself part of the revision response:
   resolved configs, evaluator audits, manifests, hashes, and tests are
   released together.

## Words to remove

Do not use: universal fingerprint, transferable correctness direction,
geometry causes correctness, repairs reasoning, robustly improves, consistently
outperforms, or proves a reasoning mechanism.

Use instead: protocol-bounded association, descriptive signature, point estimate,
paired interval, inconclusive, controlled negative, eligibility boundary,
source-transfer hypothesis, and no reliable gain under the frozen protocol.

## Reviewer-to-evidence map

| Request | Manuscript response | Evidence required |
|---|---|---|
| Consolidate methods | Single Framework section before experiments; notation table; path analogy; pseudocode | Compiles, no undefined symbols |
| Control GeoVote length | Same candidate pool; majority, token-count, residualized and logprob controls | Paired CI and exact tests for every comparator |
| Add steering baselines | CAA, ActAdd, SAE sparse, coordinate-sparse, sign-reversed and random controls | Complete locked table and OOD table |
| Explain source vs target calibration | Label-budget curve; explicit limited use case | Frozen calibration-budget analysis |
| Address 32k loops | Correctness-blind readiness gate; report R1 exclusion and Qwen long-context telemetry | Readiness manifest and long-context runs |
| Improve OOD | MATH-500 and SVAMP with fixed GSM8K configuration | Grader audits, manifests, complete per-method reports |
| Improve readability | Expand names, intuitive geometric analogy, algorithms, plain-language result paragraphs | Final PDF visual review |

## Final paper shape

1. Introduction: separate measurement from utility; state the three questions.
2. Related work: geometry, hidden-state correctness probes, steering, voting.
3. Framework: trajectory, metrics, signature, GeoVote, CrossSteer, controls and
   split protocol in one place.
4. Diagnostic phenomenon: fixed-protocol associations and scale/sample limits.
5. GeoVote audit: length-mediated negative result.
6. CrossSteer audit: locked family, behavior controls, OOD, long context.
7. Discussion: what is measured, what is not causal, and why negative controls
   matter.
8. Appendix: full tables, evaluator audits, manifests, additional diagnostics.

## Acceptance-facing conclusion

The revision should not ask the reviewer to accept a failed intervention as a
successful method.  It should show that every requested confound and baseline
was tested, that the original claims were narrowed exactly where the controls
required it, and that the trajectory framework produces a reusable measurement
and falsification protocol.  If an OOD cell is positive, report it as a
predeclared conditional result only if its paired interval, behavior controls,
and family-wise comparison support it; do not let a single point estimate
replace the complete evidence.
