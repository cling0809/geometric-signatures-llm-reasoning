# TACL 11241 B-Decision Acceptance Dashboard

**Objective:** submit a revision that lets the Action Editor verify every
mandatory revision from evidence, manuscript text, and release artifacts.
Scientific claims are strengthened by control completeness, not by selecting
favorable seeds or hiding negative rows.

## Editor-level requirements

| ID | Mandatory revision | Current state | Evidence | Manuscript/response location | Remaining action |
|---|---|---|---|---|---|
| E.1 | Consolidate fragmented methods | Complete | unified definitions, notation table, two algorithms | Paper Sec. 3; response “Mandatory revision 1” | closed after final PDF readability pass |
| E.2 | Disentangle GeoVote from length and add significance | Complete | same-pool K=8 audit; majority/text-only answer-frequency, shortest/token-count-only, residualized/no-trajectory-length, logprob and explicit 1,000-permutation adjacent-length-pair controls on the verified source hash | Paper Sec. 5, App. GeoVote; response “Mandatory revision 2” | closed after hash, candidate-pool, and paired-output verification |
| E.3a | Compare CrossSteer with CAA/ActAdd/sparse methods | Complete | eight-method locked GSM8K family with paired statistics and behavior telemetry | Paper Sec. 6; response “Completed locked result” | none beyond artifact packaging |
| E.3b | Expand OOD | Complete | all 500 MATH-500 and all 1,000 SVAMP, eight frozen methods, audited graders, plus the predeclared 4,096-token MATH-500 budget audit (3.2% baseline truncation; all null) | Paper Sec. 6 and App. complete OOD table; response “OOD” | none |
| E.3c | Analyze long-context stability | Complete | R1 correctness-blind readiness already stopped at 10% severe loops; Qwen 4096/32768 chain completed with all schedules; corrected R1 short-envelope study completed (same-target +3 pp and cross-source -1 pp, both null) | Paper Sec. 6/Limitations; response “Long context” | closed after final PDF audit |
| E.4 | Improve readability | Complete | clearer title, three-level evidence narrative, path analogy, curvature diagram, algorithms, streamlined notation and homogeneous figures | Paper Abstract/Secs. 1/3/4--8; response “Mandatory revision 4” | closed after line-by-line prose and visual PDF audit |

## Reviewer-specific high-risk points

| Concern | Current answer | Risk | Required closure |
|---|---|---:|---|
| B.1 scale-sensitive fingerprint topology | narrowed to protocol-bounded signatures; bootstrap/scale audit reported | Medium | ensure title/abstract never imply universal taxonomy |
| B.3 source transfer vs target labels | complete 0/5/10/20/50/100 curve: source-only inconclusive; five-label target-local +11 pp, Holm p=0.0443; remaining points non-monotone | Medium | preserve the full curve and bounded wording |
| B.4 32k repetition | R1 branch stopped by preregistered readiness rule; Qwen long-context complete: all schedules at both budgets | High | closed; report every schedule, including failures |
| B.5 style/length confound | locked, OOD and long-context telemetry complete; matched random and negative-direction controls included | Low | closed |
| C: no benefit over conventional steering | complete locked family shows no significant CrossSteer superiority over CAA/ActAdd/SAE; response states this directly and retains the method as a tested transfer hypothesis rather than a performance winner | High | closed by explicit claim removal, complete forest/table comparison and compute-cost disclosure |
| C: no usable software | tests, manifests, compact evidence, CPU smoke and release preflight exist | Medium | closed |

## Current scientific disposition

1. **Signature measurement:** retained as a controlled, protocol-bounded
   correctness-association result.  Stable universal fingerprint/taxonomy claims
   are removed.
2. **GeoVote:** retained as a controlled audit; the reliable performance claim is
   removed because length controls explain the apparent advantage.
3. **CrossSteer:** retained as a fully tested source-transfer hypothesis.  Locked
   GSM8K and complete OOD do not support a reliable source-direction gain.
4. **Paper-level contribution:** a trajectory-level measurement framework plus a
   no-leakage, baseline-complete evaluation that distinguishes association,
   candidate-selection utility, and causal intervention.

## Submission gates

- [x] Corrected multi-EOS natural decoding envelope.
- [x] Locked GSM8K complete-family report.
- [x] MATH-500 complete-family OOD report.
- [x] SVAMP complete-family OOD report.
- [x] Target-label-efficiency report.
- [x] Qwen 4096/32768 long-context report.
- [x] All placeholders removed from response letter/manuscript.
- [x] Every response maps to a section/table/appendix artifact.
- [x] `PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q` passes.
- [x] Tectonic build has no unresolved references or overfull boxes.
- [x] Final anonymous release preflight, clean archive, extracted archive recheck, 149 tests and CPU smoke pass.
- [x] Reviewer-style re-review finds no unsupported headline claim (see `revision/FINAL_CLAIM_AUDIT.md`).
