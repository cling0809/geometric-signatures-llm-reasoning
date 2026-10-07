# TACL 11241 final claim audit — 2026-08-19

This is a submission-readiness audit for the current revision worktree.  It
checks that the manuscript and response letter make only claims supported by the
frozen evidence; it is not a new scientific analysis.

## Reviewer-facing closure

| Requirement | Current answer | Evidence location |
|---|---|---|
| Consolidate GeoVote/CrossSteer and notation | Closed | `paper/sections/03_framework.tex`, response Mandatory revision 1 |
| Remove GeoVote length confound | GeoVote is demoted; majority is the text-only answer-frequency control, shortest-output is the token-count-only control, residualized geometry is the no-trajectory-length control, and the explicit adjacent-length-pair null is uniformly negative | `paper/sections/05_geovote.tex`, `paper/sections/09_appendix.tex`, GeoVote evidence |
| Add steering baselines | Eight frozen families are reported, including CAA, ActAdd, SAE sparse, coordinate-sparse, sign-reversed and random controls | `paper/sections/06_crosssteer.tex`, locked evidence |
| Explain scale/sample instability | Bootstrap stability is reported as retrospective, protocol-bounded evidence; universal fingerprint language is removed | `paper/sections/04_phenomenon.tex`, signature-stability evidence |
| Address target calibration | Full 0/5/10/20/50/100 curve is reported; 5-label point is positive but non-monotone curve prevents a general law | label-efficiency evidence and Table `tab:label-efficiency` |
| Address OOD | All 500 MATH-500 and 1,000 SVAMP items are reported with no OOD retuning | OOD evidence and Table `tab:crosssteer-ood` |
| Address 32k loops and schedule behavior | R1 is stopped by the frozen readiness rule; all eight Qwen long-context rows and the audit figure are reported | `paper/sections/06_crosssteer.tex`, Table `tab:long-context` |
| Address style/length confound | Token counts, repetition, truncation, stop behavior and repair/break telemetry are included | locked/OOD/long-context reports |
| Reproducibility/readability | three-level evidence narrative, hash-verified content-free per-problem tables, 149 tests, CPU smoke, required artifacts, one visual grammar per data figure, and anonymous preflight pass | paper Abstract/Introduction/Discussion and `revision/ANONYMOUS_RELEASE_PREFLIGHT.json` |

## Headline-claim guard

The active manuscript and response letter do not claim that:

- geometric signatures are universal or scale-invariant fingerprints;
- GeoVote reliably improves candidate selection;
- source CrossSteer reliably transfers correctness across models;
- a point estimate alone establishes a significant intervention gain; or
- the hidden-state direction implements a causal reasoning mechanism.

The strongest positive statement is deliberately bounded: five target labels
produce a +11 pp locked gain in one Qwen/GSM8K cell, with a paired 95% CI of
[+4,+18] pp and Holm-adjusted p=0.0443; the rest of the curve is reported and is
not monotone.  Two completed predeclared audits reinforce the negative
downstream findings: the 4,096-token MATH-500 budget audit (baseline truncation
3.2%; every paired interval crosses zero) and the corrected R1 short-envelope
study (same-target +3 pp CI [-8,+14]; cross-source -1 pp CI [-12,+10]; both
non-significant), which confirms that the submitted +16 pp does not survive
correction.

## Final verification snapshot

- Paper: 17-page Tectonic build (2026-08-19); Limitations/Artifact/Ethics finish on p.12; references begin p.13; appendix pp.15--17; Fig. 1 at 554 ppi; no overfull boxes, fatal errors, or unresolved references.
- Tests: `PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q` → 149 passed (2026-08-19).
- CPU smoke: `tacl-11241-cpu-smoke-v1` completed (prior frozen run).
- Anonymous release: 20260819 source/evidence tarball rebuilt from the 20260816
  verified baseline; draw.io Fig. 1 sources excluded; full preflight written
  beside the archive; identity-token scan remains part of the bundle builder.
- Server: downstream GPU chains complete; no experiment process remains.
- Git: submitted V1 remains untouched. The 2026-08-19 print-prep of Fig. 1, the
  Table 11 filename restoration, and the rebuilt official bundle are the
  sendable state.

This audit supports a candidate revision package.  It does not guarantee the
editor's acceptance; the remaining uncertainty is scientific/editorial, not an
unfinished experiment or a missing required control.
