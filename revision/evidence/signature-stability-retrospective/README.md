# Retrospective signature-stability audit

This compact artifact records the problem-level, 1,000-resample bootstrap audit
used to answer Reviewer B.1.  The audit resamples shared problem IDs, keeps
metric/layer cells matched, and recomputes directed nearest-neighbor relations.
It is retrospective evidence from the submitted-paper signature artifacts; it
is not a fresh corrected-generation confirmation run and it never selected an
intervention layer, sign, alpha, or seed.

The result is deliberately narrow: some directed relations are stable within a
fixed scale/protocol, but the fine topology changes across scale and is not
claimed as a universal post-training fingerprint.  The Qwen 1.5B audit is under
`/root/AI/runs/tacl-revision/signature-stability-1p5b-retrospective`; the 7B
comparison is the corresponding retrospective scale audit.

`7b_pairwise_signature_distance.csv` is the locked 7B pairwise distance matrix
behind Appendix Table 11 (GSM8K $n{=}100$, greedy, `max_new=512`).  It is
copied verbatim from the submitted-paper scale-replicate artifact
`2026-05-26_tacl-scale7b/pairwise_signature_distance.csv` in the submitted
source snapshot; the values match the reported Table 11 entries exactly
(Base--Instruct 2.131, Base--Math 2.814, Base--R1 0.844, Instruct--Math 3.076,
Instruct--R1 2.198, Math--R1 2.949).  It is retrospective, pre-correction
descriptive evidence only, not a corrected-generation confirmation run.
