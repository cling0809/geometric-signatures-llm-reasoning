# Claim Revision Map

This file prevents the revision from overstating what the evidence supports.

| Submitted / implied claim | Risk exposed by review | Revised claim permitted by evidence | Evidence required |
|---|---|---|---|
| Post-training paradigms leave stable geometric fingerprints. | Fine nearest-neighbor topology changes with scale and sample count; highlighted visual cells can imply post-hoc selection. | Under a fixed protocol, Qwen-derived checkpoints show distinct metric-layer correctness-association patterns. **Do not claim a stable nearest-neighbor fingerprint topology or a selected-cell mechanism.** | 1.5B retrospective bootstrap has only moderate nearest-neighbor stability; complete 7B audit and report both scales; retain an all-cells diagnostic heatmap without selected-region outlines. |
| GeoVote provides geometry-based utility beyond self-consistency. | The selected rule is trajectory length; token-count controls explain the majority margin. | **GeoVote is reported only as a diagnostic and is removed from the utility claim.** | Retrospective same-pool audit: raw geometry confidence has high length correlation (Spearman 0.74 on evaluation); no method exceeded majority; the explicit length-matched null is uniformly negative. |
| CrossSteer is transferable without target-side labels. | Target orientation may require target calibration; target-calibrated steering is stronger. | CrossSteer can receive a transfer claim only for a target that uses a model-valid decoding envelope, passes its frozen capability gate, and survives the full locked comparison. **R1 did not pass that gate:** 2/20 fixed official-context samples had severe repetition (10% > 5%), so no R1 transfer/repair claim is permitted. | Target-label efficiency curve and complete frozen Qwen comparison. The R1 compact-replay readiness artifact is retained as a bounded failure, not a method result. |
| CrossSteer injects correctness. | It may alter length, formatting, or a generic reasoning style. | CrossSteer changes target behavior along a source-derived correct-minus-incorrect direction; mechanism remains unresolved unless behavior controls rule out length/style explanations. | Length-matched, text/behavior, and explicit length-direction controls. |
| CrossSteer is robust at long context. | Constant injection failed at 32k with repetition loops. | CrossSteer has a bounded operating regime. The R1 official-context gate itself observed loop-dominated samples and prohibits a mitigation sweep; no robustness claim is permitted for R1. | R1 readiness manifest: 10% severe repetition exceeds the frozen 5% ceiling; Qwen long-context table is conditional on an eligible locked Qwen method. |
| Full trajectories are preferable to traditional steering. | No direct CAA/ActAdd/sparse baseline. | The full-trajectory construction is superior only in cells where it beats a fair frozen baseline; otherwise its contribution is diagnostic/cross-model transfer. | Locked-test baseline comparison plus a frozen direction-data and matched online-overhead table. |


## Baseline terminology

The revised related-work section distinguishes ActAdd (single prompt contrast),
CAA (multiple contrast pairs), and CrossSteer (source-model full-trajectory
direction). The coordinate-sparse implementation is explicitly labelled as an
ablation, not an SAE method. The formal comparison is now complete: the frozen eight-method validation and
locked-test artifacts are reported in the manuscript and response letter.  The
source-derived direction does not obtain a reliable locked or OOD gain, so no
full-trajectory superiority claim is permitted.
