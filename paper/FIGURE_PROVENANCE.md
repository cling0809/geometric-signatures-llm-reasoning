# Figure Provenance and Regeneration

This file records the sources for every figure used in the body of the
TACL-11241 major-revision manuscript.  It deliberately distinguishes
conceptual diagrams from data-derived diagnostics, and it prevents an opaque
PDF or a manually edited image from becoming the only evidence source.

## Body figures

| Manuscript figure | Artifact | Type | Generator and frozen input | Regeneration command |
|---|---|---|---|---|
| Revision protocol overview | `figures/fig1_overview.png` | Conceptual protocol schematic | Author-supplied generated design; print-prepared 2× raster of that same schematic, no experimental values | Verify with `file paper/figures/fig1_overview.png` and inspect PDF p.3; retain draw.io/SVG sources in the private worktree, not in the anonymous bundle |
| Curvature intuition | `figures/fig_curvature_schematic.pdf` (PNG preview also generated) | Conceptual 2D schematic | `scripts/make_revision_curvature_figure.py`; vector schematic, no hidden states or results | `python scripts/make_revision_curvature_figure.py` |
| Length-residualized signature matrices | `figures/fig2_signature_heatmaps.pdf` | Data-derived descriptive diagnostic; homogeneous heatmaps only | `scripts/make_enhanced_result_figures.py --which signature`; four fixed CSVs in `experiments/2026-05-17_phase3-paradigm-4/` | `python scripts/make_enhanced_result_figures.py --which signature` |
| Correctness-direction density view | `figures/fig3_correctness_separation.pdf` | Data-derived descriptive visualization; homogeneous density panels | `scripts/make_phenomenon_hidden_figures.py`; compact frozen coordinates in `data/hidden_projection_coords.csv` | `python scripts/make_phenomenon_hidden_figures.py plot --coords paper/data/hidden_projection_coords.csv --which separation` |
| Signature-distance stability | `figures/fig_signature_distance_stability.pdf` | Data-derived paired descriptive audit; homogeneous dumbbell plot | `scripts/make_enhanced_result_figures.py --which distance`; the same four frozen signature CSVs | `python scripts/make_enhanced_result_figures.py --which distance` |
| Locked intervention comparison | `figures/fig_locked_comparison_forest.pdf` | Data-derived paired result; homogeneous forest plot | `scripts/make_revision_locked_forest_figure.py`; `revision/evidence/locked-gsm8k-qwen-instruct-v3/locked_summary.csv` | `python scripts/make_revision_locked_forest_figure.py` |
| Retained long-context accuracy audit (not in body) | `figures/fig_long_context_accuracy.pdf` | Data-derived paired robustness audit; retained regeneration artifact, not a manuscript figure | `scripts/make_revision_long_context_figure.py`; four mirrored Qwen long-context CSV reports under `revision/evidence/long-context-qwen-instruct-v3/` | `python scripts/make_revision_long_context_figure.py` |
| Retained long-context length audit (not in body) | `figures/fig_long_context_length.pdf` | Data-derived behavioral-cost audit; retained regeneration artifact, not a manuscript figure | `scripts/make_revision_long_context_figure.py`; four mirrored Qwen long-context CSV reports under `revision/evidence/long-context-qwen-instruct-v3/` | `python scripts/make_revision_long_context_figure.py` |

## Visual grammar contract

Each numbered data figure answers one scientific question with one primary
chart grammar.  Multi-panel figures are allowed only when every panel shares
the same axes semantics and mark type.  Heatmaps, density curves, forest plots,
accuracy intervals, and token-length lollipops are therefore kept in separate
figure files and carry separate captions.  Figures 1 and 2 are conceptual
schematics and are not treated as quantitative multi-panel results.

### Editorial rule for multi-panel figures

For the revision, a multi-panel figure is allowed only when all panels answer
the same measurement question with the same visual marks.  We do **not** place
a heatmap beside a line/density plot, a forest plot beside a bar chart, or an
accuracy panel beside a token-cost panel.  The active manuscript therefore
uses the following frozen mapping:

| Figure | Single visual grammar | What it tests |
|---|---|---|
| 1 | conceptual workflow schematic | protocol and claim boundaries; no quantitative result |
| 2 | conceptual curvature schematic | intuition for adjacent hidden-state displacement angles; no quantitative result |
| 3 | metric--layer heatmaps | descriptive signature matrices under one protocol |
| 4 | paired dumbbell plot | budget sensitivity of signature distance |
| 5 | correctness-direction density panels | descriptive correct/incorrect separation |
| 6 | paired forest plot | complete locked intervention comparison |

The two long-context questions are intentionally separate: accuracy and output
length are not combined into one composite figure.  Likewise, the signature
heatmaps, distance audit, and correctness-density view remain separate even
though they concern the same diagnostic section.

## Shared visual system

All data-derived figures in this pass use the shared Python style module
`src/geoprobe/revision/figure_style.py`: a restrained color palette, compact
panel labels, vector-safe PDF text, and high-resolution PNG previews.  The
conceptual overview (Fig. 1) is the same author-supplied generated schematic,
print-prepared for the TACL PDF, and the curvature
schematic (Fig. 2) is kept as a separate conceptual diagram; Figures 3--6 use
the shared data-figure style, while each figure keeps its own single chart
grammar. Long-context result figures are retained only as regeneration artifacts;
the body uses Appendix Table~10 to avoid redundant quantitative displays.

## Compact coordinates for the correctness-direction figure

`data/hidden_projection_coords.csv` is a **compact derived artifact**, not a
raw hidden-state dump.  It contains exactly 11,600 rows: 100 GSM8K examples ×
29 layers × 4 diagnostic checkpoints.  Its columns are `model`, `sample_id`,
`correct`, `layer`, `x_correctness`, and `y_residual_pc`.  The SHA-256 hash is:

```text
7514cf779f0d84fbb59f2a3e6a108e3006b489afd357b323231decba3d7510b6
```

The plotting script standardizes the layer-14 `x_correctness` coordinate within
each model only for display, then draws separate correct/wrong density curves.
Both the direction and the labels arise from the same diagnostic split, so this
figure is **descriptive only**.  It must not be used to select a layer,
direction, sign, alpha, target, or causal mechanism claim.

## Artifact policy

- Regeneration changes must be made through the listed generator and committed
  together with their input/provenance update.
- The all-cells signature heatmap deliberately contains no outlined or
  highlighted metric-layer region.
- Raw trajectories, model weights, caches and long-context completions remain
  outside the repository; the release package instead provides configs, hashes,
  compact derived artifacts and verifiers.
