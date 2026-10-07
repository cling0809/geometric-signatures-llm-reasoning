# Figure layout audit — TACL 11241 major revision

## Decision

Do not use heterogeneous quantitative composites in the revised manuscript.
A multi-panel figure is acceptable only when every panel uses the same visual
marks and the same axis semantics.  A heatmap, a density/line chart, a forest
plot, a point-and-interval chart, and a token-cost lollipop are therefore
separate figures, not subpanels of one figure.

## Active body figures

| Manuscript figure | Current file | Panel grammar | Status |
|---|---|---|---|
| Fig. 1 | `paper/figures/fig1_overview.png` | conceptual workflow schematic | pass; author-supplied final design, retained as the audited high-resolution PNG |
| Fig. 2 | `paper/figures/fig_curvature_schematic.pdf` | conceptual curvature schematic | pass after full-width layout |
| Fig. 3 | `paper/figures/fig2_signature_heatmaps.pdf` | metric × layer heatmaps | pass |
| Fig. 4 | `paper/figures/fig_signature_distance_stability.pdf` | paired dumbbell plot | pass |
| Fig. 5 | `paper/figures/fig3_correctness_separation.pdf` | density panels | pass |
| Fig. 6 | `paper/figures/fig_locked_comparison_forest.pdf` | forest plot | pass |
| -- | `paper/figures/fig_long_context_accuracy.pdf` | point-and-interval panels | retained for regeneration; not included in body |
| -- | `paper/figures/fig_long_context_length.pdf` | lollipop panels | retained for regeneration; not included in body |

## Required review rule

When adding a new result, first identify its measurement question and chart
grammar.  If it does not share both with an existing figure, create a new
figure file and a new caption.  Do not append a second chart type merely to
save space.

## Visual inspection completed

The compiled 17-page PDF was rendered page-by-page and checked for:

- mixed quantitative chart types inside a single figure;
- detached captions or legends;
- clipped labels and overlapping panels;
- figures whose visual encoding contradicts its caption.

No active quantitative figure currently mixes heatmaps with curves, or
accuracy with token-cost marks.  The manuscript-level separation is also guarded
by `test_active_body_figures_follow_one_visual_grammar_contract`, so a later
space-saving edit cannot silently recombine heterogeneous result panels.  Figures 1 and 2 are conceptual diagrams; Fig.
1 is the only deliberate multi-module workflow composition and is reserved for
a future BioRender redraw.  Neither conceptual figure makes a quantitative
claim.
