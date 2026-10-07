# TACL-11241 Revision Manuscript Source

This directory contains the **major-revision manuscript source**, not the
immutable submission snapshot.  The submitted TACL artifact is preserved
outside this worktree and must not be overwritten from this directory.

## Current revision state

- The paper source has been rewritten to separate diagnostic associations from
  causal/performance claims.
- Historical GeoVote utility numbers and exploratory CrossSteer peaks are not
  carried forward as revised evidence.
- New results may be inserted only from the frozen validation/locked/OOD/
  long-context report artifacts in `../revision/`.
- LaTeX auxiliary files are kept out of this source directory. Build into a
  temporary output directory, inspect the rendered pages, and copy the verified
  PDF back to `main.pdf` only after the audit passes.

## Build after result replacement

```bash
cd paper
tectonic main.tex
```

The final pre-resubmission check must verify the new PDF, page count, figures,
references, appendix, response-letter page/line mappings, and that no
`[RESULTS PENDING]` marker remains in the active manuscript source.

## Figure provenance

Body-figure generators, the shared publication style, frozen compact inputs, and
regeneration or verification instructions are recorded in
`FIGURE_PROVENANCE.md`.  The author-supplied Fig. 1 overview is integrated as a
conceptual, data-free schematic; the quantitative figures remain tied to frozen
inputs and scripts.  In particular, the correctness-direction density view is
reproducible from a tracked compact coordinate artifact rather than from an
opaque PDF alone.

Historical figure prompts and superseded section drafts are archived outside
this active manuscript directory.  They are not figure sources and must never
be used to regenerate positive GeoVote or exploratory CrossSteer claims for this
revision.

## Section map

| File | Revision role |
|---|---|
| `sections/00_abstract.tex` | Claim-bounded abstract; no historical steering headline |
| `sections/01_intro.tex` | Motivation, contribution boundaries and frozen evidence standard |
| `sections/02_related.tex` | Trajectory analysis and steering baseline positioning |
| `sections/03_framework.tex` | Unified trajectory/signature/GeoVote/CrossSteer framework |
| `sections/04_phenomenon.tex` | Controlled diagnostic evidence only |
| `sections/05_geovote.tex` | Length-control audit and explicit diagnostic-only boundary |
| `sections/06_crosssteer.tex` | Frozen complete-family intervention comparison protocol |
| `sections/08_discussion.tex` | Interpretation, limitations and failure boundaries |
| `sections/09_appendix.tex` | Reproducibility, statistics, baseline and grader artifacts |

Do not cite old files under `results/`, historical PDFs, or archived exploratory
figures as revision results without passing through
`../revision/RESULTS_REPLACEMENT_LEDGER.md`.
