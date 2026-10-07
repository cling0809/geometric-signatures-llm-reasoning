# Reproducibility Protocol for the TACL-11241 Major Revision

This document distinguishes a **CPU installation smoke test** from the actual
GPU experiments. A successful smoke test proves that the audited code path,
manifest writing, trajectory serialization and split-isolation checks work. It
does **not** reproduce a reported scientific number.

## Immutable boundaries

- The submitted manuscript snapshot is held in a separately versioned immutable
  archive; this revision tree never writes to or resets that archive.
- Revision work is confined to the clean `codex/tacl-11241-major-revision` worktree.
- Every formal vector is fit on GSM8K IDs 0--99; validation is 100--199; locked
  in-domain evaluation is 200--299. The runner refuses declared overlap.
- A selected locked configuration is launched only by
  `scripts/revision_launch_locked_selected.py`, which fingerprints the selection
  file and refuses a different output manifest.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[torch,dev]'
PYTHONPATH=src python -m pytest -q tests
```

The authoritative revision environment records model revision, tokenizer,
prompt, evaluator, random seed, resolved configuration, git SHA and artifact
hashes in each run directory. Use `python -m pytest`, not a bare `pytest`, to
avoid interpreter mismatch.  The project requires Python 3.10 or newer; the
local `/opt/anaconda3/envs/pytorch` interpreter is Python 3.9 and is not a
supported test environment, even if its optional packages are completed.  A
supported Python 3.13 environment with PyTorch and PyArrow is the verified
configuration for the 149-test result.

## CPU-only installation smoke test

```bash
PYTHONPATH=src python scripts/revision_smoke_demo.py --out /tmp/geoprobe-revision-smoke
cat /tmp/geoprobe-revision-smoke/SMOKE_OK.json
```

Expected result: `SMOKE_OK.json`, source/target synthetic runs, and a compact
vector registry. The JSON reports two `[3,5]` vectors. Use a disposable temporary
location; these are synthetic non-paper artifacts.

## Formal GSM8K steering protocol

1. Generate only source-training calibration trajectories/completions for IDs
   0--99 using the resolved config.
2. Build a fingerprinted registry from source and target calibration runs.
3. Run the predeclared validation grid on IDs 100--199 with the fixed layer,
   alpha, decoder, norm and behavior guard recorded in
   `revision/VALIDATION_GRID_V1.md`.  For each completed grid, write an
   immutable multi-EOS sidecar before release with:

   ```bash
   PYTHONPATH=src python scripts/revision_audit_generation_runtime.py \
     --run /absolute/path/to/completed-grid \
     --generation-config /absolute/path/to/model/generation_config.json
   ```

   The sidecar records the exact EOS-token set declared by the model generation
   config, config hashes, explicit stop-reason counts, and truncation rate.  It
   never edits the original manifest or generated rows.
4. Run `scripts/revision_select_validation.py` **before** locked generation.
   Immediately render the complete validation landscape (not only the selected
   peak) with:

   ```bash
   PYTHONPATH=src python scripts/revision_build_validation_family_report.py \
     --selection /absolute/path/to/selection.json \
     --out /absolute/path/to/validation-family-report
   ```

   The report verifies every `summary.csv` hash recorded by the immutable
   selection, recomputes each decision from the frozen rule, and writes all
   layer/alpha cells plus a selected-cell appendix table.
5. Only if a method is eligible, launch its single GSM8K locked run with
   `scripts/revision_launch_locked_selected.py --execute`.
6. Freeze that selected method for all-500 MATH-500 and all-1,000 SVAMP OOD
   evaluation, then long-context evaluation. Do not scan a new layer, sign,
   alpha or metric after reading locked labels.

## Held-out diagnostic-generalization protocol

The first paper claim has a separate diagnostic-reproducibility audit.  It is
not a steering selection step and cannot be used to choose a layer, alpha,
prompt, decoder or target.  After registered trajectory runs are complete, run:

```bash
PYTHONPATH=src python scripts/revision_diagnostic_generalization.py \
  --run base=/absolute/path/to/base-run \
  --run instruct=/absolute/path/to/instruct-run \
  --run math=/absolute/path/to/math-run \
  --out /absolute/path/to/diagnostic-generalization \
  --partition-salt tacl-11241-diagnostic-generalization-v1
```

The runner refuses missing generated-token lengths and, by default, refuses a
partition in which any model has fewer than 10 correct or 10 incorrect
trajectories. It balances correct/incorrect counts across models within each
SHA-256 partition, uses all registered metrics and relative-depth bins,
residualizes each metric against `log1p(n_gen_tokens)`, and writes held-out
retrieval, bootstrap, permutation and manifest artifacts. The CLI records its
`--min-per-class` threshold in the manifest; lowering it below 10 requires an
explicit protocol amendment and cannot be used to rehabilitate an
underpowered legacy cell. Cross-lineage replication is a fresh registered run,
not a source of calibration for Qwen or CrossSteer.

Before `revision_select_validation.py` can read any validation score, it derives
(or verifies) the run-level multi-EOS sidecar directly from the declared target
model's `generation_config.json`, then writes
`<selection_stem>_family_integrity.json`.  This score-blind artifact requires
complete grid coverage, matching manifests, hashes and decoder envelopes, plus
byte-equivalent no-steering baselines across all methods.  Its SHA-256 is stored
inside the immutable selection JSON.

## R1 official-context protocol

R1-Distill does not use the Qwen short greedy envelope.  Its technical readiness
uses `max_new_tokens=32768`, `temperature=0.6`, `top_p=0.95`, a fixed
problem-indexed seed, and compact one-token replay pooling.  The 20-item
readiness audit is correctness-blind and must pass before a 100-item calibration
can be used.  In the completed revision readiness artifact, 2/20 outputs had
severe repetition (10%), above the frozen 5% ceiling, so the R1 branch is
stopped before calibration, steering, locked evaluation, or mitigation.  The
legacy R1 512-token configuration is a saturation diagnostic only; no Qwen
result substitutes for an R1 repair claim.  Reproduction of this boundary uses
the three `revision_gsm8k_r1_official32k_*.yaml` configs, the pooled
extractor/readiness audit, and both R1 protocol documents.

## What the final anonymous release must include

- exact paper revision source, response-letter Markdown, and its reproducible
  Pandoc/Tectonic build script;
- configs, scripts, unit tests, resolved-run manifests, and completed
  multi-EOS generation-runtime sidecars;
- grader/evaluator version and prompt/template description;
- per-problem correctness/behavior fields and aggregate paired statistics;
- compact vectors/provenance, not private model weights or enormous raw hidden
  dumps;
- `paper/FIGURE_PROVENANCE.md`, the body figure generators, and the compact
  correctness-projection coordinates needed to regenerate the descriptive
  density visualization; and
- an install command plus this CPU smoke command.

Before packaging, run `scripts/revision_anonymous_release_preflight.py` with the
final locked/OOD/long-context report manifests.  It inventories required source
artifacts, rejects local identity markers, records intentional exclusions, and
fails if final evidence is required but missing.

Any unavailable model artifact or benchmark license constraint will be stated
explicitly in the reproducibility checklist rather than silently omitted.

## Verified source-state snapshot — 2026-08-04

The following is a dated historical snapshot from before the frozen Qwen
validation family completed.  It validates packaging and protocol code, not
scientific performance; the current final status is recorded in
`revision/FINAL_CLAIM_AUDIT.md` and `revision/FINAL_AUTHOR_CHECKLIST.md`.

```text
PYTHONPATH=src python scripts/revision_smoke_demo.py --out /tmp/geoprobe-revision-smoke-20260804-0631
SMOKE_OK.json: protocol=tacl-11241-cpu-smoke-v1
vectors: crosssteer_source=[3,5], target_calibrated=[3,5]

PYTHONPATH=src python -m pytest -q
124 passed

PYTHONPATH=src python scripts/revision_anonymous_release_preflight.py --root . ...
source tree: ready; final anonymous evidence: pending frozen reports
```

The CPU smoke deliberately uses synthetic vectors and split IDs only.  It is not
included in the paper's numerical evidence and cannot be used to select a
method, layer, alpha, seed, decoder, or task.

## Anonymous problem-level audit release

The final package includes content-free per-problem tables under
`revision/evidence/per-problem-audit-v1/` for locked GSM8K, MATH-500, SVAMP,
and the target-label curve.  These tables retain the comparison arm, sample ID,
correctness, selected layer/strength, token counts, stopping, truncation,
repetition, diversity, and answer-marker telemetry.  They exclude benchmark
questions, gold answers, predictions, generated text, and generation seeds.

`scripts/revision_export_anonymous_per_problem.py` verifies each formal
`per_sample.parquet` against the SHA-256 recorded by its frozen report manifest
before exporting.  Every released CSV has a sibling manifest containing all
source hashes, the report-manifest hash, output hash, row count, column list, and
privacy boundary.  `tests/test_per_problem_release.py` verifies those manifests
and recomputes baseline/method accuracy, repair/break counts, mean generated
tokens, and row counts against the frozen summary tables.
