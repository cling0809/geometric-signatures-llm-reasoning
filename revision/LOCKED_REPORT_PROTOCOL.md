# Locked Steering Report Protocol V1

This protocol is executed only after a target's validation selection has been
written and all eligible methods have completed their one-shot locked runs.  It
prevents a final table from selectively copying a favorable row.

## Inputs

For every eligible method in one frozen selection file, the report builder
requires:

- `DONE`, `resolved_manifest.json`, `per_sample.parquet`, `summary.csv`, and
  `locked_launch_manifest.json`;
- a launch manifest whose selection SHA-256 and exact layer/alpha specification
  match the frozen selection;
- exactly the locked problem IDs 200--299, one no-steering and one selected
  intervention row per problem;
- token-level termination fields, behavior fields, and correctness labels.

The supplied run set must equal the complete set of **eligible** methods in the
selection artifact.  An ineligible method is retained in the selection artifact
and is not silently replaced with a different cell.

## Cross-method comparability gate

Before computing any method ranking, the builder verifies that all locked runs
share the target model, tokenizer source, prompt, decoding/max-token settings,
schedule, hook convention, and the same deterministic no-steering outputs on
every locked problem.  A mismatch refuses the report.

## Outputs

`revision_build_locked_report.py` writes:

1. `locked_summary.csv` with accuracy, paired delta, 10,000-replicate bootstrap
   CI, repair/break counts, raw exact sign p-value, Holm-adjusted p-value,
   generated-token means, truncation rate, and repetition rate;
2. `locked_summary_rows.tex` for direct inclusion in the revised table;
3. `locked_report_manifest.json` with hashes of the selection, each input, and
   the generated report outputs.

The Holm family is exactly the set of eligible methods for that target/locked
comparison.  It does not pool unrelated targets or OOD tests.

## Interpretation rule

A point estimate is never copied into the paper alone.  A superiority statement
requires the predeclared paired evidence, behavior diagnostics, and the complete
frozen baseline family.  The report is an aggregation artifact, not an
additional selection stage.

## Target-label-efficiency companion report

The target-label curve has its own report builder because its vectors are fit on
different deterministic subsets of the same target-calibration data.  It requires
all six frozen points (`0/5/10/20/50/100`), checks that each point preserved the
full target-calibrated intervention configuration and exact locked IDs, and
requires byte-equivalent unsteered baselines before reporting paired statistics.
The zero-label source-transfer reference is not omitted when it is negative.
