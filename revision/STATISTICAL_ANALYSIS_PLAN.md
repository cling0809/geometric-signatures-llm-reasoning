# Statistical Analysis Plan

## Unit of analysis

The independent unit is the **problem ID**.  All candidate samples, steered/unsteered outputs, and repeated decoding variants for one problem remain in the same split and are resampled together.

## Split protocol

For a formal GSM8K steering experiment:

```text
Source-direction training: problem IDs 0–99
Validation (layer/strength/schedule selection): IDs 100–199
Locked test: IDs 200–299
Replication reserve: IDs 300–499
```

The source vector is computed only from training IDs.  Validation and locked-test target answers never contribute to its construction.  OOD tasks use the frozen method selected on GSM8K validation; no OOD layer, alpha, sign, schedule, or threshold search is allowed.

## Primary metrics

- Accuracy / exact-answer correctness.
- Paired delta against the no-intervention or baseline output on the same problem.
- Repairs: baseline incorrect and method correct.
- Breaks: baseline correct and method incorrect.
- Best-of-N selection accuracy for GeoVote.
- Long-context stability: repetition-loop rate, max-token truncation rate, mean length, distinct-2, distinct-3, stop reason.

## Uncertainty and tests

For every predeclared primary comparison:

1. Paired nonparametric bootstrap over problem IDs, 10,000 replicates, 95% percentile CI.
2. Exact paired sign/permutation test on discordant problem-level outcomes.
3. Report raw p-values and Holm-adjusted p-values within each declared result family.
4. Never interpret a point estimate as a result without its paired CI and repairs/breaks.

## GeoVote controls

All methods consume the identical generated candidate pool.  The majority row is
also the text-only answer-frequency selector after answer normalization; the
shortest-output row is the token-count-only selector.  The primary comparisons
are original GeoVote versus these controls, length-residualized/no-trajectory-
length GeoVote, sequence and average log-probability, and the explicit
length-matched null.  Each control is reported once rather than duplicated under
synonymous names.

Length residualizers, scoring rules, signs, layers, and aggregation options are fit/selected on calibration and validation only.  The locked test is executed once after freezing those choices.

## CrossSteer baseline fairness

All formal steering methods use the same:

- source-label count;
- source training problem IDs;
- target validation and locked-test IDs;
- target model, tokenizer, prompt, decoding, max token budget, and evaluator;
- layer candidate set and alpha/schedule candidate budget;
- direction norm convention;
- output storage and statistical analysis.

Any baseline whose official construction requires a different input object (for example contrastive prompt pairs) receives the same number of source examples and a documented construction procedure.

Before vector construction, source and target calibration runs undergo a structural capability audit: exact expected IDs, one completion per ID, nondegenerate correct/incorrect counts, and generation-budget saturation.  They must also retain a pre-generation manifest with the model's complete EOS set and row-level stop/truncation telemetry.  The capability audit is bound by SHA-256 to the exact config and labels used by the vector registry; a source or target run above the 25% generated-token budget-hit ceiling cannot define a formal direction.  This gate prevents an invalid direction artifact; it does not replace the separate official/reference-score comparability check.

## Decoding randomness and seed policy

The primary GSM8K steering comparison uses **greedy decoding** (`do_sample=False`).
Consequently, model-output randomness is not a degree of freedom in the primary
comparison: for a fixed model revision, prompt, hook, and input, there is one
output per problem.  Its uncertainty is quantified over the prespecified
problem IDs by paired bootstrap and exact paired tests, not by selecting a
random seed with a favourable accuracy.

Where a method genuinely requires stochastic source sampling (for example the
K=4 prompt-contrast pool), the candidate pool is generated once with a recorded
seed and reused byte-for-byte by every applicable method.  The seed, model
revision, prompt, and generation configuration are included in the completion
manifest.

If a supplementary stochastic-decoding robustness experiment is run, its seed
set must be declared before any locked labels are inspected.  It reports every
seed, the across-seed mean and standard deviation, and a paired aggregate over
problem IDs.  It may not choose the best seed, best method-specific seed, or
best task cell for the main table.  Validation may select a configuration only
by the predeclared aggregate rule; the locked test remains one-shot.
