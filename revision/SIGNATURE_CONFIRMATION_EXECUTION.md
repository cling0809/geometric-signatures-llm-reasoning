# Claim-1 Signature Confirmation: Frozen Execution Contract

## Why this run exists

The submitted signature heatmaps are not confirmation evidence for the major
revision. They predate the new requirements for generation-envelope provenance,
correctness-label audit, explicit truncation telemetry, fixed split-level
support, and held-out statistical testing. The old figures remain descriptive
historical context only.

This execution contract tests the limited revised proposition:

> Under one fixed common extraction protocol, are the full metric-by-relative-
> depth correctness-association matrices reproducibly distinguishable across
> post-training roles on held-out problem partitions?

It does **not** test a causal training mechanism, universal fingerprint taxonomy,
or a steering-layer selection rule.

## Frozen registered conditions

### Primary matched Qwen lineage

| Role | Model | Problems | Budgets |
|---|---|---:|---:|
| Base | `Qwen/Qwen2.5-1.5B` | GSM8K IDs 0–299 | 512, 2048 |
| Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | GSM8K IDs 0–299 | 512, 2048 |
| Math-Instruct | `Qwen/Qwen2.5-Math-1.5B-Instruct` | GSM8K IDs 0–299 | 512, 2048 |
| Coder-Instruct | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | GSM8K IDs 0–299 | 512, 2048 |

### Independent matched-lineage replication

| Role | Model | Problems | Budgets |
|---|---|---:|---:|
| Base | `LLM-Research/gemma-2-2b` | GSM8K IDs 0–299 | 512, 2048 |
| Instruct | `LLM-Research/gemma-2-2b-it` | GSM8K IDs 0–299 | 512, 2048 |

The independent Gemma analysis is separate: it cannot be used to choose a
Qwen metric, layer, prompt, budget, or threshold.

The corresponding immutable configurations are:

```text
configs/revision_signature_{qwen,gemma}_*_gsm8k_300_b{512,2048}_v1.yaml
```

## Input eligibility gate

Before a run enters matrix analysis, `revision_diagnostic_generalization.py`
requires every registered run to have:

1. `DONE`, `config.yaml`, `labels.parquet`, `metrics.parquet`, and all greedy
   trajectory artifacts;
2. a manifest written before generation, binding the config hash, model EOS set,
   decoder token limit, prompt hash and extractor hash;
3. one metric vector and one greedy label per common problem ID;
4. a token budget-consistent `stop_reason` / `truncated` record for every row;
5. an observed truncation rate no greater than 25%; and
6. at least 10 correct and 10 incorrect examples for **every model** in each
   stable A/B/C hash partition after common-ID alignment.

A condition that fails a provenance or support gate is recorded as
`UNQUALIFIED`, retained with its log, and cannot be transformed into a positive
claim by changing budget, prompt, seed, task subset, metric or layer.

## Fixed matrix and statistical analysis

For each model and each A/B/C hash partition, the analysis:

1. uses all seven predeclared trajectory functionals;
2. interpolates every model to 29 relative-depth bins;
3. balances correct/incorrect counts across compared models by stable hash;
4. residualizes each metric score against `log1p(generated token count)`;
5. computes the complete AUC(incorrect vs. correct) matrix; and
6. saves **all** cells in `partition_signature_cells.csv`.

The primary test is held-out signature retrieval: a model matrix from one
partition is compared with the gallery matrices from a different partition
using centered Frobenius distance. Outputs include macro top-1 retrieval,
problem bootstrap confidence intervals, and a checkpoint-name permutation
p-value. A visual heatmap is descriptive; it is never a substitute for the
held-out retrieval test.

## Claim gate

| Outcome | Permitted paper language |
|---|---|
| Qwen held-out retrieval does not exceed the permutation null | No reproducible diagnostic-signature claim; show only an explicitly descriptive matrix, if retained. |
| Qwen succeeds but budget stability fails | The registered Qwen roles are distinguishable under the stated  condition; no stable-fingerprint or universal wording. |
| Qwen succeeds with positive budget stability | The registered Qwen diagnostic pattern is reproducible across these two extraction budgets. |
| Gemma separately succeeds | The predeclared diagnostic procedure replicates in a second matched base/instruct lineage; still no causal or universal claim. |

## Scheduling and isolation

`/root/AI/geoprobe-signature-confirmation-v2` is the dedicated fresh server worktree for the restarted confirmation chain.  The earlier `qwen-signature-confirmation-v1` directory remains preserved as an explicitly paused operational attempt; it is never merged with the fresh chain log or treated as evidence.
Its launcher waits for the ongoing frozen Qwen steering correction chain to
**complete successfully** and for GPUs to drain. If that higher-priority chain
fails operationally, the signature launcher writes a pause marker and preserves
partial trajectories for deterministic resumption; it does not consume both GPUs
while the correction is being diagnosed. It does not alter the submitted V1
snapshot, the active steering worktree, any validation score, or locked test
labels.

Launcher:

```bash
SIGNATURE_CHAIN_NAME=qwen-signature-confirmation-v2 \
bash scripts/revision_run_signature_confirmation_v1.sh
```

The companion read-only figure postprocessor must be bound to the same chain:

```bash
SIGNATURE_CHAIN_NAME=qwen-signature-confirmation-v2 \
SIGNATURE_FIGURE_CHAIN_NAME=qwen-signature-figure-postprocess-v2 \
bash scripts/revision_run_signature_figure_postprocess_v1.sh
```
