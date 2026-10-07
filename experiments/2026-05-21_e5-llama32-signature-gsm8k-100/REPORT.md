# E5 Llama-3.2-1B Cross-Family Signature Replicate

Date: 2026-05-21

## Purpose

Reviewer risk: the main phenomenon and interventions are mostly demonstrated on
the Qwen family. This replicate tests whether post-training also leaves a
geometric signature in a non-Qwen family.

## Protocol

- Family: Llama-3.2-1B
- Models:
  - `LLM-Research/Llama-3.2-1B`
  - `LLM-Research/Llama-3.2-1B-Instruct`
- Dataset: GSM8K test, first 100 examples
- Decoding: greedy
- Budget: `max_new_tokens=512`
- Extraction: metric-only streaming; no full hidden-state tensors persisted
- Signature: metric x layer AUC for predicting incorrect generations

## Results

| Model | Accuracy | Labels | Metrics |
|---|---:|---:|---:|
| Llama-3.2-1B Base | 0.04 | 100 | 11900 |
| Llama-3.2-1B Instruct | 0.39 | 100 | 11900 |

Pairwise signature distance:

| | Llama-Base | Llama-Instruct |
|---|---:|---:|
| Llama-Base | 0.000 | 2.319 |
| Llama-Instruct | 2.319 | 0.000 |

Artifacts:

- `pairwise_signature_distance.csv`
- `plots/signature_grid.png`
- `plots/pairwise_distance.png`

## Interpretation

This supports the E5 reviewer-defense claim: post-training geometric signatures
are not unique to the Qwen family. The result should be framed as a lightweight
cross-family phenomenon replicate, not as a full cross-family GeoVote or
CrossSteer demonstration.

The Base model's low accuracy is acceptable here because this is a controlled
diagnostic setting rather than an official benchmark setting. The relevant
quantity is the signature separation under matched prompt, decoding, dataset,
and token budget.

## Next Decision

If compute allows, the stronger follow-up is not another small phenomenon run,
but one of:

- GeoVote on Llama-Instruct best-of-N, if the model produces enough correct
  samples for oracle headroom.
- CrossSteer from Llama Base/Instruct direction, if we want causal evidence
  outside Qwen.

For the current paper draft, this run is enough to soften the "only Qwen?"
critique in the phenomenon section or appendix.
