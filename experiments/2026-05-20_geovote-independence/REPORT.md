# GeoVote Independence Analysis

Date: 2026-05-20

Purpose: test whether the geometric vote carries information beyond logprob and
answer frequency under the confirmed paper mainline.

## Inputs

| Run | Dataset | Budget | N | Config |
|---|---|---:|---:|---|
| `2026-05-19_qwen-math-gsm8k-2048-n8` | GSM8K-100 | 2048 | 8 | high-budget GeoVote snapshot |
| `2026-05-19_qwen-math-math500-2048-n8` | MATH-500 first 100 | 2048 | 8 | high-budget GeoVote snapshot |
| `2026-05-17_c1-qwen-math-sample16` | GSM8K-100 | 512 | 8/16 prefixes | original GeoVote N-scaling run |

## Main Results

| Dataset / run | Majority | Logprob weighted | GeoVote fixed | Best geo sweep | Oracle |
|---|---:|---:|---:|---:|---:|
| GSM8K 2048 N=8 | 0.90 | 0.83 | 0.87 (`mean_step_norm@L20`) | 0.92 (`trajectory_length@L4`, combined log) | 0.97 |
| MATH-500 2048 N=8 | 0.76 | 0.72 | 0.77 (`mean_step_norm@L20`) | 0.79 (`curvature_max@L8`) | 0.87 |
| GSM8K 512 N=16 | 0.92 | 0.84 | 0.90 (`mean_step_norm@L20`) | not reswept here | 0.99 |

## Interpretation

- GeoVote is consistently stronger than log-probability selection on the matched
  sample pool.
- On MATH-500, geometry also beats majority in the current high-budget N=8 run:
  0.79 vs 0.76 after metric/layer sweep.
- On GSM8K, majority is near ceiling, so the clean claim is complementarity and
  near-ceiling recovery rather than universal majority beating.
- The useful signal is answer-candidate level aggregation. Raw per-sample
  geometry is weak as a direct correctness classifier, but summed geometric
  weight over candidate answers is competitive with answer frequency and
  logprob.

## Margin-Stratified Takeaway

MATH-500 low-margin questions are where geometry matters most. In the fixed
2048/N=8 analysis, low-margin MATH-500 questions have:

- Majority: 0.40
- GeoVote: 0.45
- Oracle: 0.70

This supports the paper framing that geometry is useful when self-consistency is
uncertain, while high-consensus questions should preserve majority.

## Artifacts

Each subdirectory contains:

- `accuracy_summary.csv`
- `sample_signal.csv`
- `candidate_signal.csv`
- `margin_strata.csv`
- `complementarity.csv`
- per-question and per-candidate outputs

The metric/layer sweeps are in `sweeps/`.
