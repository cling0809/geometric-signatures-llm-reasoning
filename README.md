# GeoProbe: Geometric Signatures of Post-Training in LLM Reasoning Trajectories

Public reproducibility package for the accepted TACL paper *Geometric
Signatures of Post-Training in LLM Reasoning Trajectories: Separating
Observability from Transferability* (TACL submission 11241, accepted
7 October 2026).

Authors: Tianlin Chen and Yiting Cai, Guangdong University of Technology.

The package contains code, configurations, evaluator versions, content-free
per-problem tables, tests, and a CPU smoke test. It omits model weights,
benchmark text, and raw hidden states.

It studies two deliberately separated questions:

1. **Diagnostic analysis.**  Under a fixed extraction protocol, do
   post-trained Qwen-family checkpoints show different layer-wise associations
   between hidden-state trajectory metrics and answer correctness?
2. **Inference-time intervention.**  Can a source-derived trajectory direction
   improve a target model under a frozen, leakage-free evaluation protocol—and
   does it beat target-calibrated, CAA, ActAdd, SAE-space sparse activation, coordinate-sparse, sign-reversed,
   and matched-random perturbation controls?

The repository does **not** treat exploratory sweeps, retrospective plots, or
uncontrolled historical runs as final performance evidence.  Reported revision
claims must come from the versioned validation, locked-test, OOD, and
long-context artifacts described in [`revision/`](revision/).

> **Status.** The frozen revision runs and downstream audits are complete.
> The CPU smoke test below verifies installation and audit plumbing only; it
> does not reproduce a paper result.  Scientific numbers must be taken from
> the versioned evidence manifests under `revision/evidence/`.

## Quick start (CPU smoke test)

Python 3.10+ is required.  The smoke test requires PyTorch but no model
checkpoint, dataset download, GPU, or network access.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[torch,dev]"
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python scripts/revision_smoke_demo.py \
  --out /tmp/geoprobe-revision-smoke
cat /tmp/geoprobe-revision-smoke/SMOKE_OK.json
```

A successful smoke test writes two synthetic calibration runs and a compact
train-only vector registry.  The JSON must identify protocol
`tacl-11241-cpu-smoke-v1` and vectors of shape `[3, 5]`.  These files are
synthetic installation artifacts, not scientific evidence.

For the symbolic MATH-500 evaluator used in the formal OOD protocol, install
`.[revision]` as well:

```bash
python -m pip install -e ".[torch,revision,dev]"
```

## Reproducing the formal protocol

The formal protocol is intentionally split so that validation cannot leak into
locked or OOD results:

| Role | GSM8K IDs | Allowed purpose |
|---|---:|---|
| Source training | 0–99 | Build direction/vector artifacts only |
| Validation | 100–199 | Choose one eligible layer/alpha per method |
| Locked in-domain test | 200–299 | One-shot evaluation after selection freezes |
| OOD | all 500 MATH-500 + all 1,000 SVAMP | Frozen selected configurations only |

The primary evaluation uses greedy decoding with a declared 512-token budget;
long-context policies are a separate frozen study.  Every formal vector must
have a sibling provenance manifest, and every grid records prompts, model and
tokenizer identifiers, decoded completions, evaluator decisions, behavior
telemetry, configuration hashes, and paired statistics.

Read these documents before launching a full run:

- [`revision/VALIDATION_GRID_V1.md`](revision/VALIDATION_GRID_V1.md) — fixed
  layer/alpha grid and behavior guard.
- [`revision/BASELINE_PROTOCOL.md`](revision/BASELINE_PROTOCOL.md) —
  CrossSteer, target-calibrated, CAA, ActAdd, SAE-space sparse activation, coordinate-sparse, sign-reversed,
  and matched-norm random controls at the same perturbation budget.
- [`revision/OOD_PROTOCOL.md`](revision/OOD_PROTOCOL.md) — all-500 symbolic
  MATH-500 plus all-1,000 hash-pinned SVAMP evaluation after frozen in-domain
  selection.
- [`revision/LONG_CONTEXT_PROTOCOL.md`](revision/LONG_CONTEXT_PROTOCOL.md) —
  4k/32k safety, repetition, truncation, and mitigation measurements.
- [`revision/REPRODUCIBILITY.md`](revision/REPRODUCIBILITY.md) — exact
  artifact/provenance requirements and release scope.

The entry points are deliberately fail-closed:

```text
scripts/revision_build_vector_registry.py
scripts/revision_steering_validation_grid.py
scripts/revision_select_validation.py
scripts/revision_launch_locked_selected.py
scripts/revision_launch_ood_selected.py
scripts/revision_launch_long_context.py
```

In particular, `revision_select_validation.py` records a one-time validation
selection and refuses to overwrite a different selection.  Locked runs refuse
source/locked ID overlap and configuration changes.  Do **not** add a new
layer, sign, alpha, schedule, seed, or metric after reading locked labels.

## Repository map

```text
src/geoprobe/revision/   protocol, vector provenance, behavior/stats, gates
scripts/revision_*.py    audited CLI entry points and report builders
configs/revision_*.yaml  resolved calibration/contrast-pool configuration
revision/                protocols and content-free evidence tables
tests/                   CPU unit and regression tests
```

This repository does not ship model weights, saved steering vectors, raw
hidden-state tensors, or benchmark text whose license forbids redistribution.
It includes report and provenance manifests, hash-verified content-free
per-problem correctness/behavior tables, exact scripts, configs, test
instructions, and a clear list of external model/dataset prerequisites.

## Claim boundaries

- A diagnostic plot or a correctness-separation score is **not** proof of a
  universal post-training taxonomy or causal mechanism.
- A validation win is **not** a locked-test result.
- An accuracy change without paired confidence intervals, behavior telemetry,
  and stronger steering/random controls is **not** a final CrossSteer claim.
- GeoVote's retrospective length audit is a negative-control result unless a
  new locked protocol demonstrates utility beyond token length and majority
  voting.

Protocol boundaries are in [`revision/REPRODUCIBILITY.md`](revision/REPRODUCIBILITY.md).
