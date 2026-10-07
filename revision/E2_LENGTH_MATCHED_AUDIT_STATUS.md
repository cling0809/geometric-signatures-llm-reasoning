# E2 explicit length-matched control status

## Current status

**Complete and integrated as retrospective control evidence.**

The current GeoVote revision evidence contains:

- exact same-candidate-pool comparison;
- shortest-output length control;
- train-only length residualization;
- candidate-level length correlations;
- paired bootstrap intervals and exact paired tests.

The editor requirement also names an explicit **length-matched control**.  The
control has now been run on the exact retrospective candidate pool and its
summary is mirrored in the evidence bundle.  It is reported as a null
distribution, not as a new selector.

## Implemented audit

- `src/geoprobe/revision/geovote_length_matched.py`
- `scripts/revision_geovote_length_matched_audit.py`
- `tests/test_geovote_length_matched.py`

The control sorts candidates within each problem by `(n_gen_tokens, sample_idx)`
and permutes residualized geometry scores only within adjacent length pairs. It
uses no correctness labels, preserves the candidate pool and score multiset,
and reports a deterministic permutation-null distribution against majority.

## Data/access audit

The existing GeoVote manifest identifies the retrospective source hash as:

```text
5094e1e030cfd5685d0a4024a0176daaf44c3f7c678d3129629d73f46a00aa0f
```

A local archive contains a file with the same SHA-256, but the current local
PyArrow runtime fails to read it with:

```text
OSError: Repetition level histogram size mismatch
```

The local archive was readable with PyArrow 25.0.0; the input hash matched the
existing retrospective manifest.  The numerical result is now inserted only as
a control-null paragraph in Section 5 and mirrored in the evidence bundle.

## Integration rule

The completed run used calibration IDs `0:50`, evaluation IDs `50:100`, seed
`20260806`, and 1,000 permutations.  The input hash, pairing rule, and output
statistics are recorded in `length_matched_permutation_summary.json`; no result
was promoted to a positive utility claim.
