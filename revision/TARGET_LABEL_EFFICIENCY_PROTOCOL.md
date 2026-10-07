# Frozen Target-Label Efficiency Protocol V1

This analysis addresses Reviewer B's question: if target-side labels are
available, when is a source direction useful relative to a target-calibrated
direction?

## Fixed conditions

- Source target-calibration universe: GSM8K IDs 0--99 only.
- Label budgets: `0, 5, 10, 20, 50, 100`.
- Budget `0` is the already frozen external-source CrossSteer direction; it uses
  no target calibration trajectory.
- Budgets `5--100` use target trajectories only.  The target-model calibration
  run, selected layer, alpha, schedule, prompt, target model, decoder, and
  locked IDs remain fixed from the full target-calibrated method selection.
- The locked evaluation remains IDs 200--299.  No target-label-budget curve can
  alter layer, alpha, sign, schedule, or evaluator.

## Deterministic subset rule

For each nonzero budget, the builder uses a SHA-256 ordering of calibration
problem IDs separately within the correct and incorrect classes.  It takes an
approximately balanced prefix, filling any class shortage from the other class.
The only value used for stratification is the calibration correctness label;
no geometry metric, validation score, target accuracy, or locked outcome can
select IDs.  Budgets are nested and every chosen ID is stored in a manifest.

## Reporting

Report each budget's class counts, exact IDs/hash, accuracy, paired delta, CI,
repair/break counts, behavior fields, and the full 0-budget source-transfer
reference.  This is a label-efficiency curve, not a seed search.  If a smaller
budget cannot contain both correctness classes, the corresponding point is
reported as infeasible rather than imputed.

## Locked execution guard

The locked curve is launched only after the full `target_calibrated` validation
selection is frozen.  A dedicated launcher checks that every nonzero budget
vector was built from the *same target-calibration run and labels* as that
selected full-label vector.  It evaluates every point on IDs 200--299 with the
selected target-calibrated layer, alpha, schedule, decoder and intervention
mode.  It never reuses the source method's selected layer or alpha for the
curve.

The `k=0` point is the pre-built external-source CrossSteer vector under those
same target-selected intervention conditions.  Every point is evaluated in a
separate run so that the grid runner can enforce source-ID non-overlap and
preserve a separately regenerated baseline for its paired comparison.  The
final report must include all feasible budgets and the `k=0` reference; it may
not retain only a favorable budget.
