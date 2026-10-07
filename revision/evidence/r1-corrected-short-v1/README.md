# Corrected R1 short-envelope study evidence

Frozen evidence for `revision/R1_CORRECTED_SHORT_ENVELOPE_PROTOCOL.md`
(protocol tags `tacl-11241-corrected-short-envelope-calibration-v1` and the
shared `tacl-11241-frozen-steering-grid-v1` / `tacl-11241-frozen-locked-launch-v1`).

- `r1-corrected-selection-v1.json`: frozen validation selection; both arms
  eligible (target_calibrated layer 14 / alpha 0.2; crosssteer_source layer 20
  / alpha 0.2).
- `locked_report_manifest.json`: complete two-arm locked report manifest.
- `locked_target_calibrated_summary.csv` and
  `locked_crosssteer_source_summary.csv`: the locked paired summaries on GSM8K
  IDs 200--299 (baseline 0.41; same-target +3 pp CI [-8,+14] p=0.72;
  cross-source -1 pp CI [-12,+10] p=1.0).

These artifacts are corrected-generation evidence only.  The 32k
misconfigured calibration run (archived on the server as
`.32K_MISCONFIGURED`) plays no role in any result.
