# R1-Distill 32k readiness-stop evidence

This directory records the frozen R1 official-context readiness boundary that
the revised manuscript and response letter cite.

- `capability20-readiness.json` is the score-blind audit output for the
  predeclared 20-problem readiness set (`protocol
  tacl-11241-official-context-pooled-readiness-v1`).  It records
  `severe_repetition_rate = 0.100` (2/20) against the preregistered 0.050
  ceiling, the config/labels hashes, and the expected problem IDs.
- `STOPPED_BY_R1_READINESS_GATE` is the chain marker written by the formal
  R1 chain when the stop rule triggered; it prohibits any R1 calibration,
  vector, validation, locked, or long-context steering result.

Regeneration uses the same interface as the completed run:

```bash
configs/revision_gsm8k_r1_official32k_capability20.yaml
scripts/revision_audit_pooled_readiness.py
revision/R1_OFFICIAL_CONTEXT_PROTOCOL.md
revision/R1_FORMAL_EXECUTION_CHAIN.md
```

These artifacts are retrospective stop-rule evidence, not a steering result:
no R1 direction, layer, alpha, sign, or score was selected from them.
