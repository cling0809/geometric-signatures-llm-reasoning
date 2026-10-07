# CrossSteer Source-Calibration Quality Gate

## Why this corrective run is necessary

The original Qwen-Math source calibration (`GSM8K IDs 0--99`, greedy,
512 new tokens) generated at the token budget on 36 of 100 problems.  Its
formal audit had been relaxed to a 50% ceiling, but the revision's statistical
plan requires a generation-saturation gate before a direction is built.  The
rate is also outcome-skewed: 14/22 incorrect source completions versus 22/78
correct completions reached the 512-token boundary.  A direction derived from
that mixture could encode the source's cutoff behavior rather than a complete
reasoning contrast.

The existing 1,700-row `crosssteer_source` validation grid is therefore
**superseded for selection and paper claims**.  The contemporaneous
`target_calibrated` grid is also superseded: although its 512-token target
calibration has a low budget-hit rate, that legacy calibration directory was
created before pre-generation EOS/provenance manifests were required.  It may
not be retroactively certified.  The three contrast-pool controls (CAA, ActAdd, and sparse CAA) remain useful
because their target decoding configuration and source-pool provenance are
separately audited.  SAE sparse activation is also superseded: its dictionary
training states came from the legacy target calibration, which lacks the new
pre-generation EOS/stop-telemetry manifest.  The matched-norm random and
sign-reversed controls are superseded because they are constructed directly
from the old CrossSteer source vector and must be rebuilt from its audited
replacement.

## One-shot correction

- Use exactly GSM8K IDs 0--99, Qwen2.5-Math-1.5B-Instruct, the same prompt,
  greedy decoder, seed and trajectory extractor.
- Change only `max_new_tokens` from 512 to **2048**.  This is a fixed half-context
  envelope under the local checkpoint's `max_position_embeddings=4096`, chosen
  before rerunning and not adapted after inspecting a new result.
- The extractor must save its pre-generation multi-EOS envelope and per-row
  stop/truncation telemetry.
- Run the provenance-aware calibration audit with a 25% generated-token
  budget-hit ceiling and at least 10 examples from each correctness class.
- Re-extract the target-local Qwen-Instruct calibration on the same source IDs
  with its fixed 512-token envelope solely to obtain symmetric pre-generation
  EOS and stop telemetry; it is not a new label budget or a configuration
  search.
- Audit the 2048 source and 512 target runs separately at the same 25% ceiling,
  bind both audit hashes into a new vector registry, and rebuild both
  CrossSteer-source and target-calibrated directions.
- Rebuild the deterministic matched-norm random and exact sign-reversed vectors
  from the new CrossSteer direction, and rebuild the SAE sparse activation from
  the new target calibration plus the unchanged audited contrast pool.
- Rerun only CrossSteer-source, target-calibrated, matched-random,
  sign-reversed and SAE validation grids on the unchanged IDs 100--199,
  layers, alphas, target model and dual-EOS target decoder.  Do not rerun
  independent CAA, ActAdd or sparse-CAA controls merely to alter their results.

## Hard stop

If this single 2048-token source calibration still exceeds the 25% ceiling,
lacks EOS/provenance telemetry, has fewer than 10 correct or incorrect rows, or
otherwise fails its audit, no 4096/longer fallback, prompt change, seed change,
layer scan or alpha scan is allowed.  The Qwen CrossSteer source-transfer claim
is then removed from the revision.  The completed target-method controls remain
reported as controls, not as a rescued CrossSteer comparison.
