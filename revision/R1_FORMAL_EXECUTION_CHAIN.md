# R1 Official-Context Formal Execution Chain

This is the executable counterpart of
`revision/R1_FORMAL_COMPARISON_PROTOCOL.md`.  It is deliberately separate from
the Qwen 512-token complete-family experiment, because R1-Distill requires the
frozen 32,768-token sampled envelope.

## Entry gate

`scripts/revision_run_r1_formal_comparison_chain.sh` waits for
`r1-official-context-chain-v2/DONE`.  The parent chain can create that marker
only after the correctness-blind 20-problem readiness audit, the 100-problem
structural calibration audit, and the source/target vector registry succeed.
A parent failure is terminal; the formal chain does not select a new prompt,
source subset, decoder, seed, layer, alpha, or model.

## Declared formal family

The R1 validation family consists of seven R1-admissible direction methods:

1. source CrossSteer;
2. target-calibrated direction;
3. CAA prompt-final direction;
4. ActAdd prompt-final direction;
5. fixed-seed matched-norm random direction;
6. exact negative source direction;
7. 10% coordinate-sparse CAA ablation.

Each receives the same GSM8K IDs 100--199, four relative layers, four
RMS-scaled alphas, `decode_last` hook, 32,768-token budget, and fixed sampled
decoder (`temperature=0.6`, `top_p=0.95`, problem-indexed seed).  The
SAE-space baseline is not falsely adapted to R1: the approved compact replay
representation stores pooled generated-state means, while the SAE
implementation requires time-indexed trajectories.  The full SAE comparison is
therefore conducted in the Qwen primary table, exactly as stated in the R1
formal protocol.

## Selection and locked execution

Only after all seven validation directories write `DONE`, the selector compares
byte-equivalent baselines and normalized generation metadata, applies the
predeclared behavior guard, and writes one immutable selection artifact.  Every
behavior-eligible method is then run once on IDs 200--299 under the copied
32k sampled decoder.  If no method is eligible, the chain writes
`NO_ELIGIBLE_METHODS` and never manufactures a replacement cell.

The locked report contains all eligible methods, paired accuracy deltas,
bootstrap confidence intervals, exact and Holm-adjusted tests, repairs, breaks,
length, repetition, truncation, and stop telemetry.  It is not a new selection
stage.

## Downstream long-context safety suite

`scripts/revision_run_r1_long_context_chain.sh` waits for this formal chain and
uses the same immutable R1 selection artifact.  Only the predeclared
`crosssteer_source` and `target_calibrated` variants can enter; each is evaluated
at 4,096 and 32,768 maximum new tokens under constant, prefix, exponential, and
relative-hidden-RMS schedules.  It propagates the selected 32k sampled decoder
and problem-indexed seed rule through every run.  It cannot change the direction,
layer, alpha, prompt, source IDs, or decoder, and it writes an explicit
no-eligible marker if neither trajectory-direction variant passes the frozen
behavior gate.

## Server isolation

The chain exposes physical GPU 1 as logical `cuda:0`.  Its only scheduling
gate is that physical GPU 1 must be idle; it is not conditioned on any Qwen
result or Qwen-chain terminal marker.  Thus a superseded Qwen decoding failure
cannot suppress the independent R1 experiment.  It should be launched from a
clean worktree containing the decoder-preservation implementation commit
`6cac1da` or a descendant; an active Qwen worktree must not be pulled or
modified while it is generating frozen cells.
