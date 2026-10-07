# Frozen Steering Target Matrix V1

The submitted paper's strongest intervention statement concerns the historical
Qwen-Math-to-R1-Distill repair cell.  A short, shared 512-token target protocol
is not a valid capability test for R1-Distill because its long reasoning outputs
are routinely truncated.  The revision therefore keeps the two target roles
separate rather than treating a favorable Qwen control result as an R1 repair.

**Runtime status (2026-08-04 UTC).** The old R1 512-token calibration is retained
only as a budget-saturation diagnostic.  It is neither a model-capability result
nor an intervention result.  The separate 32,768-token official-context readiness
pass has now also stopped Cell~B: two of 20 fixed samples entered severe repetition
loops (10%), exceeding the preregistered 5% ceiling, although budget saturation
was 10% (below the 25% ceiling).  Therefore no R1 calibration, direction,
validation, locked or schedule-mitigation run is admissible.  Cell~A remains the
primary complete-family shared-decoder comparison.

## Roles

| Cell | Target | Role | Required evidence |
|---|---|---|---|
| A | Qwen2.5-1.5B-Instruct | primary complete-family comparison and behavior control | source CrossSteer, target-calibrated, CAA, ActAdd, SAE sparse activation, coordinate-sparse CAA, negative, matched-norm random, no steering |
| B | DeepSeek-R1-Distill-Qwen-1.5B | historical repair target under a model-valid envelope | **Stopped at correctness-blind 20-problem readiness:** 10% severe repetition exceeds the 5% ceiling; no 512-token fallback or mitigation sweep |

Both cells use Qwen2.5-Math-1.5B-Instruct as the external source for a source
CrossSteer direction.  Where target calibration is technically admissible, each
cell also receives its own target-calibrated direction.  Cell~B must not inherit
a Qwen result, layer, alpha, prompt change, or short-budget decoder decision.

## Frozen data separation

The logical data roles are shared across cells:

- GSM8K source calibration: IDs 0--99;
- validation: IDs 100--199;
- locked in-domain test: IDs 200--299.

No source-direction artifact may include validation or locked IDs.  Any locked
comparison is problem-paired with its unsteered baseline, and selection occurs
once before locked generation.

## Target-specific decoding envelopes

| Field | Cell A: Qwen-Instruct complete family | Cell B: R1 official context |
|---|---|---|
| Decoder | greedy | fixed sampling |
| Max new tokens | 512 | 32,768 |
| Sampling | none | temperature 0.6, top-p 0.95 |
| Randomness | n/a | fixed `seed * 100003 + sample_id * 1009`, shared by paired cells |
| Trajectory storage | full generated trajectory | exact generated-state mean via one-token compact replay |
| Physical GPU policy | GPU 0 for later OOD/long-context queue | GPU 1 exposed as logical `cuda:0` |

The two envelopes are deliberately reported separately.  They are not a
matched-token-budget comparison and cannot be pooled into one target average.

## Cell-specific gates

### Cell A: complete-family Qwen gate

All eight planned direction/control families use `decode_last`, absolute
RMS-normalized injection, the same 16-cell layer/alpha grid (layers
8/14/20/24; alphas 0.025/0.05/0.10/0.20), the same behavior guard and one-shot
locked launcher.  The complete locked table reports both behavior-eligible and
negative outcomes; only a locked, behavior-controlled advantage over the
complete family supports a method-superiority statement.

### Cell B: R1 official-context gate

Before any R1 direction, validation, locked, OOD, or long-context conclusion,
the 20-problem official-context readiness pass must verify exact IDs, compact
replay representation, no more than 25% budget saturation, and no more than 5%
severe repetition loops.  It is correctness-blind and cannot select a method,
layer, alpha, sign, or seed.  A technically successful 20-problem pass releases
a 100-problem 32k source/target calibration only.  That calibration must then
pass the structural class-balance and saturation audit before a separately
frozen R1 paired-comparison matrix is launched.

**Observed Cell~B decision (2026-08-04).** The 20 fixed readiness IDs produced
2/20 severe whitespace-4-gram loops (10%), above the frozen 5% ceiling; 2/20
also reached the 32,768-token budget.  The gate therefore stopped the R1 branch
before target calibration or any intervention was run.  The revised manuscript
removes the historical R1 repair claim.  It does not reclassify the old
512-token diagnostic as a model failure or substitute Cell~A as proof of repair.

## Interpretation boundary

- A positive Cell~A result is a Qwen-Instruct control/behavior result, not an
  R1 repair result.
- Cell~B did not pass its entry gate and therefore supplies no intervention,
  calibration, locked or mitigation result.
- The queued R1 4,096/32,768 long-context suite is prohibited by the readiness
  stop rule; it cannot be revived through a prompt, seed, schedule or decoder
  sweep.
- A direction may not claim superiority without the required locked paired
  comparison, behavior analysis, and declared baseline family.
- Any failure cell remains in the evidence ledger; it is not removed through a
  new prompt, seed, layer, sign, or decoding sweep.
