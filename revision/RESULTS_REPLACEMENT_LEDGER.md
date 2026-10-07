# Manuscript Results Replacement Ledger

## Purpose and non-negotiable rule

This ledger prevents a major-revision manuscript from silently retaining an
unreplicated exploratory number.  A legacy result may stay only if its artifact
passes the revision provenance audit **and** its revised claim is no stronger
than the controlled evidence.  Otherwise the result is either (i) replaced by a
new frozen-protocol table, (ii) moved to a clearly labelled exploratory or
retrospective appendix, or (iii) removed.

The submitted paper is preserved in the immutable `paper_submitted_tacl_11241`
directory.  This ledger applies only to the revision worktree.

## Result-by-result disposition

| Submitted location / result | Review risk | Revision disposition | Required replacement evidence | Claim allowed after replacement |
|---|---|---|---|---|
| Abstract: GeoVote beats log-probability on held-out MATH-500 (`00_abstract.tex`) | E2, B2: geometry score may be a length proxy. | **Remove from abstract and main utility claim.** | Existing same-pool audit plus, only if genuinely positive under a fresh locked protocol, a preregistered replacement. | Until then: GeoVote is a diagnostic whose raw confidence was strongly length-correlated; it is not evidence of task improvement. |
| `tab:geovote` clean-split MATH-500 utility table | E2, B2: selected rule / baseline advantage may be due to token count and selection. | **Replaced** by a clearly labelled retrospective GSM8K same-pool control table; removed utility table/claim. | GSM8K same-pool raw/residualized audit, majority, paired CI and exact test. | No utility superiority; GeoVote remains a negative-control diagnostic. |
| `tab:geovote-independence` | Same as above. | **Removed**; no stand-alone “independence” language. | Same-pool controls and length correlation audit. | Negative-control statement only. |
| `tab:phen-distance`, `fig:phen-heatmap`, nearest-neighbor narrative | B1: 1.5B and 7B topology differs. | Keep matrices as matched-protocol descriptive diagnostics; remove taxonomy/fingerprint language.  Move nearest-neighbor topology to appendix. | Problem-level bootstrap at 1.5B and 7B, including distance CIs and nearest-neighbor probabilities. | Distinct metric-layer correctness-association patterns under a fixed protocol; no scale-invariant fingerprint topology. |
| `fig:phen-separation` / projection-based correctness discussion | B5: projection may capture position, length, answer formatting, or visualization artifacts. | Retain only as descriptive visualization, if retained at all. | Matched controls, label permutation / position notes, and no causal or prediction-performance claim from the plot alone. | A visualization of separation in the chosen projection, not a mechanism proof. |
| `tab:crosssteer-headline` same-target no-leakage table | E3a/B3/B5: no fair steering baselines and possible behavior confounds. | **Removed pending replacement**; it cannot support a revised claim. | Frozen validation selection, one-shot locked GSM8K rows for CrossSteer source/target, CAA, ActAdd, sparse coordinate control, random and no-steering, plus repair/break, CI, exact test and behavior diagnostics. | Full-trajectory advantage only in cells that exceed all frozen baselines under matched conditions. |
| `tab:crosssteer-r1-gsm8k`, `fig:crosssteer-headline`, source-quality / layer / alpha narrative | B3: large historical sweep is exploratory; selection and target calibration unclear. | **Removed from revision body/appendix**.  Do not use its peak as a formal headline. | New validation grid must be shown in full; selection artifact fixes method/layer/alpha before locked generation. | A selected configuration and a complete validation landscape, not a post-hoc peak. |
| `tab:holdout-steering-baseline` | E3a: baseline comparison incomplete. | Replace with formal shared-protocol baseline table. | CAA / ActAdd / sparse / random use same target, source-ID budget, validation/locked IDs, alpha/layer budget and RMS norm rule. | Relative method ranking only if locked and behavior-eligible. |
| `tab:crosssteer-cross-task` | E3b: OOD needs frozen hyperparameters, not task-specific sweeps. | Replace after the in-domain method has frozen. | MATH-500 and predeclared fresh math task(s), no OOD layer/alpha/sign/schedule retuning, paired table / manifest. | OOD result limited to frozen in-domain configuration. |
| `tab:crosssteer-cross-target` and compatibility claim | B1/B3: signature orientation may not transfer robustly; target calibration is a relevant alternative. | Keep only as a boundary or supplementary diagnostic if it survives matched controls. | Target-calibrated baseline, source-transfer comparison, target-label efficiency curve, and all failure rows. | Source transfer is a bounded option, not a universal substitute for target calibration. |
| 4096/32k claims in `06_crosssteer.tex` and `app:extended-budget` | E3c/B4: 32k repetition loops are a central failure. | Remove unqualified robustness claim; add a long-context safety table. | Frozen constant/prefix/decay/relative-RMS schedules, accuracy, repetition rate, truncation, stop reason, length, distinct-n at 512/4096/32768. | Bounded operating regime; a mitigation is not accepted if it simply lowers accuracy below no-steering. |
| Random control table in `app:random-control-k20` | Need comparable direction construction and full seed reporting. | Retain only after source construction, norm matching and random-seed policy are auditable. | Fixed direction distribution, every seed, mean/SD, and paired comparison. | Direction specificity only if source direction exceeds matched random controls. |

## Main-body result table plan after all formal runs

The revised main text must contain only the following result-bearing tables:

1. **Diagnostic stability table** — two scales, bootstrap uncertainty, explicitly
   marked descriptive; no nearest-neighbor fingerprint conclusion.
2. **GeoVote control table** — same-pool length/logprob controls and its
   conclusion (including a negative conclusion if that is what the audit shows).
3. **Formal in-domain steering table** — no steering, random, CrossSteer source,
   target-calibrated direction, CAA, ActAdd and sparse control; locked paired
   deltas, CIs, exact tests, repair/break and behavior fields.
4. **Frozen OOD table** — only if the selected in-domain method is eligible;
   otherwise a concise feasibility/failure report rather than an invented
   generalization claim.
5. **Long-context behavior table** — all schedules and safety metrics, including
   the submitted constant-steering failure.

Every number in these tables must link to a completion manifest and a resolved
configuration.  Exploratory historical numbers can appear only in an appendix
with their exact status; they may not be used to support the abstract,
contributions, or conclusion.

## Publication gates

- **Strong revision:** a locked CrossSteer configuration beats matched baselines
  with a positive paired CI, while behavior controls and OOD/long-context
  evidence support the bounded claim.
- **Narrow revision:** controlled results establish a limited target/source
  behavior effect but do not beat baselines.  Remove the method-superiority
  claim; retain only a clearly labelled diagnostic contribution if the Action
  Editor considers that sufficient.
- **Insufficient method evidence:** if CrossSteer has no credible locked
  advantage or exhibits unrepaired harmful behavior, remove its utility claim
  rather than selecting an historical peak.  The response letter must state the
  correction directly.
