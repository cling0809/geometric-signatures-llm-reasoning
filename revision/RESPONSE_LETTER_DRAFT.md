# Response to the Action Editor and Reviewers — TACL 11241 Major Revision

## Cover response to the Action Editor

We sincerely thank the Action Editor and the reviewers for the conditional B decision and for the careful, technically specific guidance. We treated this revision as an opportunity to test the paper's claims more rigorously. We completed all four mandatory revisions, ran the requested controls under frozen protocols, and rewrote the manuscript to state the strongest conclusions that the resulting evidence supports.

**Original submission and Action Editor.** This response concerns TACL submission 11241, *Geometric Signatures of Post-Training in LLM Reasoning Trajectories and Their Use at Inference Time* (the revised manuscript is retitled *Geometric Signatures of Post-Training in LLM Reasoning Trajectories: Separating Observability from Transferability*). The original Action Editor is Hai Zhao.

The reviews identified a central distinction that was not sufficiently clear in the submitted paper: a measurable hidden-state difference is not automatically a useful selector or a transferable intervention. We agree. The revised paper now separates three evidence levels: (i) **observability**, meaning a reproducible trajectory--correctness association under a fixed protocol; (ii) **candidate-selection utility**, meaning an incremental same-pool gain over answer agreement, length, and likelihood controls; and (iii) **intervention efficacy**, meaning a paired locked gain over fair steering baselines without behavioral degradation.

The revision makes three narrower contributions. First, it provides a unified, length-residualized protocol for measuring metric-by-layer trajectory associations. Second, it introduces an explicit evidence hierarchy that prevents diagnostic separability from being presented as evidence of utility or mechanism. Third, it reports a frozen audit with conventional and sparse steering baselines, paired statistics, behavior telemetry, complete MATH-500 and SVAMP evaluations, long-context schedules, and a reproducibility package prepared for post-acceptance release. Where the requested controls weakened a submitted claim, we removed or narrowed that claim. We believe the resulting paper has value beyond its specific null findings: it gives future work a reusable, baseline-complete template for separating what hidden-state geometry can be measured from what it can actually deliver, and it shows concretely that two attractive hypotheses (geometry-weighted voting and source-direction transfer) do not survive their natural controls at this scale.

At the observability level, the retained retrospective artifacts show different metric-by-layer signature matrices for matched post-training variants under the fixed protocol. Retrospective bootstrap, generation-budget, and scale audits identify which contrasts recur and which fine topological relations remain protocol-sensitive; a separately preregistered fresh confirmation chain did not qualify because at least one required model in every registered cell exceeded the truncation ceiling. At the two downstream levels, the results are negative: GeoVote does not provide reliable same-pool selection utility, and source-derived CrossSteer does not provide a reliable locked or OOD gain. A separate predeclared target-label curve contains one bounded positive cell---five target labels improve locked Qwen/GSM8K accuracy by 11 percentage points with a family-wise adjusted $p=0.0443$---but the complete curve is non-monotone, so we do not present it as a general law or as evidence that source transfer works.

We are grateful for the opportunity to make these boundaries explicit. The revised manuscript does not claim that every geometric association is a mechanism, a universal post-training fingerprint, or an effective intervention. Its conclusion is more limited and, we believe, more reliable: trajectory geometry can be measured under a controlled protocol, but observability, selection, and intervention must be established separately.

**Verification map.** The unified definitions, notation table, curvature intuition, and Algorithms 1--2 are in Section 3 (PDF pp. 4--6). Signature matrices, the budget audit, and the scale audit are in Section 4 (pp. 6--9; Figure 3 for the heatmaps; Table 3 and Figure 4 for the budget audit; the scale audit in the Section 4 prose and Appendix Table 11), with Figure 5 on p. 9. The GeoVote audit is in Section 5 (pp. 9--10; Table 4). Locked steering, target-label, OOD, and long-context results are in Section 6 (pp. 10--12; Tables 5--6 and Figure 6), with detailed OOD and long-context tables in the Appendix, Tables 8--10 (p. 17). Discussion and Limitations are on p. 12. These locations refer to the final 17-page manuscript PDF.

**How to read this response.** For every substantive editor/reviewer comment, we give a direct answer, report the resulting evidence, and identify the exact revised pages, sections, tables, or figures. Where a change alters a central claim boundary, we also quote representative revised wording below; overlapping comments point to the same passage rather than repeating it in full.

## Summary of the revision

The decision letter for this submission was re-sent on August 12, 2026. In direct response to the four mandatory revisions exactly as stated there, we made the following changes:

- **(1) Justify scale-dependent signature stability.** We added 1,000-resample problem-level bootstrap audits at 1.5B and 7B and a retrospective generation-budget sensitivity comparison. The manuscript no longer claims a universal fingerprint or stable taxonomy; it reports only protocol-bounded observability, and the fine nearest-neighbor topology is presented as a stability diagnostic.
- **(2) Add steering baselines (CAA/ActAdd, sparse activation steering) and statistical tests to decouple geometric gains from length heuristics.** For GeoVote we ran same-pool majority, shortest-output, residualized, likelihood, and 1,000-permutation length-matched controls with paired intervals and exact tests, and withdrew the utility claim. For steering we added a locked eight-method family (CrossSteer, target calibration, CAA, ActAdd, SAE-sparse, coordinate-sparse, sign-reversed, matched-random) against the shared no-steering baseline, with paired intervals and Holm correction; no reliable advantage survives.
- **(3) Expand OOD generalization analysis and address CrossSteer failure modes.** We evaluated the complete frozen family on all 500 MATH-500 and all 1,000 SVAMP items, added a predeclared 4,096-token MATH-500 budget audit (baseline truncation falls to 3.2%; every paired interval crosses zero), audited Qwen at 4,096/32,768 tokens under four schedules with behavior telemetry, stopped the R1 official-context branch by its preregistered readiness gate, and ran a predeclared corrected R1 short-envelope rerun (also null).
- **(4) Consolidate methods and improve clarity with intuitive explanations and unambiguous notation.** We unified trajectory, signature, GeoVote, and CrossSteer definitions in Section 3 with a notation table, pen-path and curvature intuition, and Algorithms 1--2; expanded names at first use; streamlined notation; and reorganized the paper around observability, selection utility, and intervention efficacy, in a direct report-first style.

We also strengthened reproducibility with resolved configurations, compact evidence manifests, per-problem audit tables, a CPU smoke test, and 149 unit tests.

## Response to the Action Editor

### AE-1. Scale-dependent signature stability

> **Mandatory revision (1).** "justify scale-dependent signature stability;"

**Response.** The submitted wording could be read as claiming a scale-invariant taxonomy. We therefore performed problem-level bootstrap audits at 1.5B and 7B (1,000 resamples each) and a retrospective generation-budget sensitivity comparison. The audits show that some metric--layer associations recur, while fine nearest-neighbor topology changes with scale and sample size (Section 4, pp. 6--9; Table 3 and Figure 4 for the budget audit; the scale audit is reported in the Section 4 prose and Appendix Table 11).

**Result.** We removed the unqualified terms *fingerprint* and *stable taxonomy*. The revised manuscript claims only that matched Qwen-derived checkpoints exhibit distinct metric--layer correctness-association patterns under the specified extraction protocol, and it presents the nearest-neighbor topology as a stability diagnostic rather than a model-selection or steering rule. Two provenance boundaries are stated in the paper rather than hidden: the 2048-token comparison is a retrospective sensitivity check on legacy artifacts without full decoder/replay provenance, and a separately preregistered fresh confirmation chain (Qwen 512/2048, Gemma 512/2048) did not enter confirmatory analysis because at least one required model in each cell exceeded the predeclared 25% truncation ceiling; the submitted matrices are therefore retained as retrospective, protocol-bounded descriptive evidence.

**Representative revised wording.** “We therefore treat the distance matrix as a protocol-specific descriptive summary rather than a protocol-invariant post-training fingerprint.”

**Evidence and manuscript changes.** Section 4 (pp. 6--9) reports the scale and budget audits (Table 3 and Figure 4 for the budget audit; the scale audit in the Section 4 prose and Appendix Table 11); the Abstract, Introduction, and Discussion state the narrower boundary.

### AE-2. Steering baselines, length controls, and statistical tests

> **Mandatory revision (2).** "add steering baselines (CAA/ActAdd, sparse activation steering) and statistical tests to decouple geometric gains from length heuristics;"

**Response.** We addressed the length confound at both levels of the paper. For GeoVote, we ran an auditable same-pool control on 50 held-out GSM8K questions with the same $K=8$ candidates for every selector, comparing majority voting, shortest-output, length-residualized geometry, log-probability-weighted voting, and an explicit 1,000-permutation length-matched null (raw geometry confidence correlates 0.742--0.830 with generated length). The requested controls do not support a reliable GeoVote selection gain, so the submitted utility claim was removed and GeoVote is retained only as a controlled diagnostic result. For steering, we added CAA, ActAdd, SAE sparse activation, coordinate-sparse CAA, sign-reversed CrossSteer, matched-norm random, and target calibration under one frozen, locked protocol with paired intervals, exact tests, and Holm correction (Section 6, Table 6; Figure 6); no locked row shows a reliable gain.

**Representative revised wording.** “Accordingly, GeoVote is not supported as a reliable inference-time selector in this setting.”

**Evidence and manuscript changes.** GeoVote is in Section 5 (pp. 9--10), Table 4, and the Appendix; the locked intervention family is in Section 6 (pp. 10--12), Tables 5--6, and Figure 6; the predeclared statistical and behavior protocols are documented in Section 3 and the Appendix.

### AE-3. OOD generalization and CrossSteer failure modes

> **Mandatory revision (3).** "expand OOD generalization analysis and address CrossSteer failure modes;"

**Response.** We completed the requested OOD and failure-mode audits without retuning on the outcome datasets. The complete frozen family was evaluated on all 500 MATH-500 questions and all 1,000 SVAMP items with pinned evaluators. A predeclared 4,096-token MATH-500 budget audit reduces baseline truncation from 45.8% to 3.2% and still finds every paired interval including zero (Appendix Table 9). Qwen was audited at 4,096/32,768 tokens under four frozen schedules with behavior telemetry; no schedule yields a reliable gain. For R1-Distill, the official-context readiness audit crossed the preregistered repetition-loop threshold, so that branch was stopped by rule; a separate predeclared corrected short-envelope rerun (greedy 512, corrected EOS) is also null on the locked split (same-target +3 pp, CI [-8,+14]; cross-source -1 pp, CI [-12,+10]).

**Result.** No locked, OOD, budget-audit, long-context, or corrected-R1 comparison gives a reliable source-direction gain. The submitted generalization and robustness claims were removed, and the failure modes are reported directly rather than rescued post hoc.

**Representative revised wording.** “Across locked GSM8K, two OOD datasets, and long-context schedules, the source direction has no reliable gain.”

**Evidence and manuscript changes.** Results are in Section 6 (pp. 10--12), Figure 6, and Appendix Tables 8--10 (p. 17); the R1 readiness rule, the corrected R1 evidence, and the 4,096-token audit are documented in Section 6 and Appendix Tables 8--10.

### AE-4. Consolidated methods and clearer exposition

> **Mandatory revision (4).** "consolidate methods and improve clarity with intuitive explanations and unambiguous notation. Please submit a revised manuscript with a point-by-point response to all reviewer comments."

**Response.** The submitted methods were distributed across the GeoVote and CrossSteer results sections, and the formal presentation required too much reconstruction by the reader. We moved trajectory extraction, signature construction, GeoVote, CrossSteer, baselines, data splits, selection rules, and statistics into the unified framework section (Section 3), added a notation table, explained the trajectory as a pen path through hidden-state space with a curvature sketch, added Algorithms 1--2, expanded names at first use, replaced overloaded variables with explicit answer and trajectory notation, and reorganized the paper around observability, selection utility, and intervention efficacy.

**Representative revised wording.** “A visible geometric difference does not automatically pass the latter two tests; our claims therefore follow the paired, locked comparisons.”

**Evidence and manuscript changes.** Section 3 (pp. 4--6), Figure 2, Table 1, and Algorithms 1--2; the Abstract, Introduction, Results, and Discussion were rewritten in a direct report-first style, and the present letter is the requested point-by-point response to the Action Editor and to Reviewers A--C.

## Response to Reviewer A

We thank Reviewer A for recognizing the novelty of the trajectory analysis and for identifying the manuscript-organization problem directly.

### A-1. Consolidate the fragmented methods

> **Reviewer comment.** “The organization of the paper can be significantly improved. The content of the methods section can be consolidated into a single chapter, e.g., Section 5.1 and Section 6.1 can be integrated into a section before the experiment.”

**Response.** We followed this suggestion directly. The submitted methods were distributed across the GeoVote and CrossSteer results sections, which made it difficult to determine the protocol before seeing the outcomes. We moved trajectory extraction, signature construction, GeoVote, CrossSteer, baselines, data splits, selection rules, and statistics into the unified framework section before the experiments.

**Changes in the manuscript.** Section 3 (pp. 4--6) now contains the notation table, path and curvature intuition, the complete protocol, and Algorithms 1--2. Sections 5 and 6 retain task-specific evaluation and results rather than reintroducing method definitions. We also removed superseded exploratory method prose from the active revision package.

## Response to Reviewer B

We thank Reviewer B for the detailed technical review. The questions about scale stability, length confounding, target calibration, conventional baselines, repetition, behavior, and notation directly shaped the revised experiments and the final claim boundaries.

### B-1. Stability across model scale and sample size

> **Reviewer comment.** “How can these signatures be considered stable “fingerprints” of post-training paradigms if their relative topological relationships do not preserve across model scales?”

**Response.** The original *fingerprint* language was too strong. We performed problem-level bootstrap audits at both scales. At 1.5B, Base's nearest neighbor is Instruct in 0.631 of bootstrap resamples, while R1's nearest neighbor is Base in 0.460 and Instruct in 0.379. At 7B, Base--R1 and R1--Base are stable (0.993 and 0.995), whereas Instruct--Base (0.506) and Math--Base (0.512) are not. The submitted n=300 slice observation belongs to the same sensitivity class: the revision replaces single-slice topology with the 1,000-resample problem-level bootstrap, which quantifies sample-size sensitivity directly at each scale. These results do not support a scale-invariant nearest-neighbor taxonomy.

**Result.** We retain the narrower claim that Qwen-derived checkpoints show distinct metric--layer correctness-association patterns under a fixed extraction protocol. We do not call this a universal fingerprint, and we do not use nearest-neighbor topology as a model-selection or steering rule.

**Changes in the manuscript.** Section 4 (pp. 6--9) reports the scale and budget audits (Table 3 and Figure 4 for the budget audit; the scale audit in the Section 4 prose and Appendix Table 11); the Abstract, Introduction, and Discussion now state the narrower boundary.

### B-2. GeoVote versus completion length and significance

> **Reviewer comment.** “Does GeoVote provide any practical or conceptual utility over a trivial completion-length heuristic? Can you provide an experimental setting where the trajectory metrics provide a statistically significant improvement over a simple token-count heuristic?”

**Response.** The submitted headline result used a MATH-500 $N=16$ pool and a frozen \textsc{TrajLen}@L4 rule. That original pool is not presented as a newly confirmed result in the revision. Instead, we use a retained, auditable submitted-era GSM8K pool as a replacement retrospective same-pool audit: 50 held-out questions, a hash-pinned 2,048-token run, and the same $K=8$ candidates for every selector. This replacement audit tests whether the retained historical trajectory score survives answer-agreement and length controls; it is not a direct re-replication of the submitted MATH-500 $N=16$ result. The frozen retained score for this GSM8K pool is \textsc{StepNorm} at layer 20 with sign $=$ min (smaller \textsc{StepNorm} receives larger vote weight). \textsc{StepNorm} is the per-step mean of \textsc{TrajLen}, but the frozen layer is 20 rather than the submitted layer 4, so the audited row is this pool's retained score rather than a re-run or length-controlled replica of \textsc{TrajLen}@L4. The historical vote weights each candidate by $(\mu+\varepsilon)^{-1}$ and returns the parsed answer with the largest total weight, and the residualized row re-runs the weighted vote with softmax weights over the residuals (larger residualized confidence, i.e.\ smaller length-adjusted \textsc{StepNorm}, receives larger weight). Metric, layer, sign, parser, and candidate pool were fixed before comparison. Raw geometry confidence correlated strongly with generated length (Spearman 0.742 on evaluation and 0.830 on calibration). Under the frozen comparisons, GeoVote reached 0.90 versus 0.94 majority (delta -4 pp, 95% CI [-10,0], exact paired $p=0.50$); shortest-output and log-length-residualized GeoVote both reached 0.92 (delta -2 pp, CI [-10,+4], $p=1.0$). Across 1,000 predeclared length-matched permutations, every mean delta was negative. The shortest-output row is the direct token-count-only control. We did not add a separate length-weighted vote because weighting votes by completion length would introduce another length heuristic rather than an independent control; the residualized and length-matched analyses test whether geometry contributes beyond that heuristic.

**Direct answer.** No. Under the requested controls, we did not find statistically reliable incremental utility over majority or token length. We therefore removed the submitted GeoVote performance claim and retain GeoVote only as a controlled diagnostic result.

**Changes in the manuscript.** Section 5 (pp. 9--10), Table 4, and the Appendix now report the complete same-pool controls, paired intervals, exact tests, and length-matched null. The paper explicitly states that an association does not imply selection utility.

### B-3. Source transfer, target calibration, and conventional steering baselines

> **Reviewer comment.** “Since cross-model transfer requires calibration data on a source model and a geometric signature from a target model to determine orientation, what is the realistic use case for CrossSteer over simply using the target's calibration data to run target-calibrated steering? Furthermore, can you justify why standard CAA/ActAdd (mentioned in Limitations) benchmarks were omitted, and how CrossSteer compares to them in performance and steering efficiency?”

**Response.** Target-calibrated steering is now treated as a direct baseline rather than a secondary comparison. We added a complete target-label curve at budgets 0/5/10/20/50/100 and evaluated eight intervention methods (CrossSteer, target calibration, CAA, ActAdd, SAE sparse activation, coordinate-sparse CAA, sign-reversed CrossSteer, matched-norm random) against the shared no-steering baseline under one frozen protocol.

The locked Qwen/GSM8K baseline is 0.62. Source CrossSteer reaches 0.61 (-1 pp, CI [-6,+4]); CAA reaches 0.70 (+8 pp, CI [0,+16]); SAE sparse reaches 0.69 (+7 pp, CI [0,+14]); and no family-wise comparison is significant. In the separately frozen target-label curve, the zero-label source row reaches 0.68 (+6 pp, CI [-1,+13]), while the five-label target-local row reaches 0.73 (+11 pp, CI [+4,+18], Holm-adjusted $p=0.0443$). The remaining curve is non-monotone and inconclusive. The two source numbers differ by configuration, not by a data inconsistency: 0.61 is the locked row with CrossSteer's own validation-selected layer 14 / alpha 0.025, whereas 0.68 applies the same frozen source vector at the target-calibrated layer 20 / alpha 0.1 used by the label-curve family.

**Direct answer.** The revision does not establish CrossSteer superiority, source-transfer efficacy, or a general label-efficiency law. The realistic use case is now framed only as a test of whether source-derived trajectory information transfers without target labels; in this study, that test is negative. We disclose the additional offline source-trajectory collection cost and distinguish it from the shared one-vector online injection cost. We do not present the five-label cell as evidence that source transfer works.

**Changes in the manuscript.** Section 6 (pp. 10--12), Tables 5--6, and Figure 6 report the complete locked family, target-label curve, paired intervals, and behavior controls. Appendix Table 7 provides the direction-construction/online-overhead accounting; Related Work distinguishes inference-time baselines from methods requiring target-specific optimization.

### B-4. Long-context repetition loops and safety

> **Reviewer comment.** “In Appendix E, you note that when extending the token budget from 4096 tokens to a 32k greedy budget on MATH-500, CrossSteer fails completely, dropping target performance from 0.42 to 0.33 due to the target model entering repetition loops. What does this tell us about the safety and stability of CrossSteer? Have you explored any normalization or decay techniques (such as decaying the steering scale over long contexts) to mitigate these repetition loops?”

**Response.** We do not treat the submitted 32k failure as a minor artifact. The official-context R1 readiness audit found severe repetition in 2/20 outputs (10%), above the preregistered 5% ceiling; the stop rule therefore prohibited R1 calibration, steering, and schedule mitigation. For Qwen-Instruct, we completed the predeclared 4,096/32,768-token audit with constant, prefix-256, exponential-1024, and relative-hidden-RMS schedules. Source CrossSteer remained 61/61/61/63% against a shared 62% baseline at both budgets, with all paired intervals including zero and Holm-adjusted $p=1.0$. Two source schedules added more than 300 tokens at 32k without improving accuracy. One qualification is stated in the paper: mean Qwen completion lengths stayed far below the caps (largest cell mean approximately 614 tokens at 32k; 4,096-token truncation rates 0--1%), so neither cap was ever binding and the 4k/32k rows are numerically identical; the Qwen rows therefore test schedule invariance at a short effective length rather than a bound long context. The model whose 32k loops motivated the question, R1-Distill, is the one that stopped by rule.

**Direct answer.** We tested normalization and decay schedules only under the frozen predeclared rule and found no reliable rescue. Because R1 failed the capability gate before intervention, we did not manufacture an R1 mitigation result after observing the failure. A separate predeclared corrected short-envelope R1 study (greedy 512, corrected EOS) is likewise null on the locked split (same-target +3 pp [-8,+14]; cross-source -1 pp [-12,+10]), so the submitted 32k-era claims do not survive correction either.

**Changes in the manuscript.** Section 6 (pp. 10--12) and Appendix Table 10 (p. 17) report the schedule results, token-length costs, stop reasons, and R1 gate. The historical positive long-context claim was removed.

### B-5. Correctness versus text style, length, and formatting

> **Reviewer comment.** “Given that correct and incorrect trajectories separate early on (often correlating with length), how can we be sure that CrossSteer is actually directing “correctness” rather than merely steering the target toward a lengthier generation style that correlates with correct answers? Have you conducted any behavioral or n-gram analyses on the steered vs. unsteered generations to identify exactly what text-level changes the vector induces?”

**Response.** Trajectory telemetry alone cannot identify a semantic reasoning mechanism. Every formal steering row now records token count, repeated 4-gram fraction, distinct n-gram ratio, answer-marker position, truncation and stop behavior, and paired repair/break outcomes. The preregistered behavior guard rejects cells with a repeated-4-gram increase above 0.05 or mean length above 1.5 times baseline. In locked GSM8K, CrossSteer changes mean length by -18.2 tokens, and repeated-4-gram changes remain within 0.004 of the baseline value 0.058. MATH-500 and SVAMP include per-method stopping and truncation telemetry.

**Direct answer and boundary.** These controls rule out large changes in length, repetition, truncation, and answer formatting in the reported cells, but they do not prove that the vector injects a unique reasoning or self-correction mechanism. We therefore present the telemetry as behavioral controls, not mechanism evidence, and removed the stronger mechanistic interpretation.

**Changes in the manuscript.** Section 6 (pp. 10--12), Figure 6, and Appendix Tables 8--10 report these controls; the Discussion and Limitations explicitly state that residual causal alternatives remain.

### B-6a. Names and first-use definitions

> **Reviewer comment.** “The terms GeoVote and CrossSteer are introduced abruptly in the Abstract and Introduction before formally spelling them out or explaining their definitions (such as "GEometric VOTing" or "CROSS-model STEERing").”

**Response.** Correct. At first use, GeoVote is expanded as *geometric voting* and CrossSteer as *cross-model steering*, and each receives a one-sentence operational definition before any result is discussed.

**Changes in the manuscript.** Abstract, Introduction, and Section 3 (pp. 1--6).

### B-6b. Intuition, diagrams, and pseudocode

> **Reviewer comment.** “They omit intuitive spatial analogies, visual diagrams of the curvature angles, or pseudocode.”

**Response.** The submitted equations required too much mental translation. Section 3 now introduces the hidden-state sequence as a path through a high-dimensional space, explains curvature using a 2D sketch while computing it in the original layer space, and gives Algorithms 1--2 for candidate selection and steering.

**Changes in the manuscript.** Figure 2 and Section 3 (pp. 4--6), including Algorithms 1--2.

### B-6c. Ambiguous notation

> **Reviewer comment.** “Minor: What is y_i in Section 3.3? What is m(H.) in Eq 3? The variable m seems to be overloaded.”

**Response.** Thank you for identifying this concrete notation problem. We removed the undefined $y_i$, use gold answer $a_i$, and reserve $\phi$ for the trajectory functional. The notation table defines every symbol at first use, and $m$ is no longer overloaded.

**Changes in the manuscript.** Section 3 and Table 1 (pp. 4--6).

### B-6d. General readability and exposition

> **Reviewer comment.** “Apart from these questions, I found the paper to be very hard to read.”

**Response.** We took this criticism seriously and rewrote the manuscript in a report-first style. We now state what was generated, which candidates share a pool, how a score is selected, and what conclusion follows before introducing the corresponding formal detail. We also moved qualifications into captions, replaced compressed or cryptic sentences with direct descriptions, and removed positive-result wording that was not supported after the controls.

**Changes in the manuscript.** Abstract, Introduction, Sections 3--6, Discussion, Figures 1--6, and Appendix. The final manuscript has no unresolved references or overfull boxes.

## Response to Reviewer C

We thank Reviewer C for recognizing the full-trajectory perspective as a solid extension of prior work and for explaining clearly why the submitted baselines, OOD evidence, prose, and release plan were not yet sufficient for TACL. We addressed each of those points below.

### C-1. OOD generalization

> **Reviewer comment.** “I would like to see this section expanded with a fuller discussion of the OOD generalization, and perhaps extension to additional math tasks.”

**Response.** We completed the requested expansion using all 500 MATH-500 questions and all 1,000 SVAMP items. Configurations were selected on GSM8K and then frozen; there was no OOD retuning. CrossSteer does not show a reliable gain on either dataset: MATH-500 is 0.332 versus 0.338 baseline, and SVAMP is 0.787 versus 0.803 baseline, with paired intervals including zero. MATH-500 is explicitly a 512-token stress envelope, where the unsteered baseline truncation rate is 45.8%; the much lower SVAMP baseline truncation rate is 1.9%. We therefore interpret MATH-500 together with its truncation telemetry rather than as a clean long-form benchmark reproduction. The full eight-method tables and evaluator audits are provided rather than reporting only the source row. A predeclared 4,096-token budget audit of the same family (new Appendix Table 9) confirms this reading: baseline truncation falls to 3.2%, and no method shows a reliable paired gain (all Holm $p=1.0$), so the 512-token stress envelope does not mask a favorable higher-budget result.

**Changes in the manuscript.** Section 6 (pp. 10--12), Appendix Table 8 (p. 17), and Appendix Table 9 (p. 17).

### C-2. Comparison with conventional steering

> **Reviewer comment.** “At a minimum, compare trajectory steering to traditional steering vectors; if they aren’t an improvement, there seems to be no benefit to calculating the full trajectory.”

**Response.** This is the correct standard for the intervention claim. The complete locked family does not establish a performance benefit for CrossSteer: source CrossSteer is -1 pp, while CAA and SAE sparse have +8 and +7 pp point estimates, and all paired intervals include zero after family correction. We therefore remove the claim that full-trajectory steering is superior. We retain the trajectory as a measurement object and as a falsifiable source-transfer test, not as a validated intervention method. The comparison also discloses offline construction cost and the common online vector-addition cost. Bias-only adaptation and reference-free steering optimize target-specific interventions with reinforcement learning or preference data; we cite them as a different supervision setting rather than presenting an apples-to-oranges score comparison as a fair inference-time baseline.

**Changes in the manuscript.** Section 6, Table 6, Figure 6, and Appendix Table 7 (pp. 10--12 and 16).

### C-3. Clearer prose and direct examples

> **Reviewer comment.** “Stylistically, the paper is very dense and reads almost cryptically at times, making it hard to follow.” The reviewer gave examples involving the candidate score, diagnostic accuracy, and source transfer.

**Response.** We rewrote those statements rather than merely adding definitions. For example, the revised manuscript now says: “We reused a frozen, hash-pinned submitted-era historical pool of eight candidate solutions per question from a 2,048-token sampling run for 50 held-out GSM8K questions, and applied every selector to exactly the same candidate pool.” It explains in the Table 2 caption that the diagnostic accuracies characterize the available correct/incorrect trajectory mix and are not official benchmark results. It also states directly that source CrossSteer does not improve the locked target over the shared baseline and conventional controls; the paper no longer asks the reader to infer a benefit from an exploratory point estimate.

**Changes in the manuscript.** Introduction, Sections 3, 5, 6, and Discussion (pp. 1--12). Figures 1--6 and their captions were also rewritten for directness.

### C-4. Reproducibility and software

> **Reviewer comment.** The original review rated reproducibility as difficult and the promised software as unusable.

**Response.** We acknowledge this weakness in the submitted version. The revision adds in-paper reproducibility details and prepares an installable package for post-acceptance release with resolved configurations, split and hash manifests, compact evidence artifacts, a CPU-only smoke test, figure-generation scripts, per-problem audit tables, and 149 unit tests. The release candidate was independently extracted and passed all 149 tests under the supported Python versions. We also document evaluator versions, generation settings, model/tokenizer revisions, seeds, stop rules, and the EOS configuration correction that triggered the complete rerun.

**Changes in the manuscript.** Appendix reproducibility section (p. 15) and the Artifact Availability statement. In accordance with TACL policy, no supplementary material or external anonymous link is included in the resubmission bundle; the manuscript states exactly what will be released upon acceptance and what cannot be redistributed.

### C-5. Dataset release and licensing boundary

> **Reviewer comment.** The review form also rated the promised dataset release as having low impact.

**Response.** We do not introduce a new dataset, and we do not redistribute GSM8K, MATH-500, or SVAMP question text or model weights. Instead, the planned post-acceptance release will include problem IDs and hashes, fixed split manifests, evaluator versions, resolved configurations, content-free per-problem outcomes, compact evidence summaries, and scripts that regenerate the analyses when the user supplies the legally available benchmarks and model checkpoints. This choice preserves benchmark and model licensing while making the reported comparisons auditable.

**Changes in the manuscript.** Appendix reproducibility section (p. 15) and the Artifact Availability statement.

### C-6. Related work, benefits, and limitations

> **Reviewer comment.** It was difficult to determine exactly how the work relates to previous work or what its benefits and limitations are.

**Response.** We expanded Related Work to distinguish trajectory diagnostics, best-of-$N$ controls, conventional activation steering, sparse activation steering, and methods that learn target-specific interventions. We also state the comparison standard directly: a full trajectory has a measurement benefit only if it provides reproducible associations, and it has an intervention benefit only if it beats fair locked baselines. The present evidence supports the former under a fixed protocol, but not the latter.

**Changes in the manuscript.** Section 2 and Discussion/Limitations (pp. 2--4 and p. 12).

## Transparency note on the corrected generation configuration

During the revision, we detected that an early steering-runner generation call passed the tokenizer's single EOS id instead of the Qwen2.5-Instruct model's declared two-id terminator set. This forced every completion in that early family to the 512-token limit. We corrected the runner, invalidated the affected family, and re-ran validation, selection, locked, OOD, and long-context stages under the model-valid EOS configuration. No layer, alpha, split, method, prompt, or evaluator was changed to improve an outcome. All revised intervention, OOD, long-context, and corrected R1 numbers come from the corrected frozen runs. The signature matrices, 2048-token sensitivity comparison, 7B scale audit, and GeoVote candidate pool are explicitly retained as retrospective descriptive artifacts rather than fresh confirmatory evidence. The internal release audit keeps superseded manifests only for provenance and marks them invalid for intervention claims.

For clarity, we state the disposition of each submitted headline number:

- **GeoVote +5.0 pp over log-probability voting on held-out MATH-500 (submitted Abstract).** Withdrawn: the requested same-pool length, shortest-output, residualized, and length-matched permutation controls do not support a reliable incremental gain (AE-2, B-2).
- **Same-target activation addition on R1-Distill, +16 pp on GSM8K ($p=0.009$; submitted Abstract).** Invalidated: the submitted R1 result came from a run family whose generation call overrode the model-declared terminator set with the tokenizer's single EOS id. We therefore invalidated the family as a whole and reran the R1 short-envelope setting under corrected multi-EOS handling. We additionally ran a predeclared corrected short-envelope study (greedy 512, corrected EOS, same no-leakage splits, frozen grid): on the locked split the corrected same-target direction gives +3 pp (CI [-8,+14], $p=0.72$) and the frozen Qwen-Math source direction gives -1 pp (CI [-12,+10], $p=1.0$), against a corrected baseline of 0.41 (up from the submitted 0.32). The submitted +16 pp therefore does not survive correction, and the withdrawal now rests on corrected data. Note that 85--86% of completions in this short-budget rerun reached the generation cap, with mean repeated-four-gram fractions of 0.236 (source) and 0.270 (target calibration), so it audits the submitted short-budget setting rather than establishing model-valid efficacy. The official 32k sampled-decoder path remains stopped by its readiness gate (B-4).
- **Curvature-orientation compatibility diagnostic matching 7/7 steering cells, including a 19 pp overshoot (submitted Abstract and Appendix).** Withdrawn as an intervention claim: orientation is retained only as a descriptive calibration-time covariate, and the locked eight-method comparison does not support transfer predictions (B-1, B-3, C-2).

We include this note prominently because the configuration defect affected the evidentiary validity of the earlier run, and we believe the correct response is full disclosure and complete rerunning rather than selective replacement.

## Closing response

We thank the Action Editor and reviewers for identifying the gaps in the submitted version. The revision does not attempt to preserve every original positive interpretation. It keeps the controlled diagnostic finding, withdraws unsupported claims about GeoVote utility and source-transfer efficacy, adds the requested baselines and OOD/long-context audits, and makes the software and evidence boundary explicit.

We hope this narrower and more transparent paper addresses the concerns and gives the reader a reliable account of what trajectory geometry can---and cannot---establish under the evaluated protocol. We appreciate the time the Action Editor and reviewers have invested in the paper and thank them for considering the revised manuscript.
