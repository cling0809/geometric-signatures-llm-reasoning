# Decision Letter

Dear Authors:

As TACL action editor for submission 11241, "Geometric Signatures of Post-Training in LLM Reasoning Trajectories and Their Use at Inference Time", I am happy to tell you that I am accepting your paper subject (conditional) to your making specific revisions within two months.

Rolling submission: You do *not* need to wait for a submission cycle to submit your work. Your revision can be submitted *anytime* within those two months. We accept B-decision revision submissions on a rolling acceptance basis.

LIST OF MANDATORY REVISIONS:

Reviewers commend the novelty of geometric trajectory analysis and the GeoVote/CrossSteer frameworks, but identify critical gaps requiring resolution prior to publication.

I encourage all the following concerns from all reviews should be carefully handled in your revisions:

(1) justify scale-dependent signature stability;

(2) add steering baselines (CAA/ActAdd, sparse activation steering) and statistical tests to decouple geometric gains from length heuristics;

(3) expand OOD generalization analysis and address CrossSteer failure modes;

(4) consolidate methods and improve clarity with intuitive explanations and unambiguous notation. Please submit a revised manuscript with a point-by-point response to all reviewer comments.



Generally, your revised version will be handled by the same action editor (me) and the same reviewers (if necessary) in making the final decision --- which, *if* all requested revisions are made, will be final acceptance.

You are allowed one to two extra pages of content to accommodate these revisions. To submit your revised version, follow the instructions in the "Revision and Resubmission Policy for TACL Submissions" section of the Author Guidelines at https://transacl.org/ojs/index.php/tacl/about/submissions#authorGuidelines .

Effective March 1, 2024, TACL will allow papers submitted to the journal to contain appendices . The guidance and allowances can be found here: https://transacl.org/index.php/tacl/announcement/view/105.

Thank you for submitting to TACL, and I look forward to your revised version!

Hai Zhao,

AGI Institute, Shanghai Jiao Tong University,

zhaohai@cs.sjtu.edu.cn


## Reviewer A

This work introduces a geometric signature, a metric matrix whose cells measure whether trajectory statistics separate correct from incorrect generations. This work also proposes an inference-time intervention framework that leverages signals derived from source models to improve target model reasoning or diagnose confidence. GEOVOTE confirms these signatures carry a confidence signal beyond log-probability (though gains over majority are mostly length-driven), while CROSSSTEER shows that a source-model direction can repair an oppositely oriented target but overshoots an already-aligned one, with curvature orientation predicting both effects.

Strengths:

1. The proposed length-controlled geometric signature for generated reasoning trajectories is novel and interesting for LLM reasoning.

2. Experimental results show that post-training leaves separable geometric signatures in model trajectories.

3. The introduced geometric metrics for generated reasoning trajectories make sense and show effectiveness for understanding the reasoning path with LLMs.

Weaknesses:

1. The organization of the paper can be significantly improved. The content of the methods section can be consolidated into a single chapter, e.g., Section 5.1 and Section 6.1 can be integrated into a section before the experiment.

Recommendation: Revisions Required


## Reviewer B

This paper looks into whether the hidden-state trajectories of Large Language Models (LLMs) retain reusable traces of their post-training, which are summarized using a layer-wise metric called a geometric signature. By analyzing a family of Qwen2.5-derived models, the authors demonstrate that different post-training paradigms (such as instruction tuning, math specialization, and reasoning distillation) produce distinct, separable geometric patterns. Reasoning-distilled models (like R1-DISTILL) are geometrically closer to instruction-tuned models than to math specialists, and they uniquely reverse a trajectory curvature sign pattern shared by other variants.

Based on these findings, the authors introduce GeoVote, an inference-time voting method that weights best-of-N samples by trajectory scores to outperform log-probability voting on mathematical benchmarks. They also propose CrossSteer, an intervention method that injects a correctness-directing activation vector from a source model into a target model's hidden layers. CrossSteer successfully improves performance on targets with opposite geometric orientations (such as R1-DISTILL), and a target model's signature can be used to predict whether this transfer will be corrective or cause oversteering.

A major strength of this work is its conceptual shift to the temporal dimension of reasoning. This paper investigates the temporal dynamics of LLM reasoning by tracking layer-wise hidden-state trajectories, compiling them into geometric signatures that distinguish correct from incorrect answers. The authors' findings that various post-training methods lead to distinct and separable geometric patterns are very interesting. Building on these findings, both GeoVote and Crosssteer methods  shift from a static to a temporal analytical framework using the entire sequence of generated tokens as a continuous layer-wise trajectory, unlike previous geometry and probing papers that typically evaluate static, single-point activation states or individual tokens.

However, key limitations include scale-dependent taxonomy inconsistencies, a lack of intuitive explanations for the steering vectors, and the fact that GeoVote's predictive edge is largely a length-mediated heuristic rather than a pure geometric signal.

1) In the 1.5B scale diagnostic, you show that R1-DISTILL clusters very closely to INSTRUCT (distance of 1.13). However, your Qwen-7B scale replicate (Table 13) shows that the nearest neighbor of R1-DISTILL shifts to the BASE model (distance of 0.844 vs. 2.198 to Instruct). A similar shift in clustering order is observed when scaling the evaluation slice to n=300 (Section 4.2 / Appendix C).

This shift suggests that the fine-grained geometric paradigm taxonomy you present is highly sensitive to model scale and sample size. Why does reasoning distillation cluster with Base at the 7B scale but with Instruct at the 1.5B scale? How can these signatures be considered stable "fingerprints" of post-training paradigms if their relative topological relationships do not preserve across model scales?
2) At N=16, the frozen geometric rule selected during calibration is TRAJLEN@L4. However, your control audit reveals that a simple token-count-only baseline (ngen, min) achieves the exact same performance margin over majority voting (0.705 vs. 0.693, or 0.713 for weighted voting). Furthermore, when curvature is residualized against token length, the geometric advantage over majority voting entirely evaporates (dropping to 0.690 vs. 0.693).

Does GeoVote provide any practical or conceptual utility over a trivial completion-length heuristic? Can you provide an experimental setting where the trajectory metrics provide a statistically significant improvement over a simple token-count heuristic? On this note, please provide significance tests for your comparisons.
3) Since cross-model transfer requires calibration data on a source model and a geometric signature from a target model to determine orientation, what is the realistic use case for CrossSteer over simply using the target's calibration data to run target-calibrated steering? Furthermore, can you justify why standard CAA/ActAdd  (mentioned in Limitations) benchmarks were omitted, and how CrossSteer compares to them in performance and steering efficiency?

4) In Appendix E, you note that when extending the token budget from 4096 tokens to a 32k greedy budget on MATH-500, CrossSteer fails completely, dropping target performance from 0.42 to 0.33 due to the target model entering repetition loops. What does this tell us about the safety and stability of CrossSteer? Have you explored any normalization or decay techniques (such as decaying the steering scale over long contexts) to mitigate these repetition loops?

5)  In the discussion section, you state that it is unclear whether the steering vector is injecting reasoning/self-correction behaviors, or merely acting as a style or formatting shift (e.g., forcing the model to write out longer formatting steps). Given that correct and incorrect trajectories separate early on (often correlating with length), how can we be sure that CrossSteer is actually directing "correctness" rather than merely steering the target toward a lengthier generation style that correlates with correct answers? Have you conducted any behavioral or n-gram analyses on the steered vs. unsteered generations to identify exactly what text-level changes the vector induces?

Apart from these questions, I found the paper to be very hard to read.

The terms GeoVote and CrossSteer are introduced abruptly in the Abstract and Introduction before formally spelling them out or explaining their definitions (such as "GEometric VOTing" or "CROSS-model STEERing"). They are dropped into the text as self-evident marketing names.
Sections 3, 5, and 6 introduce a rapid, dense succession of mathematical variables, dimensionality brackets, and subscripts. The authors explain these formulations strictly in formal mathematical notation. They omit intuitive spatial analogies, visual diagrams of the curvature angles, or pseudocode. A reader is forced to mentally translate abstract tensor operations into physical geometric paths, which severely hurts readability.
Minor: What is y_i in Section 3.3? What is m(H.) in Eq 3? The variable m seems to be overloaded.



Recommendation: Revisions Required


## Reviewer C
Recommendation: Resubmit for Review


CLARITY: For the reasonably well-prepared reader, is it clear what was done and why? Is the paper well-written and well-structured?

2. Important questions were hard to resolve even with effort.


INNOVATIVENESS: How original is the approach? Does this paper break new ground in topic, methodology, or content? How exciting and innovative is the research it describes?

Note that a paper can score high for innovativeness even if its impact will be limited.

3. Respectable: A nice research contribution that represents a notable extension of prior approaches or methodologies.

SOUNDNESS/CORRECTNESS: First, is the technical approach sound and well-chosen? Second, can one trust the claims of the paper -- are they supported by proper experiments and are the results of the experiments correctly interpreted?

3. Fairly reasonable work. The approach is not bad, and at least the main claims are probably correct, but I am not entirely ready to accept them (based on the material in the paper).


RELATED WORK: Does the submission make clear where the presented system sits with respect to existing literature? Are the references adequate?

Note that the existing literature includes preprints, but in the case of preprints:

Authors should be informed of but not penalized for missing very recent and/or not widely known work.
If a refereed version exists, authors should cite it in addition to or instead of the preprint.
3. Bibliography and comparison are somewhat helpful, but it could be hard for a reader to determine exactly how this work relates to previous work or what its benefits and limitations are.


SUBSTANCE: Does this paper have enough substance (in terms of the amount of work), or would it benefit from more ideas or analysis?

Note that papers or preprints appearing less than three months before a paper is submitted to TACL are considered contemporaneous with the submission. This relieves authors from the obligation to make detailed comparisons that require additional experiments and/or in-depth analysis, although authors should still cite and discuss contemporaneous work to the degree feasible.

4. Represents an appropriate amount of work for a publication in this journal. (most submissions)

IMPACT OF IDEAS OR RESULTS: How significant is the work described? If the ideas are novel, will they also be useful or inspirational? If the results are sound, are they also important? Does the paper bring new insights into the nature of the problem?

3. Interesting but not too influential. The work will be cited, but mainly for comparison or as a source of minor contributions.

REPLICABILITY: Will members of the ACL community be able to reproduce or verify the results in this paper?

3. They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.

IMPACT OF PROMISED SOFTWARE:  If the authors state (in anonymous fashion) that their software will be available, what is the expected impact of the software package?

1. No usable software will be released.


IMPACT OF PROMISED DATASET(S): If the authors state (in anonymous fashion) that datasets will be released, how valuable will they be to others?

1. No usable datasets will be released.


TACL-WORTHY AS IS? In answering, think over all your scores above. If a paper has some weaknesses, but you really got a lot out of it, feel free to recommend it. If a paper is solid but you could live without it, let us know that you're ambivalent.

Reviewers: after you save this review form, you'll have to make a confidential recommendation to the editors via pull-down menu as to: what degree of revision would be needed to make the submission eventually TACL-worthy?

3. Ambivalent: OK but does not seem up to the standards of TACL.


Detailed Comments for the Authors

Reviewers, please draft your comments on your own filesystem and then copy the results into the text-entry box.  You will thus have a saved copy in case of system glitches.

The paper looks at hidden-state trajectories and how they can be used to distinguish signatures of different post-training regimes, or for steering interventions. The approach is somewhat related to extracting steering vectors, but instead of choosing a single sequence position and layer to use for a steering intervention, a full trajectory is extracted from all layers, capturing the series of changes from one hidden state to the next. The trajectories are transformed into several scalar metrics based on length, step size, and curvature for the signature analysis, and aggregated for a ranking experiment and a steering intervention.

The approach of looking at the full trajectory is a solid extension of prior work, and it seems reasonable that the progress of the whole trajectory may contain information not represented by hidden states at a single position. The analysis of trajectory signatures on Qwen models provides interesting results, showing that models with math SFT/RL vs CoT distillation may both get the right answer on GSM8K, but they arrive at it by trajectories with quite different geometry.

The ranking experiment results are somewhat unclear, but steering does show a positive result. Steering is done using averaged trajectory steps and taking the difference between correct and incorrect model answers. Most of the experiments in the paper use an in-domain held-out test set (from GSM8K), which does raise the question of whether the steering result generalizes to other similar tasks or represents overfitting. However, section 6.4 does briefly discuss cross-dataset generalization to MATH-500. I would like to see this section expanded with a fuller discussion of the OOD generalization, and perhaps extension to additional math tasks.

The paper could be really improved with baselines for the steering method, which would take it beyond a proof of concept. At a minimum, compare trajectory steering to traditional steering vectors; if they aren’t an improvement, there seems to be no benefit to calculating the full trajectory. There are also many recent improvements to traditional steering, e.g. bias-only adaptation (Sinii et al 2025), reference-free steering (Wu et al. 2025), sparse activation steering (Bayat et al. 2025), which could be points of comparison.

Stylistically, the paper is very dense and reads almost cryptically at times, making it hard to follow. It would be much more understandable if it followed a standard, straightforward “report exactly what we did and what happened” style. Examples: line 494: “The same signature supplies two quantities used below. A scalar trajectory score ranks sampled solutions.” This would be a lot clearer as “We generated n samples and used the xyz metric to rank them.” Line 535: “The absolute accuracies are diagnostic values, not benchmark ceilings.” I think this means something like: “Because we used a small model and test set, the results in Table X are not state of the art on Benchmark Y. We are only interested in the relative performance across methods.” Line 742: “Thus CrossSteer claims transferable intervention without target-side labels, not dominance over target-calibrated steering.” I think this means something like “Table 6 shows that steering vectors derived from a post-trained model’s own trajectories perform better than ones derived from a different Qwen model, but there is still some benefit.”
REVIEWER CONFIDENCE

4. Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
