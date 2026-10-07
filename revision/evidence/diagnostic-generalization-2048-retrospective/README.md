# Historical 2048 Qwen Retrieval: Quarantined Retrospective Artifact

This directory records the disposition of the server artifact
`/root/AI/runs/tacl-revision/diagnostic-generalization-2048-budget`.

The artifact reported a nominal held-out model-retrieval rate of 0.667 (three
Qwen roles, 100 shared GSM8K problems) and a label-permutation value of 0.0105.
It is **not paper evidence** and must not be cited as a preregistered,
significant, cross-budget, or generalization result.

It fails the revision protocol's two claim-eligibility gates:

1. Its May 2026 source run directories retain only `config.yaml`,
   `labels.parquet`, and `metrics.parquet`.  They do not preserve an auditable
   resolved tokenizer/chat template, complete EOS set, generation telemetry,
   compact trajectory records, or replay provenance.  The revision cannot
   establish that they use the corrected Qwen decoding envelope.
2. Under the frozen stable SHA-256 partitioning, the common matched incorrect
   count in partition B is only 3 per model.  The revised protocol requires at
   least 10 correct and 10 incorrect trajectories for every model and
   partition before length residualization and retrieval.

No model, metric, layer, alpha, seed, prompt, or result was changed to reach
this decision.  The disposition is a provenance/sample-support safeguard added
because a separate legacy Qwen steering pipeline was found to omit the second
EOS token.  The correct follow-up is a fresh, fully audited diagnostic run, not
a retrospective reanalysis or partition reshuffle.
