"""Answer-selection strategies for best-of-N sampling.

Given N samples per question (with predictions, sequence logprobs, and
per-layer geometric metrics), compare strategies for picking the final answer:

  - greedy_pass1   : the first sample (idx=0)
  - random_pick    : a random sample (sanity baseline)
  - majority_vote  : the most common prediction (self-consistency baseline)
  - logprob_max    : sample with highest sequence_logprob
  - logprob_weighted_vote : votes weighted by exp(normalized logprob)
  - geo_min        : sample with min geometric metric value
  - geo_max        : sample with max geometric metric value
  - geo_weighted_vote (+ sign) : votes weighted by metric value (or its inverse)
  - oracle         : pick any correct sample if exists (upper bound)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Literal

import numpy as np
import pandas as pd

from geoprobe.datasets.gsm8k import is_correct as is_correct_gsm8k


def _best_by(values: list[float], maximize: bool) -> int:
    return int(np.argmax(values) if maximize else np.argmin(values))


def _weighted_vote(preds: list, weights: list[float]) -> object:
    """Weighted majority. If two preds tie on weight, the first wins."""
    tot: dict = defaultdict(float)
    for p, w in zip(preds, weights):
        if p is None:
            continue
        tot[p] += w
    if not tot:
        return None
    return max(tot.items(), key=lambda kv: kv[1])[0]


def _majority(preds: list) -> object:
    nonnull = [p for p in preds if p is not None]
    if not nonnull:
        return None
    return Counter(nonnull).most_common(1)[0][0]


def evaluate_strategies(
    labels_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
    geo_metric: str | None = None,
    geo_layer: int | None = None,
    geo_sign: Literal["min", "max"] = "min",
    rng_seed: int = 0,
    grader: Callable[[object, object], bool] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame [strategy, accuracy, n_questions].

    geo_metric / geo_layer / geo_sign control the geometric strategies.
    geo_sign='min' = pick smallest value (sign-flipped: lower => more correct).
    """
    rng = np.random.default_rng(rng_seed)
    if grader is None:
        grader = is_correct_gsm8k  # backward-compat default
    by_q = labels_df.groupby("sample_id")
    # Optional metrics lookup
    metrics_lookup = None
    if metrics_df is not None and geo_metric is not None and geo_layer is not None:
        filt = metrics_df[(metrics_df["metric"] == geo_metric) &
                          (metrics_df["layer"] == geo_layer)]
        # multi-index on (sample_id, sample_idx)
        metrics_lookup = filt.set_index(["sample_id", "sample_idx"])["value"].to_dict()

    rows: list[dict] = []
    correct_counts: dict[str, int] = defaultdict(int)
    n_questions = 0

    for sample_id, g in by_q:
        g = g.sort_values("sample_idx")
        preds = g["pred"].tolist()
        golds = g["gold"].tolist()
        gold = golds[0]
        correctness = g["correct"].tolist()
        logprobs = g["sequence_logprob"].tolist()
        idxs = g["sample_idx"].tolist()
        n_questions += 1

        # --- strategy: greedy/first sample
        correct_counts["greedy_pass1"] += int(bool(correctness[0]))

        # --- strategy: random
        r = int(rng.integers(0, len(preds)))
        correct_counts["random_pick"] += int(bool(correctness[r]))

        # --- strategy: majority vote
        mv = _majority(preds)
        correct_counts["majority_vote"] += int(grader(mv, gold))

        # --- strategy: logprob max
        i = _best_by(logprobs, maximize=True)
        correct_counts["logprob_max"] += int(bool(correctness[i]))

        # --- strategy: logprob-weighted vote (softmax over logprobs)
        lp = np.array(logprobs, dtype=float)
        lp = lp - lp.max()
        w = np.exp(lp)
        correct_counts["logprob_weighted_vote"] += int(grader(_weighted_vote(preds, list(w)), gold))

        # --- strategy: oracle (any-correct)
        correct_counts["oracle"] += int(any(correctness))

        # --- geometric strategies (if metric provided)
        if metrics_lookup is not None:
            vals = [metrics_lookup.get((sample_id, ix), float("nan")) for ix in idxs]
            if not any(np.isnan(vals)):
                # min or max by sign
                i = _best_by(vals, maximize=(geo_sign == "max"))
                correct_counts[f"geo_{geo_sign}_{geo_metric}_L{geo_layer}"] += int(bool(correctness[i]))

                # weighted vote: if sign='min', weight by 1/value; if 'max', by value
                vals_np = np.array(vals, dtype=float)
                if geo_sign == "min":
                    # avoid div by 0
                    w = 1.0 / (vals_np + 1e-8)
                else:
                    w = vals_np
                # softmax-style normalization
                w = np.maximum(w, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                correct_counts[f"geo_{geo_sign}_weighted_vote_{geo_metric}_L{geo_layer}"] += int(
                    grader(_weighted_vote(preds, list(w)), gold)
                )

                # --- COMBINED: majority-count multiplied by geo-confidence sum ---
                # For each candidate answer a:
                #   score(a) = count(a) * sum_{i: pred_i == a} w_i
                # rewards answers that are both popular AND geo-confident
                # (where w_i = 1/value if sign=min, else value).
                tot_count = defaultdict(int)
                tot_score = defaultdict(float)
                for p, wi in zip(preds, w):
                    if p is None:
                        continue
                    tot_count[p] += 1
                    tot_score[p] += float(wi)
                if tot_count:
                    combined = {a: tot_count[a] * tot_score[a] for a in tot_count}
                    pick = max(combined.items(), key=lambda kv: kv[1])[0]
                else:
                    pick = None
                correct_counts[f"geo_majority_combined_{geo_metric}_L{geo_layer}"] += int(grader(pick, gold))

                # --- COMBINED-ALT: log-additive score combining majority-count and geo-confidence
                # score(a) = log(count(a)+1) + λ * log(sum_geo_weight(a)+ε)
                # Less spiky than multiplicative, useful sanity check.
                lam = 1.0
                eps = 1e-8
                if tot_count:
                    combined_log = {
                        a: np.log(tot_count[a] + 1) + lam * np.log(tot_score[a] + eps)
                        for a in tot_count
                    }
                    pick_log = max(combined_log.items(), key=lambda kv: kv[1])[0]
                else:
                    pick_log = None
                correct_counts[f"geo_majority_logsum_{geo_metric}_L{geo_layer}"] += int(grader(pick_log, gold))

                # --- CONDITIONAL: use majority if it has >= N/2 (clear majority),
                # else fall back to geo-weighted vote. The intuition is that geo helps
                # exactly when sample-disagreement is high.
                pick_cond = None
                if tot_count:
                    top_a, top_n = max(tot_count.items(), key=lambda kv: kv[1])
                    n_total = sum(tot_count.values())
                    if top_n * 2 >= n_total:
                        pick_cond = top_a
                    else:
                        pick_cond = _weighted_vote(preds, list(w))
                correct_counts[f"geo_then_majority_{geo_metric}_L{geo_layer}"] += int(grader(pick_cond, gold))

                # --- TOP-2 RERANK: take top-2 answers by majority count; if tie, geo-pick.
                # If one answer dominates, majority is used. If top-2 are close,
                # geo decides between them — using only samples that voted for those two.
                pick_top2 = None
                if tot_count:
                    sorted_by_count = sorted(tot_count.items(), key=lambda kv: -kv[1])
                    if len(sorted_by_count) == 1 or sorted_by_count[0][1] > sorted_by_count[1][1]:
                        pick_top2 = sorted_by_count[0][0]
                    else:
                        # tie between top-1 and top-2: geo-pick between them
                        a, b = sorted_by_count[0][0], sorted_by_count[1][0]
                        sa = sum(wi for p, wi in zip(preds, w) if p == a)
                        sb = sum(wi for p, wi in zip(preds, w) if p == b)
                        pick_top2 = a if sa >= sb else b
                correct_counts[f"majority_geo_tiebreak_{geo_metric}_L{geo_layer}"] += int(grader(pick_top2, gold))

    for name, c in correct_counts.items():
        rows.append({"strategy": name, "accuracy": c / n_questions, "n_questions": n_questions})

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
