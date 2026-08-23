"""
Downstream IPS-LTR evaluation: does better propensity *estimation* translate
into better *ranking* (nDCG)?

Motivation
----------
CIKM'26 reviewers (all three) and the metareview asked whether harmonic's
propensity-estimation variance reduction produces any downstream ranking gain.
Both uniform and harmonic are *consistent* propensity estimators, so any gain
can only appear when propensities are estimated from LIMITED data (few
impressions), where harmonic's lower variance yields a more accurate p_hat_k.
This script isolates that regime.

Design (semi-synthetic, Yahoo LTR Set 1 relevance grades as ground truth)
------------------------------------------------------------------------
Two decoupled logs, as in a real pipeline (estimate propensities once, reuse):

 1. HARVESTING log  -> estimate per-position propensities p_hat_k under each
    weighting scheme, via AdjacentChainEstimator (reuses experiment3 generator).
    We sweep BASE_IMPRESSIONS to move from the noisy (low-data) regime to the
    well-powered regime.

 2. EVALUATION set  -> a logging policy ranks each query's docs by a noisy
    version of true relevance and assigns positions 1..M. Clicks are simulated
    under PBM. We recover IPS-debiased relevance
        rel_hat(q,d) = clickrate(q,d) / p_hat_{pos(q,d)},
    rank docs by rel_hat, and score nDCG@K against the true grades. Ranking by
    the IPS-debiased relevance is the Bayes-optimal (infinite-capacity) IPS-LTR
    ranker, so this is a faithful upper bound on what a learned IPS-LTR model
    targets, without training a 700-feature model on multi-GB files.

Anchors compared per data budget:
    naive     : p_hat_k = 1  (no position correction)  -> lower bound
    uniform   : Agarwal et al. weighting (the paper's baseline)
    harmonic  : our weight
    oracle    : true p_k                                 -> upper bound

Usage:
    python paper/experiment_downstream_ltr.py --pilot     # fast smoke test
    python paper/experiment_downstream_ltr.py             # full run
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import (
    AdjacentChainEstimator,
)

# Reuse the validated experiment3 loader + harvesting-log generator.
from experiment3_yahoo_semisynthetic import (
    generate_dataset,
    load_yahoo,
    true_propensities,
)

YAHOO_PATH = "/home/alessandro/Documents/data/Yahoo_ltr/set1.valid.txt"  # smaller = faster
CACHE_PATH = "/tmp/yahoo_valid_rel.parquet"
RESULTS_DIR = "images/exp_downstream_ltr"

NUM_POSITIONS = 10
ETA = 1.0
TRAFFIC_SPLIT = 0.80
IMBALANCE_RATIO = 20  # heavy-tailed regime where weighting matters

# Data-budget sweep: base impressions in the HARVESTING log. Low => noisy p_hat.
IMPRESSION_BUDGETS = [3, 5, 10, 20, 50]

# Evaluation set
EVAL_IMPRESSIONS = 10      # sessions per (q,d) in the eval log
LOGGING_NOISE = 0.5        # std of gaussian noise added to rel for logging policy
NDCG_K = 10
MIN_DOCS = 10              # queries need at least this many docs to rank


def ndcg_at_k(ranked_rel_grades: np.ndarray, k: int) -> float:
    """nDCG@k given the true relevance grades in the *predicted* rank order."""
    gains = (2.0 ** ranked_rel_grades - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains[:k] * discounts[:k]))
    ideal = np.sort(ranked_rel_grades)[::-1]
    ideal_gains = (2.0 ** ideal - 1.0)
    idcg = float(np.sum(ideal_gains[:k] * discounts[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def build_eval_log(rel_lookup, grade_lookup, query_ids, true_bias, rng):
    """One evaluation ranking per query under a noisy-relevance logging policy.

    Returns a list of dicts: query_id, doc positions, clickrate, true grade.
    Each doc is placed at one position 1..M by the logging policy.
    """
    rows = []
    for qid in query_ids:
        rels = rel_lookup[qid]
        grades = grade_lookup[qid]
        n_docs = len(rels)
        if n_docs < MIN_DOCS:
            continue
        # Logging policy: rank by noisy relevance, take top M for positions 1..M.
        noisy = rels + rng.normal(0, LOGGING_NOISE, size=n_docs)
        order = np.argsort(noisy)[::-1][:NUM_POSITIONS]
        for pos_idx, d in enumerate(order):
            p_k = true_bias[pos_idx]
            rel = float(rels[d])
            n = EVAL_IMPRESSIONS
            clicks = int(rng.binomial(n, p_k * rel))
            rows.append(
                {
                    "query_id": qid,
                    "position": pos_idx + 1,   # 1-indexed
                    "clickrate": clicks / n,
                    "grade": float(grades[d]),
                }
            )
    return pd.DataFrame(rows)


def score_ndcg(eval_df, p_hat, k):
    """Rank each query by IPS-debiased relevance clickrate / p_hat_pos; mean nDCG@k."""
    # p_hat indexed 0..M-1 for positions 1..M
    pos = eval_df["position"].values - 1
    rel_hat = eval_df["clickrate"].values / np.maximum(p_hat[pos], 1e-8)
    tmp = eval_df.assign(rel_hat=rel_hat)
    scores = []
    for _, g in tmp.groupby("query_id"):
        gg = g.sort_values("rel_hat", ascending=False)
        scores.append(ndcg_at_k(gg["grade"].values, k))
    return float(np.mean(scores)) if scores else np.nan


def estimate_propensity(harvest_df, scheme, num_positions):
    est = AdjacentChainEstimator(weighting=scheme)
    try:
        r = est(
            harvest_df,
            query_col="query_id",
            doc_col="doc_id",
            imps_col="impressions",
            clicks_col="clicks",
        )
        vals = r.set_index("position")["examination"].values
        if len(vals) != num_positions or np.any(~np.isfinite(vals)):
            return None
        return vals
    except Exception:
        return None


def run(pilot: bool):
    num_trials = 10 if pilot else 200
    queries_per_trial = 60 if pilot else 150
    budgets = [5, 20] if pilot else IMPRESSION_BUDGETS

    print(f"Loading Yahoo relevance from {YAHOO_PATH} …")
    if os.path.exists(CACHE_PATH):
        df = pd.read_parquet(CACHE_PATH)
    else:
        df = load_yahoo(YAHOO_PATH)
        # load_yahoo returns rel; recover integer grade for nDCG gains.
        df["grade"] = np.round(np.log2(df["rel"] * 15.0 + 1.0)).astype(int)
        df.to_parquet(CACHE_PATH)
    print(f"  {df.query_id.nunique()} queries, {len(df)} docs")

    rel_lookup = {q: g["rel"].values for q, g in df.groupby("query_id")}
    grade_lookup = {q: g["grade"].values for q, g in df.groupby("query_id")}
    all_qids = list(rel_lookup.keys())

    true_bias = true_propensities(NUM_POSITIONS, ETA)
    naive_p = np.ones(NUM_POSITIONS)

    rng_main = np.random.default_rng(0)
    records = []

    for budget in budgets:
        per_scheme = {s: [] for s in ["naive", "uniform", "harmonic", "oracle"]}
        for _ in range(num_trials):
            rng = np.random.default_rng(rng_main.integers(0, 2**31))
            q_idx = rng.choice(len(all_qids), size=queries_per_trial, replace=False)
            trial_qids = [all_qids[i] for i in q_idx]

            # 1. harvesting log -> propensity estimates
            harvest = generate_dataset(
                rel_lookup, trial_qids, true_bias,
                base_impressions=budget, imbalance_ratio=IMBALANCE_RATIO,
                traffic_split=TRAFFIC_SPLIT, rng=rng,
            )
            p_uniform = estimate_propensity(harvest, "original", NUM_POSITIONS)
            p_harmonic = estimate_propensity(harvest, "harmonic", NUM_POSITIONS)
            if p_uniform is None or p_harmonic is None:
                continue

            # 2. eval log -> nDCG under each propensity vector
            eval_df = build_eval_log(rel_lookup, grade_lookup, trial_qids, true_bias, rng)
            if eval_df.empty:
                continue
            per_scheme["naive"].append(score_ndcg(eval_df, naive_p, NDCG_K))
            per_scheme["uniform"].append(score_ndcg(eval_df, p_uniform, NDCG_K))
            per_scheme["harmonic"].append(score_ndcg(eval_df, p_harmonic, NDCG_K))
            per_scheme["oracle"].append(score_ndcg(eval_df, true_bias, NDCG_K))

        row = {"budget": budget, "n_trials": len(per_scheme["uniform"])}
        for s in ["naive", "uniform", "harmonic", "oracle"]:
            arr = np.array(per_scheme[s])
            row[f"ndcg_{s}"] = float(np.mean(arr))
            row[f"se_{s}"] = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) else np.nan
        # paired delta harmonic - uniform
        u = np.array(per_scheme["uniform"])
        h = np.array(per_scheme["harmonic"])
        d = h - u
        row["delta_h_minus_u"] = float(np.mean(d))
        row["delta_se"] = float(np.std(d) / np.sqrt(len(d))) if len(d) else np.nan
        records.append(row)
        print(
            f"budget={budget:3d} | naive={row['ndcg_naive']:.4f} "
            f"uniform={row['ndcg_uniform']:.4f} harmonic={row['ndcg_harmonic']:.4f} "
            f"oracle={row['ndcg_oracle']:.4f} | "
            f"Δ(h-u)={row['delta_h_minus_u']:+.4f} ± {row['delta_se']:.4f}"
        )

    out = pd.DataFrame(records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "downstream_ndcg.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="fast smoke test")
    args = ap.parse_args()
    run(args.pilot)
