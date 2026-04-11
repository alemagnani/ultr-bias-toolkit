"""
Experiment 3: Yahoo LTR Semi-Synthetic

Uses real query-document relevance labels from the Yahoo Learning to Rank
Challenge dataset (Set 1) as ground-truth relevance, then simulates clicks
under the Position-Based Model with the anchor-item impression imbalance model.

This is "semi-synthetic" because:
  - Relevance rel(q,d) comes from real human-judged grades (0–4)
  - Clicks are simulated: P(C=1|q,d,k) = p_k * rel(q,d)
  - Impression imbalance follows the same anchor-item model as Experiment 1

Key advantage over Experiment 1: the relevance distribution is realistic
(many grade-0 docs, few grade-4), which tests whether variance reduction
holds beyond the uniform-random-relevance setting.

Usage:
    python paper/experiment3_yahoo_semisynthetic.py

Outputs:
    images/exp3_yahoo/ with CSV and PNG.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YAHOO_TRAIN_PATH = "/home/alessandro/Documents/data/Yahoo_ltr/set1.train.txt"
YAHOO_TEST_PATH  = "/home/alessandro/Documents/data/Yahoo_ltr/set1.test.txt"
RESULTS_DIR      = "images/exp3_yahoo"

NUM_POSITIONS    = 10
ETA              = 1.0
NUM_TRIALS       = 200
QUERIES_PER_TRIAL = 150   # subsample of queries per trial for speed
BASE_IMPRESSIONS  = 20

TRAFFIC_SPLITS   = [0.80]                      # fixed split for main result
IMBALANCE_RATIOS = [1, 2, 5, 10, 20, 50, 100]  # sweep for figure

SCHEMES = ["original", "min", "harmonic", "clipped_5"]
SCHEME_LABELS = {
    "original":   "Original",
    "min":        "Min",
    "harmonic":   "Harmonic",
    "clipped_5":  r"Clipped $\tau$=5",
}
SCHEME_COLORS = {
    "original":  "black",
    "min":       "blue",
    "harmonic":  "red",
    "clipped_5": "green",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_yahoo(path: str) -> pd.DataFrame:
    """Load Yahoo SVMLight file; return DataFrame with query_id, doc_id, rel."""
    print(f"Loading {path} …")
    # load_svmlight_file is slow for features; we only need labels + qids
    _, y, qids = load_svmlight_file(path, query_id=True)
    y = y.astype(int)

    df = pd.DataFrame({"query_id": qids, "rel_grade": y})
    # Assign local doc_id (0-indexed within each query)
    df["doc_id"] = df.groupby("query_id").cumcount()
    # Convert grade to continuous relevance: (2^grade - 1) / 15
    df["rel"] = (2.0 ** df["rel_grade"] - 1.0) / 15.0
    return df[["query_id", "doc_id", "rel"]]


# ---------------------------------------------------------------------------
# Dataset generation (mirrors experiment1, but uses Yahoo rel)
# ---------------------------------------------------------------------------

def generate_dataset(
    rel_lookup: dict,          # {query_id: array of rel values per doc}
    query_ids: list,
    true_bias: np.ndarray,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate click data using Yahoo relevance grades + anchor-item impressions."""
    rows = []
    global_doc_id = 0
    num_positions = len(true_bias)

    for pos in range(1, num_positions):
        p_k  = true_bias[pos - 1]
        p_kp = true_bias[pos]

        for qid in query_ids:
            rels = rel_lookup[qid]
            n_docs = len(rels)
            if n_docs == 0:
                continue

            n_anchor  = max(1, int(n_docs * traffic_split))
            n_regular = n_docs - n_anchor

            # Shuffle doc order so anchor/regular assignment varies per trial
            perm = rng.permutation(n_docs)

            for idx in range(n_anchor):
                rel = float(rels[perm[idx]])
                n_k  = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                               base_impressions * imbalance_ratio * 2.0)))
                n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
                c_k  = int(rng.binomial(n_k,  rel * p_k))
                c_kp = int(rng.binomial(n_kp, rel * p_kp))
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos,   "impressions": n_k,  "clicks": c_k})
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos+1, "impressions": n_kp, "clicks": c_kp})
                global_doc_id += 1

            for idx in range(n_anchor, n_anchor + n_regular):
                rel = float(rels[perm[idx]])
                n_k  = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
                n_kp = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
                c_k  = int(rng.binomial(n_k,  rel * p_k))
                c_kp = int(rng.binomial(n_kp, rel * p_kp))
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos,   "impressions": n_k,  "clicks": c_k})
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos+1, "impressions": n_kp, "clicks": c_kp})
                global_doc_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def run_sweep(rel_lookup: dict, all_query_ids: list, seed: int = 0) -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []
    rng_main = np.random.default_rng(seed)

    total = len(TRAFFIC_SPLITS) * len(IMBALANCE_RATIOS)
    pbar = tqdm(total=total, desc="Yahoo sweep")

    for split in TRAFFIC_SPLITS:
        for ratio in IMBALANCE_RATIOS:
            rng = np.random.default_rng(rng_main.integers(0, 2**31))
            trial_results = {s: [] for s in SCHEMES}

            for _ in range(NUM_TRIALS):
                # Subsample queries for this trial
                q_idx = rng.choice(len(all_query_ids), size=QUERIES_PER_TRIAL, replace=False)
                trial_qids = [all_query_ids[i] for i in q_idx]

                df = generate_dataset(rel_lookup, trial_qids, true_bias,
                                      BASE_IMPRESSIONS, ratio, split, rng)

                for scheme in SCHEMES:
                    est = AdjacentChainEstimator(weighting=scheme)
                    try:
                        r = est(df, query_col="query_id", doc_col="doc_id",
                                imps_col="impressions", clicks_col="clicks")
                        vals = r.set_index("position")["examination"].values
                        if len(vals) != len(true_bias):
                            vals = np.full(len(true_bias), np.nan)
                    except Exception:
                        vals = np.full(len(true_bias), np.nan)
                    trial_results[scheme].append(vals)

            for s in SCHEMES:
                arr      = np.array(trial_results[s])
                mean_est = np.nanmean(arr, axis=0)
                var_est  = np.nanvar(arr, axis=0)
                bias2    = (mean_est - true_bias) ** 2
                records.append({
                    "traffic_split":   split,
                    "imbalance_ratio": ratio,
                    "scheme":          s,
                    "mse":      float(np.nanmean(var_est + bias2)),
                    "variance": float(np.nanmean(var_est)),
                    "bias2":    float(np.nanmean(bias2)),
                })

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    split_val = 0.80

    sub = results[results["traffic_split"] == split_val]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SCHEMES:
        d = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        axes[0].plot(d["imbalance_ratio"], d["variance"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
        axes[1].plot(d["imbalance_ratio"], d["mse"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Imbalance ratio", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} vs. Imbalance (Yahoo, split={split_val})", fontsize=13)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Yahoo semi-synthetic: weighting scheme comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp3_variance_vs_imbalance.png"), dpi=150)
    plt.close()

    # Variance reduction relative to original
    fig, ax = plt.subplots(figsize=(9, 5))
    orig = results[results["scheme"] == "original"][
        ["traffic_split", "imbalance_ratio", "variance"]
    ].rename(columns={"variance": "var_orig"})

    for s in ["min", "harmonic", "clipped_5"]:
        d = results[results["scheme"] == s].merge(orig, on=["traffic_split", "imbalance_ratio"])
        d["reduction"] = (1 - d["variance"] / d["var_orig"]) * 100
        d_fixed = d[d["traffic_split"] == split_val].sort_values("imbalance_ratio")
        ax.plot(d_fixed["imbalance_ratio"], d_fixed["reduction"],
                marker="o", color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Imbalance ratio", fontsize=12)
    ax.set_ylabel("Variance reduction vs. original (%)", fontsize=12)
    ax.set_title(f"Yahoo semi-synthetic: variance reduction (split={split_val})", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp3_variance_reduction.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_summary(results: pd.DataFrame) -> None:
    sub = results[results["traffic_split"] == 0.80]
    print("\n--- Variance at split=80/20 (Yahoo semi-synthetic) ---")
    print(f"{'Scheme':<20} " + " ".join(f"r={r:>3}" for r in IMBALANCE_RATIOS))
    for s in SCHEMES:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " ".join(f"{v:.5f}" for v in row["variance"].values)
        print(f"{SCHEME_LABELS[s]:<20} {vals}")

    print("\n--- MSE at split=80/20 ---")
    for s in SCHEMES:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " ".join(f"{v:.5f}" for v in row["mse"].values)
        print(f"{SCHEME_LABELS[s]:<20} {vals}")

    print("\n--- Harmonic variance reduction vs original ---")
    for ratio in IMBALANCE_RATIOS:
        o = sub[(sub["scheme"] == "original") & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        h = sub[(sub["scheme"] == "harmonic") & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        print(f"  ratio={ratio:>4}: original={o:.5f}  harmonic={h:.5f}  "
              f"reduction={(1-h/o)*100:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(YAHOO_TEST_PATH):
        print(f"ERROR: Yahoo test file not found at {YAHOO_TEST_PATH}")
        sys.exit(1)

    # Load Yahoo data (test set only for evaluation)
    yahoo_df = load_yahoo(YAHOO_TEST_PATH)
    print(f"Loaded {len(yahoo_df):,} (query, doc) pairs from {yahoo_df['query_id'].nunique():,} queries.")

    # Build per-query relevance lookup
    rel_lookup = {
        qid: grp["rel"].values
        for qid, grp in yahoo_df.groupby("query_id")
    }
    # Keep only queries with at least 2 docs (needed for meaningful splits)
    rel_lookup = {qid: rels for qid, rels in rel_lookup.items() if len(rels) >= 2}
    all_query_ids = sorted(rel_lookup.keys())
    print(f"Queries with ≥2 docs: {len(all_query_ids):,}")

    print(f"\nRunning sweep: {NUM_TRIALS} trials × {QUERIES_PER_TRIAL} queries each")
    print(f"  Traffic splits: {TRAFFIC_SPLITS}")
    print(f"  Imbalance ratios: {IMBALANCE_RATIOS}")

    results = run_sweep(rel_lookup, all_query_ids)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp3_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_results(results, RESULTS_DIR)
    print_summary(results)
    print("\nExperiment 3 (Yahoo semi-synthetic) done.")
