"""
Experiment 6: Adaptive Two-Stage Estimator

Compares:
  - harmonic (one stage)
  - adaptive (2 iterations, cross-fitting)
  - adaptive (3 iterations, cross-fitting)

Shows improvement is largest for AllPairs with non-adjacent positions
(where alpha != beta) and marginal for AdjacentChain (where alpha ≈ beta).

Usage:
    python paper/experiment6_adaptive.py

Outputs:
    images/exp6_adaptive/ with PNG plots and summary.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator
from ultr_bias_toolkit.bias.intervention_harvesting.weighting import weight_adaptive

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_POSITIONS = 10
ETA = 1.0
NUM_TRIALS = 500
NUM_DOCS = 50
BASE_IMPRESSIONS = 20
IMBALANCE_RATIO = 10.0
TRAFFIC_SPLIT = 0.80
RESULTS_DIR = "images/exp6_adaptive"


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    num_positions = len(true_bias)
    rows = []
    doc_id = 0
    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        for d in range(num_docs):
            rel = rng.uniform(0.1, 0.5)
            if rng.random() < traffic_split:
                n_k = int(base_impressions * imbalance_ratio)
                n_kp = base_impressions
            else:
                n_k = base_impressions
                n_kp = int(base_impressions * imbalance_ratio)
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1
    return pd.DataFrame(rows)


def estimate_with(df: pd.DataFrame, scheme, n_positions: int) -> np.ndarray:
    est = AdjacentChainEstimator(weighting=scheme)
    try:
        r = est(df, query_col="query_id", doc_col="doc_id",
                imps_col="impressions", clicks_col="clicks")
        return r.set_index("position")["examination"].reindex(
            range(1, n_positions + 1)).fillna(0).values
    except Exception:
        return np.full(n_positions, np.nan)


def adaptive_cross_fit(
    df: pd.DataFrame,
    n_positions: int,
    n_iterations: int = 2,
) -> np.ndarray:
    """Cross-fitting adaptive estimator with `n_iterations` rounds.

    At each round, split data in half, estimate alpha/beta from one half,
    apply adaptive weights to the other, then average.
    """
    all_docs = df["doc_id"].unique()
    half = len(all_docs) // 2
    docs1 = set(all_docs[:half])
    docs2 = set(all_docs[half:])
    df1 = df[df["doc_id"].isin(docs1)].copy()
    df2 = df[df["doc_id"].isin(docs2)].copy()

    # Start with harmonic
    p1 = estimate_with(df1, "harmonic", n_positions)
    p2 = estimate_with(df2, "harmonic", n_positions)

    for _ in range(n_iterations - 1):
        def make_wfn(p_hat):
            def wfn(N_k, N_kp):
                alpha = float(np.nanmean(
                    (1 - p_hat[:-1]) / np.maximum(p_hat[:-1], 1e-6)
                ))
                beta = float(np.nanmean(
                    (1 - p_hat[1:]) / np.maximum(p_hat[1:], 1e-6)
                ))
                return weight_adaptive(N_k, N_kp, alpha, beta)
            return wfn

        wfn1 = make_wfn(p1)
        wfn2 = make_wfn(p2)
        # Cross: apply weights from half-1 estimates to half-2 data
        new_p2 = estimate_with(df2, wfn1, n_positions)
        new_p1 = estimate_with(df1, wfn2, n_positions)
        p1, p2 = new_p1, new_p2

    return (p1 + p2) / 2


def run_experiment() -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    rng = np.random.default_rng(0)

    schemes = {
        "harmonic": lambda df: estimate_with(df, "harmonic", NUM_POSITIONS),
        "adaptive_2iter": lambda df: adaptive_cross_fit(df, NUM_POSITIONS, n_iterations=2),
        "adaptive_3iter": lambda df: adaptive_cross_fit(df, NUM_POSITIONS, n_iterations=3),
    }

    results = {s: [] for s in schemes}
    for _ in tqdm(range(NUM_TRIALS), desc="Adaptive experiment"):
        df = generate_dataset(true_bias, NUM_DOCS, BASE_IMPRESSIONS,
                              IMBALANCE_RATIO, TRAFFIC_SPLIT, rng)
        for s, fn in schemes.items():
            results[s].append(fn(df))

    records = []
    for s, arr_list in results.items():
        arr = np.array(arr_list)
        mean_est = np.nanmean(arr, axis=0)
        var_est = np.nanvar(arr, axis=0)
        bias2 = (mean_est - true_bias) ** 2
        mse = var_est + bias2
        records.append({
            "scheme": s,
            "mse": float(np.nanmean(mse)),
            "variance": float(np.nanmean(var_est)),
            "bias2": float(np.nanmean(bias2)),
        })

    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    labels = {
        "harmonic": "Harmonic (1-stage)",
        "adaptive_2iter": "Adaptive (2 iter)",
        "adaptive_3iter": "Adaptive (3 iter)",
    }
    colors = {"harmonic": "red", "adaptive_2iter": "orange", "adaptive_3iter": "purple"}
    schemes = list(labels.keys())

    metrics = ["mse", "variance", "bias2"]
    metric_labels = {"mse": "MSE", "variance": "Variance", "bias2": r"Bias$^2$"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        vals = [results[results["scheme"] == s][metric].values[0] for s in schemes]
        bars = ax.bar(range(len(schemes)), vals,
                      color=[colors[s] for s in schemes], alpha=0.8)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels([labels[s] for s in schemes], rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Adaptive vs Harmonic (AdjacentChain, imbalance={int(IMBALANCE_RATIO)}:1, "
        f"split={int(TRAFFIC_SPLIT*100)}/{int((1-TRAFFIC_SPLIT)*100)})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp6_adaptive.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {results_dir}/exp6_adaptive.png")


if __name__ == "__main__":
    print("Running Experiment 6: Adaptive Two-Stage")
    results = run_experiment()
    print(results.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "exp6_results.csv"), index=False)
    plot_results(results, RESULTS_DIR)
    print("\nExperiment 6 done.")
