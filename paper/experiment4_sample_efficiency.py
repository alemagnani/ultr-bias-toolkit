"""
Experiment 4: Sample Efficiency

Shows how MSE scales with data volume for each weighting scheme,
directly demonstrating the "less data requirements" claim.

Setup: anchor-item model, split=0.80, ratio=50.
Sweeps base_impressions to vary data volume.
At each level, MSE = variance + bias^2 is measured over 500 trials.

The horizontal gap between curves at any fixed MSE threshold shows how
many more impressions "original" needs vs "harmonic" to achieve the same accuracy.

Usage:
    python paper/experiment4_sample_efficiency.py

Outputs:
    images/exp4_sample_efficiency/ with CSV and PNG.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator

RESULTS_DIR      = "images/exp4_sample_efficiency"
NUM_POSITIONS    = 10
ETA              = 1.0
NUM_TRIALS       = 500
NUM_DOCS         = 50
IMBALANCE_RATIO  = 50.0
TRAFFIC_SPLIT    = 0.80

BASE_IMP_VALUES  = [2, 5, 10, 20, 30, 50, 75, 100]

SCHEMES = ["original", "min", "harmonic", "clipped_5"]
SCHEME_LABELS = {
    "original":  "Original",
    "min":       "Min",
    "harmonic":  "Harmonic",
    "clipped_5": r"Clipped $\tau$=5",
}
SCHEME_COLORS = {
    "original":  "black",
    "min":       "blue",
    "harmonic":  "red",
    "clipped_5": "green",
}


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(true_bias, num_docs, base_impressions, imbalance_ratio,
                     traffic_split, rng):
    rows = []
    doc_id = 0
    n_anchor  = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor

    for pos in range(1, len(true_bias)):
        p_k  = true_bias[pos - 1]
        p_kp = true_bias[pos]

        for _ in range(n_anchor):
            rel  = rng.uniform(0.1, 0.5)
            n_k  = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                           base_impressions * imbalance_ratio * 2.0)))
            n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k,  "clicks": int(rng.binomial(n_k,  rel * p_k))})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": int(rng.binomial(n_kp, rel * p_kp))})
            doc_id += 1

        for _ in range(n_regular):
            rel  = rng.uniform(0.1, 0.5)
            n_k  = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            n_kp = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k,  "clicks": int(rng.binomial(n_k,  rel * p_k))})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": int(rng.binomial(n_kp, rel * p_kp))})
            doc_id += 1

    return pd.DataFrame(rows)


def run_sweep(seed: int = 0) -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records   = []

    for base_imp in tqdm(BASE_IMP_VALUES, desc="Sample efficiency sweep"):
        rng = np.random.default_rng(seed)
        trial_results = {s: [] for s in SCHEMES}

        for _ in range(NUM_TRIALS):
            df = generate_dataset(true_bias, NUM_DOCS, base_imp,
                                  IMBALANCE_RATIO, TRAFFIC_SPLIT, rng)
            for scheme in SCHEMES:
                est = AdjacentChainEstimator(weighting=scheme)
                try:
                    r    = est(df, query_col="query_id", doc_col="doc_id",
                               imps_col="impressions", clicks_col="clicks")
                    vals = r.set_index("position")["examination"].values
                    if len(vals) != len(true_bias):
                        vals = np.full(len(true_bias), np.nan)
                except Exception:
                    vals = np.full(len(true_bias), np.nan)
                trial_results[scheme].append(vals)

        # Mean impressions per (q,d,k) — proxy for "data volume"
        # Anchor docs contribute base_imp*ratio on one side, base_imp on the other
        mean_n = (TRAFFIC_SPLIT * (base_imp * IMBALANCE_RATIO + base_imp) / 2
                  + (1 - TRAFFIC_SPLIT) * base_imp)

        for s in SCHEMES:
            arr      = np.array(trial_results[s])
            mean_est = np.nanmean(arr, axis=0)
            var_est  = np.nanvar(arr, axis=0)
            bias2    = (mean_est - true_bias) ** 2
            records.append({
                "base_impressions": base_imp,
                "mean_impressions": mean_n,
                "scheme":           s,
                "mse":      float(np.nanmean(var_est + bias2)),
                "variance": float(np.nanmean(var_est)),
                "bias2":    float(np.nanmean(bias2)),
            })

    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SCHEMES:
        d = results[results["scheme"] == s].sort_values("base_impressions")
        ax.plot(d["base_impressions"], d["mse"], marker="o",
                color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    ax.set_xlabel("Base impressions per document", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(f"Sample efficiency (ratio={int(IMBALANCE_RATIO)}:1, split={TRAFFIC_SPLIT})",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp4_mse_vs_impressions.png"), dpi=150)
    plt.close()

    # Variance only (no bias contamination)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ["original", "min", "harmonic"]:   # skip clipped (near-zero var)
        d = results[results["scheme"] == s].sort_values("base_impressions")
        ax.plot(d["base_impressions"], d["variance"], marker="o",
                color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    ax.set_xlabel("Base impressions per document", fontsize=12)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title(f"Variance vs. data volume (ratio={int(IMBALANCE_RATIO)}:1, split={TRAFFIC_SPLIT})",
                 fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp4_variance_vs_impressions.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_summary(results: pd.DataFrame) -> None:
    print(f"\n--- MSE vs base impressions (ratio={int(IMBALANCE_RATIO)}, split={TRAFFIC_SPLIT}) ---")
    print(f"{'Scheme':<20} " + " ".join(f"b={b:>3}" for b in BASE_IMP_VALUES))
    for s in SCHEMES:
        row  = results[results["scheme"] == s].sort_values("base_impressions")
        vals = " ".join(f"{v:.5f}" for v in row["mse"].values)
        print(f"{SCHEME_LABELS[s]:<20} {vals}")

    # Data savings: at what base_imp does harmonic match original's MSE at base_imp=5?
    orig = results[results["scheme"] == "original"].sort_values("base_impressions")
    harm = results[results["scheme"] == "harmonic"].sort_values("base_impressions")
    print("\n--- Harmonic MSE vs Original MSE ---")
    for _, row_o in orig.iterrows():
        row_h = harm[harm["base_impressions"] == row_o["base_impressions"]].iloc[0]
        saving = (1 - row_h["mse"] / row_o["mse"]) * 100
        print(f"  b={row_o['base_impressions']:>3}: orig={row_o['mse']:.5f}  "
              f"harmonic={row_h['mse']:.5f}  MSE reduction={saving:+.1f}%")


if __name__ == "__main__":
    print(f"Running Experiment 4: Sample Efficiency")
    print(f"  Ratio={IMBALANCE_RATIO}, split={TRAFFIC_SPLIT}, trials={NUM_TRIALS}")
    print(f"  Base impression sweep: {BASE_IMP_VALUES}")

    results = run_sweep()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp4_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_results(results, RESULTS_DIR)
    print_summary(results)
    print("\nExperiment 4 done.")
