"""
Experiment 1: Imbalance Sweep (Synthetic)

PBM with p_k = (1/k)^eta, eta=1.0, M=10 positions.
Sweeps:
  - traffic split: 50/50 ... 95/5
  - frequency imbalance ratio: 1:1 ... 100:1
For each configuration, runs 500 independent trials and reports
MSE, variance, and bias^2 for each weighting scheme.

Usage:
    python paper/experiment1_imbalance_sweep.py

Outputs:
    images/exp1_imbalance_sweep/ with PNG plots and a summary CSV.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_POSITIONS = 10
ETA = 1.0
NUM_TRIALS = 500
NUM_QUERIES = 1          # single query with many (doc, position) pairs
NUM_DOCS = 50            # docs shown at both positions in each adjacent pair
RESULTS_DIR = "images/exp1_imbalance_sweep"

TRAFFIC_SPLITS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # fraction at higher position
IMBALANCE_RATIOS = [1, 2, 5, 10, 20, 50, 100]           # N_k / N_kp for "anchor" docs

SCHEMES = ["original", "min", "harmonic", "clipped_5"]
SCHEME_LABELS = {
    "original": "Original",
    "min": "Min",
    "harmonic": "Harmonic",
    "clipped_5": r"Clipped $\tau$=5",
}
SCHEME_COLORS = {
    "original": "black",
    "min": "blue",
    "harmonic": "red",
    "clipped_5": "green",
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

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
    """Generate aggregated click data for adjacent pairs.

    Anchor-item model: ``traffic_split`` fraction of docs are "anchor" items
    with many impressions at position k (N_k = base * imbalance_ratio, log-
    uniform up to 2x that) and few at k' (N_kp = base, variable).
    The remaining docs are "regular" items with balanced small impressions.

    This produces SYSTEMATIC imbalance (N_k^i > N_kp^i for all docs) with
    VARYING per-doc ratios — the regime where harmonic outperforms original.
    ``traffic_split`` controls the fraction of anchor items; higher split
    means more severe systematic imbalance.
    """
    rows = []
    doc_id = 0
    num_positions = len(true_bias)
    n_anchor = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor

    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]

        # Anchor docs: large N_k, small N_kp, varying ratios
        for d in range(n_anchor):
            rel = rng.uniform(0.1, 0.5)
            n_k = int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                   base_impressions * imbalance_ratio * 2.0))
            n_kp = int(rng.uniform(1, base_impressions + 1))
            n_k = max(n_k, 1)
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

        # Regular docs: small balanced impressions
        for d in range(n_regular):
            rel = rng.uniform(0.1, 0.5)
            n_k = int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5))
            n_kp = int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5))
            n_k = max(n_k, 1)
            n_kp = max(n_kp, 1)
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> dict:
    """Run one trial and return estimated propensities for each scheme."""
    df = generate_dataset(true_bias, num_docs, base_impressions,
                          imbalance_ratio, traffic_split, rng)

    estimates = {}
    for scheme in SCHEMES:
        est = AdjacentChainEstimator(weighting=scheme)
        try:
            result = est(df, query_col="query_id", doc_col="doc_id",
                         imps_col="impressions", clicks_col="clicks")
            vals = result.set_index("position")["examination"].values
            if len(vals) != len(true_bias):
                vals = np.full(len(true_bias), np.nan)
        except Exception:
            vals = np.full(len(true_bias), np.nan)
        estimates[scheme] = vals

    return estimates


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    traffic_splits=TRAFFIC_SPLITS,
    imbalance_ratios=IMBALANCE_RATIOS,
    num_trials=NUM_TRIALS,
    base_impressions=20,
    seed=0,
) -> pd.DataFrame:
    """Run the full sweep and return a DataFrame with MSE/var/bias^2 results."""
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []

    total = len(traffic_splits) * len(imbalance_ratios)
    pbar = tqdm(total=total, desc="Imbalance sweep")

    for split in traffic_splits:
        for ratio in imbalance_ratios:
            rng = np.random.default_rng(seed)
            trial_results = {s: [] for s in SCHEMES}

            for _ in range(num_trials):
                est = run_trial(true_bias, NUM_DOCS, base_impressions,
                                ratio, split, rng)
                for s in SCHEMES:
                    trial_results[s].append(est[s])

            for s in SCHEMES:
                arr = np.array(trial_results[s])  # (num_trials, num_positions)
                mean_est = np.nanmean(arr, axis=0)
                var_est = np.nanvar(arr, axis=0)
                bias2 = (mean_est - true_bias) ** 2
                mse = var_est + bias2

                records.append({
                    "traffic_split": split,
                    "imbalance_ratio": ratio,
                    "scheme": s,
                    "mse": float(np.nanmean(mse)),
                    "variance": float(np.nanmean(var_est)),
                    "bias2": float(np.nanmean(bias2)),
                })

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot 1: variance vs imbalance ratio (for fixed traffic split = 0.80) ---
    split_val = 0.80
    sub = results[results["traffic_split"] == split_val]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SCHEMES:
        d = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        color = SCHEME_COLORS.get(s, "gray")
        label = SCHEME_LABELS.get(s, s)
        axes[0].plot(d["imbalance_ratio"], d["variance"], marker="o",
                     color=color, label=label)
        axes[1].plot(d["imbalance_ratio"], d["mse"], marker="o",
                     color=color, label=label)

    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Imbalance ratio (N_k / N_{k'})", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} vs. Imbalance (traffic split={split_val})", fontsize=13)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_vs_imbalance.png"), dpi=150)
    plt.close()

    # --- Plot 2: variance vs traffic split (for fixed imbalance ratio = 10) ---
    ratio_val = 10
    sub2 = results[results["imbalance_ratio"] == ratio_val]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SCHEMES:
        d = sub2[sub2["scheme"] == s].sort_values("traffic_split")
        color = SCHEME_COLORS.get(s, "gray")
        label = SCHEME_LABELS.get(s, s)
        axes[0].plot(d["traffic_split"] * 100, d["variance"], marker="o",
                     color=color, label=label)
        axes[1].plot(d["traffic_split"] * 100, d["mse"], marker="o",
                     color=color, label=label)

    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Traffic split (% at higher position)", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} vs. Traffic Split (ratio={ratio_val})", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_vs_split.png"), dpi=150)
    plt.close()

    # --- Plot 3: variance reduction relative to original ---
    fig, ax = plt.subplots(figsize=(10, 6))
    orig = results[results["scheme"] == "original"][["traffic_split", "imbalance_ratio", "variance"]]
    orig = orig.rename(columns={"variance": "var_orig"})

    for s in SCHEMES:
        if s == "original":
            continue
        d = results[results["scheme"] == s].merge(orig, on=["traffic_split", "imbalance_ratio"])
        d["var_reduction"] = (1 - d["variance"] / d["var_orig"]) * 100
        d_fixed_split = d[d["traffic_split"] == split_val].sort_values("imbalance_ratio")
        color = SCHEME_COLORS.get(s, "gray")
        label = SCHEME_LABELS.get(s, s)
        ax.plot(d_fixed_split["imbalance_ratio"], d_fixed_split["var_reduction"],
                marker="o", color=color, label=label)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Imbalance ratio (N_k / N_{k'})", fontsize=12)
    ax.set_ylabel("Variance reduction (%)", fontsize=12)
    ax.set_title(f"Variance reduction vs. original (traffic split={split_val})", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_reduction.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_latex_table(results: pd.DataFrame) -> None:
    """Print LaTeX table for fixed split=0.80, all imbalance ratios."""
    sub = results[results["traffic_split"] == 0.80]
    scheme_labels = SCHEME_LABELS

    header = "Scheme & " + " & ".join(str(r) for r in IMBALANCE_RATIOS) + r" \\"
    print("\n--- LaTeX Table: Variance (traffic split=80/20) ---")
    print(r"\begin{tabular}{l" + "r" * len(IMBALANCE_RATIOS) + "}")
    print(r"\hline")
    print(f"{'Scheme':<20} & " + " & ".join(f"ratio={r}" for r in IMBALANCE_RATIOS) + r" \\")
    print(r"\hline")
    for s in SCHEMES:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " & ".join(f"{v:.4f}" for v in row["variance"].values)
        print(f"{scheme_labels.get(s, s):<20} & {vals} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Experiment 1: Imbalance Sweep")
    print(f"  Positions: {NUM_POSITIONS}, eta={ETA}, trials={NUM_TRIALS}")
    print(f"  Traffic splits: {TRAFFIC_SPLITS}")
    print(f"  Imbalance ratios: {IMBALANCE_RATIOS}")

    results = run_sweep()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp1_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_sweep_results(results, RESULTS_DIR)
    print_latex_table(results)
    print("\nExperiment 1 done.")
