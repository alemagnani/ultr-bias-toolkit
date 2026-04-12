"""
Experiment 5: Production A/B Testing Simulation

Demonstrates that anchor-item impression imbalance arises naturally from
realistic multi-ranker production deployments, and quantifies harmonic's
advantage in this setting.

Production model:
  - A "default" ranker handles fraction alpha_0 of traffic; it shows each doc
    at its natural position k.
  - K-1 "treatment" rankers each handle fraction alpha_i of traffic; each
    independently swaps adjacent position pairs (k, k+1) with probability p_swap,
    so a doc that is at position k in the default may appear at k+1 in a treatment.
  - The resulting impression ratio N_k / N_{k+1} for a doc observed at both
    positions scales directly with the traffic allocation: ratio ≈ alpha_0 / alpha_i.

We compare two configurations:
  Uniform:     K=5, each ranker 20% traffic, p_swap=0.5
  Production:  K=5, default 80%, four treatments 5% each, p_swap=0.5

The ratio distribution plot directly shows that production A/B testing
creates heavy-tailed imbalance (the regime where harmonic weighting helps),
while uniform traffic stays near ratio=1 (the Baidu-like regime).

Usage:
    python paper/experiment5_production_simulation.py

Outputs:
    images/exp5_production/ with CSV and PNG.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator

RESULTS_DIR   = "images/exp5_production"
NUM_POSITIONS = 10
ETA           = 1.0
NUM_QUERIES   = 300
NUM_DOCS      = 40
BASE_IMP      = 30    # base impressions per (ranker × query × doc) per period
P_SWAP        = 0.5   # probability treatment ranker swaps a pair
NUM_TRIALS    = 500

CONFIGS = {
    "uniform":    [0.20, 0.20, 0.20, 0.20, 0.20],
    "production": [0.80, 0.05, 0.05, 0.05, 0.05],
}

SCHEMES = ["original", "harmonic"]
SCHEME_LABELS = {"original": "Original", "harmonic": "Harmonic"}
SCHEME_COLORS = {"original": "black",    "harmonic": "red"}


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def make_treatment_ranking(default_ranking, p_swap, rng):
    """Swap each adjacent pair (k, k+1) independently with probability p_swap."""
    ranking = list(default_ranking)
    for i in range(len(ranking) - 1):
        if rng.random() < p_swap:
            ranking[i], ranking[i + 1] = ranking[i + 1], ranking[i]
    return ranking


def generate_dataset(traffic_alloc, true_bias, rng):
    """
    Each query has a fixed default ranking. Treatment rankers apply random
    adjacent swaps. Impressions at each (doc, pos) cell are Binomial draws
    proportional to the ranker's traffic share.
    """
    K = len(traffic_alloc)
    rows = []

    for q_idx in range(NUM_QUERIES):
        # Relevance per doc (fixed per query)
        rel = rng.uniform(0.1, 0.9, size=NUM_DOCS)

        # Default ranking: random permutation of docs into positions
        default_ranking = rng.permutation(NUM_DOCS)[:NUM_POSITIONS].tolist()

        # Accumulate impressions per (doc, pos)
        imp_map = {}
        clk_map = {}

        for r_idx in range(K):
            alpha = traffic_alloc[r_idx]
            if r_idx == 0:
                ranking = default_ranking
            else:
                ranking = make_treatment_ranking(default_ranking, P_SWAP, rng)

            for pos_0, doc in enumerate(ranking):
                pos = pos_0 + 1
                n = rng.binomial(BASE_IMP, alpha)
                p_click = true_bias[pos - 1] * rel[doc]
                c = rng.binomial(n, min(p_click, 1.0))
                key = (doc, pos)
                imp_map[key] = imp_map.get(key, 0) + n
                clk_map[key] = clk_map.get(key, 0) + c

        for (doc, pos), n_total in imp_map.items():
            rows.append({
                "query_id":    q_idx,
                "doc_id":      doc,
                "position":    pos,
                "impressions": n_total,
                "clicks":      clk_map[(doc, pos)],
            })

    return pd.DataFrame(rows)


def compute_ratio_distribution(df):
    """Compute N_k / N_{k+1} for all adjacent-position (q,d) pairs."""
    ratios = []
    for (_, _), grp in df.groupby(["query_id", "doc_id"]):
        pos_map = dict(zip(grp["position"], grp["impressions"]))
        for k in range(1, NUM_POSITIONS):
            if k in pos_map and (k + 1) in pos_map:
                nk, nkp = pos_map[k], pos_map[k + 1]
                if nkp > 0:
                    ratios.append(nk / nkp)
    return np.array(ratios)


def run_mse_sweep(traffic_alloc, rng_seed=0):
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    rng = np.random.default_rng(rng_seed)
    trial_results = {s: [] for s in SCHEMES}

    for _ in range(NUM_TRIALS):
        df = generate_dataset(traffic_alloc, true_bias, rng)
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

    results = {}
    for s in SCHEMES:
        arr = np.array(trial_results[s])
        mean_est = np.nanmean(arr, axis=0)
        var_est  = np.nanvar(arr, axis=0)
        bias2    = (mean_est - true_bias) ** 2
        results[s] = {
            "mse":      float(np.nanmean(var_est + bias2)),
            "variance": float(np.nanmean(var_est)),
            "bias2":    float(np.nanmean(bias2)),
        }
    return results


def plot_ratio_distributions(ratio_data: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(-1, 2, 50)

    titles = {
        "uniform":    "Uniform traffic (5 × 20%)",
        "production": "Production A/B test (80% + 4 × 5%)",
    }
    colors = {"uniform": "steelblue", "production": "tomato"}

    for ax, (config_name, ratios) in zip(axes, ratio_data.items()):
        ax.hist(ratios, bins=bins, color=colors[config_name], alpha=0.75, density=True)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="ratio = 1")
        ax.set_xscale("log")
        ax.set_xlabel(r"$N_k\,/\,N_{k+1}$ impression ratio", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(titles[config_name], fontsize=12)
        pct_bal = np.mean((ratios >= 0.5) & (ratios <= 2.0)) * 100
        pct_ext = np.mean(ratios > 10) * 100
        med     = np.median(ratios)
        p90     = np.percentile(ratios, 90)
        ax.text(0.97, 0.95,
                f"[0.5, 2.0]: {pct_bal:.0f}%\n>10: {pct_ext:.0f}%\n"
                f"median: {med:.1f}\np90: {p90:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp5_ratio_distribution.png"), dpi=150)
    plt.close()
    print("Ratio distribution plot saved.")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    true_bias = true_propensities(NUM_POSITIONS, ETA)

    # --- Ratio distributions ---
    print("Computing ratio distributions...")
    rng0 = np.random.default_rng(42)
    ratio_data = {}
    for config_name, alloc in CONFIGS.items():
        # Use a larger sample for stable histograms
        dfs = [generate_dataset(alloc, true_bias, rng0) for _ in range(20)]
        df_all = pd.concat(dfs, ignore_index=True)
        ratios = compute_ratio_distribution(df_all)
        ratio_data[config_name] = ratios
        pct_bal = np.mean((ratios >= 0.5) & (ratios <= 2.0)) * 100
        pct_ext = np.mean(ratios > 10) * 100
        print(f"  {config_name:12s}: n={len(ratios):,}  "
              f"[0.5,2.0]={pct_bal:.1f}%  >10={pct_ext:.1f}%  "
              f"median={np.median(ratios):.2f}  p90={np.percentile(ratios,90):.1f}")

    plot_ratio_distributions(ratio_data, RESULTS_DIR)

    # --- MSE comparison ---
    print("\nRunning MSE sweep (500 trials × 2 configs)...")
    mse_results = {}
    for config_name, alloc in tqdm(CONFIGS.items()):
        mse_results[config_name] = run_mse_sweep(alloc, rng_seed=0)

    print("\n--- MSE results ---")
    for config_name, res in mse_results.items():
        print(f"\n{config_name}:")
        for s in SCHEMES:
            print(f"  {SCHEME_LABELS[s]:12s}: MSE={res[s]['mse']:.5f}  "
                  f"Var={res[s]['variance']:.5f}  Bias2={res[s]['bias2']:.5f}")
        red = (1 - res["harmonic"]["mse"] / res["original"]["mse"]) * 100
        print(f"  Harmonic MSE reduction vs original: {red:+.1f}%")

    # Save CSV
    records = []
    for config_name, res in mse_results.items():
        for s in SCHEMES:
            records.append({"config": config_name, "scheme": s, **res[s]})
    pd.DataFrame(records).to_csv(
        os.path.join(RESULTS_DIR, "exp5_results.csv"), index=False)
    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Experiment 5 done.")
