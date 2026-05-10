"""
Experiment 4b: Sample-efficiency heatmap across (s, r).

Generalises experiment4 from a single (s=0.80, r=50) cell to a heatmap.
For each (split, ratio), we estimate the data-equivalence factor

    DEF(s, r) = b_orig(s, r) / b_harm(s, r)

where b_scheme(s, r) is the smallest base-impression value at which the
scheme's MSE falls below a fixed target (median of harmonic MSE at b=2 across
the (s, r) grid -- a comparable cross-grid threshold).

We sweep base_impressions on a logarithmic-ish grid and interpolate b_scheme
linearly in log(b) to find where MSE crosses the threshold for each scheme.

Inputs/Outputs:
    images/exp4b_efficiency_heatmap/
      - exp4b_results.csv       full MSE vs b grid
      - exp4b_def.csv           data-equivalence factor per (s, r)
      - exp4b_def_heatmap.png   heatmap visualisation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import (
    AdjacentChainEstimator,
)

RESULTS_DIR = "images/exp4b_efficiency_heatmap"
NUM_POSITIONS = 10
ETA = 1.0
NUM_TRIALS = 200          # per (s, r, b) cell -- keep moderate for grid
NUM_DOCS = 50

SPLITS = [0.50, 0.70, 0.80, 0.90, 0.95]
RATIOS = [1, 5, 10, 20, 50, 100]
BASE_IMPRESSIONS = [2, 5, 10, 20, 50, 100]
SCHEMES = ["original", "harmonic"]


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(true_bias, num_docs, base_impressions, imbalance_ratio,
                     traffic_split, rng):
    rows = []
    doc_id = 0
    n_anchor = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor
    for pos in range(1, len(true_bias)):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        for _ in range(n_anchor):
            rel = rng.uniform(0.1, 0.5)
            n_k = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                          base_impressions * imbalance_ratio * 2.0)))
            n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                          "impressions": n_k, "clicks": int(rng.binomial(n_k, rel * p_k))})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                          "impressions": n_kp, "clicks": int(rng.binomial(n_kp, rel * p_kp))})
            doc_id += 1
        for _ in range(n_regular):
            rel = rng.uniform(0.1, 0.5)
            n_k = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            n_kp = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                          "impressions": n_k, "clicks": int(rng.binomial(n_k, rel * p_k))})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                          "impressions": n_kp, "clicks": int(rng.binomial(n_kp, rel * p_kp))})
            doc_id += 1
    return pd.DataFrame(rows)


def run_full_grid(seed: int = 0) -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []
    total = len(SPLITS) * len(RATIOS) * len(BASE_IMPRESSIONS)
    pbar = tqdm(total=total, desc="MSE grid (s, r, b)")
    for split in SPLITS:
        for ratio in RATIOS:
            for base_imp in BASE_IMPRESSIONS:
                rng = np.random.default_rng(seed + int(1e6 * split + 1e3 * ratio + base_imp))
                trial_results = {s: [] for s in SCHEMES}
                for _ in range(NUM_TRIALS):
                    df = generate_dataset(true_bias, NUM_DOCS, base_imp,
                                           ratio, split, rng)
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
                    arr = np.array(trial_results[s])
                    mean_est = np.nanmean(arr, axis=0)
                    var_est = np.nanvar(arr, axis=0)
                    bias2 = (mean_est - true_bias) ** 2
                    records.append({
                        "split": split, "ratio": ratio, "base_imp": base_imp,
                        "scheme": s,
                        "mse": float(np.nanmean(var_est + bias2)),
                        "variance": float(np.nanmean(var_est)),
                        "bias2": float(np.nanmean(bias2)),
                    })
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


def crossing(b_values: np.ndarray, mse_values: np.ndarray, target: float) -> float:
    """Smallest b at which mse drops below target, via linear interp in log b."""
    b = np.log(np.asarray(b_values, dtype=float))
    m = np.asarray(mse_values, dtype=float)
    if np.all(m > target):
        return float(np.exp(b[-1]))  # never reaches; report highest b
    if np.all(m <= target):
        return float(np.exp(b[0]))   # always reaches; report lowest b
    # find first crossing
    for i in range(1, len(m)):
        if m[i - 1] > target >= m[i]:
            # interp in log b
            t = (target - m[i - 1]) / (m[i] - m[i - 1] + 1e-30)
            t = np.clip(t, 0.0, 1.0)
            return float(np.exp(b[i - 1] + t * (b[i] - b[i - 1])))
    return float(np.exp(b[-1]))


def compute_def(results: pd.DataFrame) -> pd.DataFrame:
    """For each (s, r), compute b_required for a fixed cross-grid MSE target."""
    # target: median harmonic MSE at b=2 across (s, r). A consistent cross-grid threshold.
    target = float(results[(results["scheme"] == "harmonic") & (results["base_imp"] == 2)]["mse"].median())
    rows = []
    for split in SPLITS:
        for ratio in RATIOS:
            sub = results[(results["split"] == split) & (results["ratio"] == ratio)]
            sub_h = sub[sub["scheme"] == "harmonic"].sort_values("base_imp")
            sub_o = sub[sub["scheme"] == "original"].sort_values("base_imp")
            b_h = crossing(sub_h["base_imp"].values, sub_h["mse"].values, target)
            b_o = crossing(sub_o["base_imp"].values, sub_o["mse"].values, target)
            rows.append({"split": split, "ratio": ratio,
                         "b_harmonic": b_h, "b_original": b_o,
                         "DEF": b_o / b_h, "target": target})
    return pd.DataFrame(rows)


def plot_heatmap(def_df: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    pivot = def_df.pivot(index="split", columns="ratio", values="DEF")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn",
                   vmin=1.0, vmax=max(2.0, float(np.nanmax(pivot.values))))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Data-equivalence factor (b_original / b_harmonic)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(r) for r in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{s:.2f}" for s in pivot.index])
    ax.set_xlabel("Imbalance ratio r")
    ax.set_ylabel("Traffic split s")
    ax.set_title("Data-equivalence factor of harmonic vs. original (target: median harmonic@b=2)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "exp4b_def_heatmap.png", dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Running grid over splits={SPLITS}, ratios={RATIOS}, "
          f"base_imps={BASE_IMPRESSIONS}, trials={NUM_TRIALS}.")
    results = run_full_grid()
    results.to_csv(Path(RESULTS_DIR) / "exp4b_results.csv", index=False)
    def_df = compute_def(results)
    def_df.to_csv(Path(RESULTS_DIR) / "exp4b_def.csv", index=False)
    plot_heatmap(def_df, RESULTS_DIR)
    print("\n--- Data-equivalence factor (DEF) per (s, r) ---")
    print(def_df.to_string(index=False))
    print(f"\nMedian DEF = {def_df['DEF'].median():.2f}")
    print(f"Mean   DEF = {def_df['DEF'].mean():.2f}")
    print(f"Min/Max  = {def_df['DEF'].min():.2f}/{def_df['DEF'].max():.2f}")
    print(f"\nSaved CSV + heatmap under {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
