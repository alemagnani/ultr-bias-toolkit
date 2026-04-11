"""
Baidu-ULTR Dataset Statistics

Computes statistics to characterise how suitable the Baidu-ULTR dataset is
for impression-imbalance-based intervention harvesting.

Key questions:
  1. How many impressions per (query, doc, position) cell?
  2. What fraction of (query, doc) pairs appear at 2+ positions?
  3. For pairs in adjacent interventional sets: what is the N_k/N_{k+1} ratio?

Expected finding: most cells have N=1 or very few observations; the severe
imbalance regime (N_k >> N_{k'}) that motivates harmonic weighting is absent.

Usage:
    python paper/experiment_baidu_stats.py [--cache-dir PATH] [--max-rows N]

Outputs:
    images/exp_baidu_stats/  with CSV summary and PNG histograms.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "images/exp_baidu_stats"
FEATURE_URL = (
    "https://huggingface.co/datasets/philipphager/baidu-ultr_baidu-mlm-ctr"
    "/resolve/main/parts/train-features.feather"
)
DEFAULT_CACHE = os.path.expanduser("~/.cache/baidu_ultr")
MAX_POSITIONS = 10   # only keep positions 1..10 (standard IH regime)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cache_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    feature_path = os.path.join(cache_dir, "train-features.feather")

    cols = ["query_md5", "url_md5", "position", "click"]

    if not os.path.exists(feature_path):
        print(f"Feather file not found at {feature_path}.")
        print(f"Attempting to download from HuggingFace …")
        try:
            df = pd.read_feather(FEATURE_URL, columns=cols)
            df.to_feather(feature_path)
            print(f"Saved to {feature_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually and place at:", feature_path)
            sys.exit(1)
    else:
        print(f"Loading from {feature_path} …")
        df = pd.read_feather(feature_path, columns=cols)

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"Using first {max_rows:,} rows.")

    print(f"Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    # Filter to standard top-10 positions
    df = df[df["position"].between(1, MAX_POSITIONS)].copy()
    print(f"Rows after filtering to positions 1-{MAX_POSITIONS}: {len(df):,}")

    # -----------------------------------------------------------------------
    # 1. Aggregate to (query, doc, position) cells
    # -----------------------------------------------------------------------
    cell = (
        df.groupby(["query_md5", "url_md5", "position"], observed=True)
        .agg(impressions=("click", "count"), clicks=("click", "sum"))
        .reset_index()
    )
    print(f"Unique (q, d, pos) cells: {len(cell):,}")

    # -----------------------------------------------------------------------
    # 2. Impressions-per-cell distribution
    # -----------------------------------------------------------------------
    imp_dist = cell["impressions"].value_counts().sort_index()
    frac_single = (cell["impressions"] == 1).mean()
    frac_le3    = (cell["impressions"] <= 3).mean()
    median_imp  = cell["impressions"].median()
    mean_imp    = cell["impressions"].mean()

    print(f"\n--- Impressions per (q, d, pos) cell ---")
    print(f"  Fraction with N=1  : {frac_single:.1%}")
    print(f"  Fraction with N≤3  : {frac_le3:.1%}")
    print(f"  Median impressions : {median_imp:.1f}")
    print(f"  Mean   impressions : {mean_imp:.2f}")

    # -----------------------------------------------------------------------
    # 3. Fraction of (q, d) pairs seen at 2+ positions
    # -----------------------------------------------------------------------
    qd_positions = (
        cell.groupby(["query_md5", "url_md5"], observed=True)["position"]
        .nunique()
        .reset_index()
        .rename(columns={"position": "n_positions"})
    )
    n_qd_total      = len(qd_positions)
    n_qd_multi      = (qd_positions["n_positions"] >= 2).sum()
    frac_qd_multi   = n_qd_multi / n_qd_total

    print(f"\n--- (query, doc) pair coverage ---")
    print(f"  Total (q, d) pairs          : {n_qd_total:,}")
    print(f"  Pairs at 2+ positions       : {n_qd_multi:,} ({frac_qd_multi:.1%})")
    print(f"  Pairs at only 1 position    : {n_qd_total - n_qd_multi:,} ({1-frac_qd_multi:.1%})")

    # -----------------------------------------------------------------------
    # 4. N_k / N_{k+1} ratio for adjacent pairs in interventional sets
    # -----------------------------------------------------------------------
    # For each (q, d) appearing at both position k and k+1, compute the ratio
    ratios = []
    for pos_k in range(1, MAX_POSITIONS):
        pos_kp = pos_k + 1
        left  = cell[cell["position"] == pos_k][["query_md5", "url_md5", "impressions"]].rename(
            columns={"impressions": "N_k"})
        right = cell[cell["position"] == pos_kp][["query_md5", "url_md5", "impressions"]].rename(
            columns={"impressions": "N_kp"})
        merged = left.merge(right, on=["query_md5", "url_md5"])
        if len(merged) == 0:
            continue
        merged["ratio"] = merged["N_k"] / merged["N_kp"]
        ratios.append(merged[["N_k", "N_kp", "ratio"]])

    if ratios:
        ratio_df = pd.concat(ratios, ignore_index=True)
        print(f"\n--- N_k / N_{{k+1}} ratio for adjacent-pair docs ---")
        print(f"  Pairs in adjacent interventional sets: {len(ratio_df):,}")
        print(f"  Fraction with ratio in [0.5, 2.0]   : {ratio_df['ratio'].between(0.5, 2.0).mean():.1%}")
        print(f"  Fraction with ratio > 5             : {(ratio_df['ratio'] > 5).mean():.1%}")
        print(f"  Fraction with ratio > 10            : {(ratio_df['ratio'] > 10).mean():.1%}")
        print(f"  Median ratio                        : {ratio_df['ratio'].median():.2f}")
        print(f"  90th-percentile ratio               : {ratio_df['ratio'].quantile(0.90):.2f}")
        print(f"  99th-percentile ratio               : {ratio_df['ratio'].quantile(0.99):.2f}")
    else:
        ratio_df = pd.DataFrame(columns=["N_k", "N_kp", "ratio"])
        print("\nNo adjacent pairs found.")

    return {
        "cell": cell,
        "imp_dist": imp_dist,
        "qd_positions": qd_positions,
        "ratio_df": ratio_df,
        "stats": {
            "n_rows": len(df),
            "n_cells": len(cell),
            "frac_single_impression": frac_single,
            "frac_le3_impressions": frac_le3,
            "median_impressions": median_imp,
            "mean_impressions": mean_imp,
            "n_qd_total": n_qd_total,
            "n_qd_multi_pos": int(n_qd_multi),
            "frac_qd_multi_pos": frac_qd_multi,
            "n_adjacent_pairs": len(ratio_df),
            "frac_ratio_balanced": float(ratio_df["ratio"].between(0.5, 2.0).mean()) if len(ratio_df) else float("nan"),
            "median_ratio": float(ratio_df["ratio"].median()) if len(ratio_df) else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stats(results: dict, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    cell     = results["cell"]
    ratio_df = results["ratio_df"]

    # --- Plot 1: impressions-per-cell histogram (cap at 20) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    cap = 20
    capped = cell["impressions"].clip(upper=cap)
    counts = capped.value_counts().sort_index()
    ax.bar(counts.index, counts.values / counts.values.sum(), color="steelblue", alpha=0.8)
    ax.set_xlabel("Impressions per (query, doc, position) cell", fontsize=12)
    ax.set_ylabel("Fraction of cells", fontsize=12)
    ax.set_title("Impression count distribution — Baidu-ULTR (train)", fontsize=13)
    ax.set_xticks(range(1, cap + 1))
    last_label = f"≥{cap}"
    labels = [str(i) for i in range(1, cap)] + [last_label]
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baidu_impression_dist.png"), dpi=150)
    plt.close()

    # --- Plot 2: N_k / N_{k+1} ratio histogram ---
    if len(ratio_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        log_ratio = np.log10(ratio_df["ratio"].clip(lower=1e-3, upper=1e3))
        ax.hist(log_ratio, bins=60, color="salmon", alpha=0.8, density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="ratio = 1")
        ax.set_xlabel(r"$\log_{10}(N_k / N_{k+1})$", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(r"Impression ratio $N_k / N_{k+1}$ for adjacent-pair docs — Baidu-ULTR", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "baidu_ratio_dist.png"), dpi=150)
        plt.close()

    # --- Plot 3: number-of-positions per (q,d) pair ---
    qd = results["qd_positions"]
    fig, ax = plt.subplots(figsize=(7, 4))
    n_pos_counts = qd["n_positions"].value_counts().sort_index()
    ax.bar(n_pos_counts.index.astype(str), n_pos_counts.values / n_pos_counts.values.sum(),
           color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Number of distinct positions for (query, doc) pair", fontsize=11)
    ax.set_ylabel("Fraction of pairs", fontsize=11)
    ax.set_title("Position coverage per (query, doc) pair — Baidu-ULTR", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baidu_position_coverage.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baidu-ULTR dataset statistics")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE,
                        help="Directory to cache the feather file")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows loaded (for quick testing)")
    args = parser.parse_args()

    df = load_data(args.cache_dir, args.max_rows)
    results = compute_stats(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save scalar stats
    stats_df = pd.DataFrame([results["stats"]])
    stats_path = os.path.join(RESULTS_DIR, "baidu_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStats saved to {stats_path}")

    # Save ratio distribution
    if len(results["ratio_df"]) > 0:
        ratio_path = os.path.join(RESULTS_DIR, "baidu_ratio_dist.csv")
        results["ratio_df"]["ratio"].describe().to_csv(ratio_path)
        print(f"Ratio summary saved to {ratio_path}")

    plot_stats(results, RESULTS_DIR)

    # Print compact paper-ready summary
    s = results["stats"]
    print("\n--- Paper-ready summary ---")
    print(f"  Total (q,d,pos) cells      : {s['n_cells']:,}")
    print(f"  Cells with N=1             : {s['frac_single_impression']:.1%}")
    print(f"  Cells with N≤3             : {s['frac_le3_impressions']:.1%}")
    print(f"  Mean impressions per cell  : {s['mean_impressions']:.2f}")
    print(f"  (q,d) pairs at 2+ pos      : {s['frac_qd_multi_pos']:.1%}")
    print(f"  Adjacent pairs (IH input)  : {s['n_adjacent_pairs']:,}")
    print(f"  Ratio in [0.5, 2] (balanced): {s['frac_ratio_balanced']:.1%}")
    print(f"  Median N_k/N_k+1           : {s['median_ratio']:.2f}")


if __name__ == "__main__":
    main()
