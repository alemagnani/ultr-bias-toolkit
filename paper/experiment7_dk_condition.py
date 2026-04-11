"""
Experiment 7: Verify D_k >= 0 Condition (Theorem 6)

For each experimental configuration, computes D_k and D_{k'} for harmonic
weights and verifies:
  - D_k >= 0 holds in all systematic imbalance configurations
  - D_k can fail under adversarial mixed imbalance

Usage:
    python paper/experiment7_dk_condition.py

Outputs:
    images/exp7_dk_condition/ with PNG plots and a summary CSV.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.weighting import variance_decomposition

RESULTS_DIR = "images/exp7_dk_condition"
ETA = 1.0
NUM_POSITIONS = 10
NUM_DOCS = 100


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def systematic_imbalance_counts(
    num_docs: int,
    base: int,
    ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> tuple:
    """Generate N_k, N_kp under SYSTEMATIC (anchor-item) imbalance.

    ``traffic_split`` fraction of docs are "anchor" items with many
    impressions at k (large N_k) and few at k' (small N_kp).
    The rest are "regular" items with approximately balanced, small impressions.

    D_k >= 0 holds because harmonic down-weights anchor items and
    up-weights regular items.  Regular items have smaller N_k, so
    concentrating weight on them increases V_k (more noise per unit weight).
    """
    n_anchor = int(num_docs * traffic_split)
    n_regular = num_docs - n_anchor

    # Anchor items: large N_k, small N_kp
    N_k_anc = rng.integers(int(base * ratio * 0.5), int(base * ratio * 2) + 1,
                            size=n_anchor).astype(float)
    N_kp_anc = rng.integers(1, base + 1, size=n_anchor).astype(float)

    # Regular items: small and roughly balanced
    N_k_reg = rng.integers(base // 2, base * 2 + 1, size=n_regular).astype(float)
    N_kp_reg = (N_k_reg * rng.uniform(0.4, 0.9, size=n_regular)).clip(min=1.0)

    N_k = np.concatenate([N_k_anc, N_k_reg])
    N_kp = np.concatenate([N_kp_anc, N_kp_reg])
    return N_k, N_kp


def mixed_adversarial_counts(
    num_docs: int,
    base: int,
    ratio: float,
    rng: np.random.Generator,
) -> tuple:
    """Generate N_k, N_kp under adversarial MIXED imbalance.

    The imbalance direction is MIXED (some N_k >> N_kp, some reversed),
    with VARYING ratios per doc.  This can make harmonic weights
    concentrate weight unfavourably, causing D_k < 0 (harmonic increases V_k
    at one position) and potentially total > 0 (harmonic worse overall).

    Construction: 40% have large N_k (ratio * base), 60% have small N_k
    (base / ratio), creating asymmetric mixing.  Ratios also vary log-uniformly.
    """
    n_large = int(num_docs * 0.4)
    n_small = num_docs - n_large

    # Large-N_k docs: N_k = ratio * base (variable), N_kp = base (variable, smaller)
    N_k_large = np.exp(rng.uniform(np.log(base * ratio * 0.5),
                                    np.log(base * ratio * 2), size=n_large))
    N_kp_large = np.exp(rng.uniform(np.log(base * 0.5),
                                     np.log(base * 1.5), size=n_large))

    # Small-N_k docs: N_k = base (variable), N_kp = ratio * base (variable, larger)
    N_k_small = np.exp(rng.uniform(np.log(base * 0.5),
                                    np.log(base * 1.5), size=n_small))
    N_kp_small = np.exp(rng.uniform(np.log(base * ratio * 0.5),
                                     np.log(base * ratio * 2), size=n_small))

    N_k = np.concatenate([N_k_large, N_k_small])
    N_kp = np.concatenate([N_kp_large, N_kp_small])
    return N_k.clip(min=1.0), N_kp.clip(min=1.0)


def run_experiment() -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    rng = np.random.default_rng(42)
    records = []

    imbalance_ratios = [1, 2, 5, 10, 20, 50, 100]
    traffic_splits = [0.60, 0.80, 0.95]

    for adj_pos in range(1, NUM_POSITIONS):
        p_k = true_bias[adj_pos - 1]
        p_kp = true_bias[adj_pos]
        alpha = (1 - p_k) / max(p_k, 1e-6)
        beta = (1 - p_kp) / max(p_kp, 1e-6)

        # Systematic imbalance
        for ratio in imbalance_ratios:
            for split in traffic_splits:
                N_k, N_kp = systematic_imbalance_counts(
                    NUM_DOCS, base=20, ratio=ratio, traffic_split=split, rng=rng
                )
                decomp = variance_decomposition(N_k, N_kp, alpha, beta)
                records.append({
                    "config": "systematic",
                    "position_k": adj_pos,
                    "imbalance_ratio": ratio,
                    "traffic_split": split,
                    "alpha": alpha,
                    "beta": beta,
                    **decomp,
                })

        # Mixed adversarial
        for ratio in imbalance_ratios:
            N_k, N_kp = mixed_adversarial_counts(NUM_DOCS, base=20, ratio=ratio, rng=rng)
            decomp = variance_decomposition(N_k, N_kp, alpha, beta)
            records.append({
                "config": "adversarial_mixed",
                "position_k": adj_pos,
                "imbalance_ratio": ratio,
                "traffic_split": None,
                "alpha": alpha,
                "beta": beta,
                **decomp,
            })

    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot 1: D_k by imbalance ratio for systematic config (averaged over positions) ---
    sys_df = results[results["config"] == "systematic"].copy()
    adv_df = results[results["config"] == "adversarial_mixed"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for split, color in zip([0.60, 0.80, 0.95], ["blue", "red", "green"]):
        sub = sys_df[sys_df["traffic_split"] == split].groupby("imbalance_ratio")["D_k"].mean()
        axes[0].plot(sub.index, sub.values, marker="o", color=color,
                     label=f"Split {int(split*100)}/{int((1-split)*100)}")

    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, label="D_k=0")
    axes[0].set_xlabel("Imbalance ratio", fontsize=12)
    axes[0].set_ylabel(r"$D_k$ (mean over positions)", fontsize=12)
    axes[0].set_title("Systematic imbalance: D_k ≥ 0 (harmonic improves)", fontsize=12)
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Adversarial
    adv_mean = adv_df.groupby("imbalance_ratio")["D_k"].mean()
    axes[1].plot(adv_mean.index, adv_mean.values, marker="s", color="orange",
                 label="Adversarial mixed (mean)")
    adv_min = adv_df.groupby("imbalance_ratio")["D_k"].min()
    axes[1].plot(adv_min.index, adv_min.values, marker="v", color="brown",
                 label="Adversarial mixed (min)")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, label="D_k=0")
    axes[1].set_xlabel("Imbalance ratio", fontsize=12)
    axes[1].set_ylabel(r"$D_k$", fontsize=12)
    axes[1].set_title("Adversarial mixed: D_k can be < 0", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Theorem 6: Variance decomposition — D_k condition", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp7_dk_condition.png"), dpi=150)
    plt.close()

    # --- Plot 2: Total variance reduction (term1 + term2) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for split, color in zip([0.60, 0.80, 0.95], ["blue", "red", "green"]):
        sub = sys_df[sys_df["traffic_split"] == split].groupby("imbalance_ratio")["total"].mean()
        axes[0].plot(sub.index, sub.values, marker="o", color=color,
                     label=f"Split {int(split*100)}/{int((1-split)*100)}")

    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Imbalance ratio", fontsize=12)
    axes[0].set_ylabel("Total variance reduction (harmonic - original)", fontsize=11)
    axes[0].set_title("Systematic: always negative (harmonic better)", fontsize=12)
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    adv_total = adv_df.groupby("imbalance_ratio")["total"].mean()
    axes[1].plot(adv_total.index, adv_total.values, marker="s", color="orange",
                 label="Adversarial mixed (mean)")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Imbalance ratio", fontsize=12)
    axes[1].set_ylabel("Total variance reduction", fontsize=11)
    axes[1].set_title("Adversarial mixed: can be positive (harmonic worse)", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Variance reduction: harmonic vs original (Theorem 6)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp7_variance_reduction.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_summary(results: pd.DataFrame) -> None:
    tol = 1e-10
    sys_df = results[results["config"] == "systematic"]
    adv_df = results[results["config"] == "adversarial_mixed"]

    def pct(n, total):
        return f"{n}/{total} ({100*n/total:.1f}%)"

    print("\n--- Theorem 6 Verification ---")
    print("Cauchy-Schwarz (D_k + D_kp <= 0, should be 0 violations):")
    n_cs_sys = (sys_df["D_k"] + sys_df["D_kp"] > tol).sum()
    n_cs_adv = (adv_df["D_k"] + adv_df["D_kp"] > tol).sum()
    print(f"  Systematic violations: {n_cs_sys}/{len(sys_df)} [expected 0]")
    print(f"  Adversarial violations: {n_cs_adv}/{len(adv_df)} [expected 0]")
    print()
    print("Systematic anchor-item imbalance:")
    n_dk_pos = (sys_df["D_k"] >= -tol).sum()
    n_worse = (sys_df["total"] > tol).sum()
    print(f"  D_k >= 0 (sufficient condition for term2 <= 0): {pct(n_dk_pos, len(sys_df))}")
    print(f"  Total > 0 (harmonic worse): {n_worse}/{len(sys_df)} [expected 0]")
    print()
    print("Adversarial mixed imbalance:")
    n_dk_neg = (adv_df["D_k"] < -tol).sum()
    n_worse_adv = (adv_df["total"] > tol).sum()
    print(f"  D_k < 0 (sufficient condition violated): {pct(n_dk_neg, len(adv_df))}")
    print(f"  Total > 0 (harmonic worse): {n_worse_adv}/{len(adv_df)}")


if __name__ == "__main__":
    print("Running Experiment 7: D_k condition verification")
    results = run_experiment()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "exp7_results.csv"), index=False)
    print(f"Results saved to {RESULTS_DIR}/exp7_results.csv")

    plot_results(results, RESULTS_DIR)
    print_summary(results)
    print("\nExperiment 7 done.")
