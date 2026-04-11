"""
Oracle Adaptive Experiment

Asks: if we "cheat" and use the TRUE propensity ratio at each position pair
to set the optimal adaptive weights, how much better can we do than harmonic?

For each adjacent pair (k, k+1) the variance-optimal weight is:
    omega_i* = N_k^i * N_{k'}^i / (alpha * N_{k'}^i + beta * N_k^i)
where alpha = (1-p_k)/p_k and beta = (1-p_{k'})/p_{k'} are known from the
simulation ground truth.  When alpha == beta this reduces to harmonic; when
they differ (e.g. non-adjacent or high-contrast positions) the oracle weight
is strictly better.

This validates the theory without the noise of the two-stage cross-fitting
estimator.

Usage:
    python paper/experiment_oracle.py

Outputs:
    images/exp_oracle/ with a CSV and PNG.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import AdjacentChainEstimator

RESULTS_DIR = "images/exp_oracle"
NUM_POSITIONS = 10
ETA = 1.0
NUM_TRIALS = 500
NUM_DOCS = 50
BASE_IMPRESSIONS = 20
TRAFFIC_SPLITS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
IMBALANCE_RATIOS = [1, 2, 5, 10, 20, 50, 100]

SCHEMES = ["original", "min", "harmonic"]
SCHEME_LABELS = {
    "original": "Original",
    "min": "Min",
    "harmonic": "Harmonic",
    "oracle": "Oracle adaptive",
}
SCHEME_COLORS = {
    "original": "black",
    "min": "blue",
    "harmonic": "red",
    "oracle": "purple",
}


# ---------------------------------------------------------------------------
# Shared helpers
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
    """Anchor-item model: same as Experiment 1."""
    rows = []
    doc_id = 0
    num_positions = len(true_bias)
    n_anchor = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor

    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]

        for _ in range(n_anchor):
            rel = rng.uniform(0.1, 0.5)
            n_k = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                          base_impressions * imbalance_ratio * 2.0)))
            n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

        for _ in range(n_regular):
            rel = rng.uniform(0.1, 0.5)
            n_k = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            n_kp = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Oracle estimator
# ---------------------------------------------------------------------------

def oracle_estimate(df: pd.DataFrame, true_bias: np.ndarray) -> np.ndarray:
    """Estimate propensities using oracle (true) alpha/beta at each adjacent pair.

    For pair (k, k+1):
        alpha = (1 - p_k) / p_k
        beta  = (1 - p_{k+1}) / p_{k+1}
        omega_i = N_k * N_{k+1} / (alpha * N_{k+1} + beta * N_k)

    Adjacent ratios are chained into absolute propensities exactly as in
    AdjacentChainEstimator (normalised so p[0] = 1).
    """
    n = len(true_bias)
    eps = 1e-9

    # Build per-position lookup for each doc
    df_by_pos = {pos: grp.set_index("doc_id") for pos, grp in df.groupby("position")}

    log_p = np.zeros(n)  # log propensity accumulator (p[0] = 1 -> log = 0)

    for k in range(1, n):   # pair: position k (1-indexed) and k+1
        pos_k = k
        pos_kp = k + 1

        if pos_k not in df_by_pos or pos_kp not in df_by_pos:
            continue

        dk = df_by_pos[pos_k]
        dkp = df_by_pos[pos_kp]

        # Docs present at both positions
        common = dk.index.intersection(dkp.index)
        if len(common) == 0:
            continue

        N_k  = dk.loc[common, "impressions"].values.astype(float)
        N_kp = dkp.loc[common, "impressions"].values.astype(float)
        C_k  = dk.loc[common, "clicks"].values.astype(float)
        C_kp = dkp.loc[common, "clicks"].values.astype(float)

        p_k  = true_bias[pos_k - 1]
        p_kp = true_bias[pos_kp - 1]
        alpha = (1.0 - p_k)  / max(p_k,  eps)
        beta  = (1.0 - p_kp) / max(p_kp, eps)

        denom = alpha * N_kp + beta * N_k
        denom = np.where(denom < eps, eps, denom)
        omega = N_k * N_kp / denom

        W = omega.sum()
        if W < eps:
            continue

        A = np.sum(omega * C_k  / np.where(N_k  < eps, eps, N_k))
        B = np.sum(omega * C_kp / np.where(N_kp < eps, eps, N_kp))

        if B < eps:
            continue

        # ratio = p_k / p_{k+1}  =>  p_{k+1} = p_k * B/A
        log_p[pos_kp - 1] = log_p[pos_k - 1] + np.log(max(B / A, eps))

    p_hat = np.exp(log_p)
    p_hat = p_hat / p_hat[0]   # normalise p[0] = 1
    return p_hat


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    traffic_splits=TRAFFIC_SPLITS,
    imbalance_ratios=IMBALANCE_RATIOS,
    num_trials=NUM_TRIALS,
    seed=0,
) -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    all_schemes = SCHEMES + ["oracle"]
    records = []

    total = len(traffic_splits) * len(imbalance_ratios)
    pbar = tqdm(total=total, desc="Oracle sweep")

    for split in traffic_splits:
        for ratio in imbalance_ratios:
            rng = np.random.default_rng(seed)
            trial_results = {s: [] for s in all_schemes}

            for _ in range(num_trials):
                df = generate_dataset(true_bias, NUM_DOCS, BASE_IMPRESSIONS,
                                      ratio, split, rng)

                # Standard schemes via AdjacentChainEstimator
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

                # Oracle adaptive
                try:
                    vals = oracle_estimate(df, true_bias)
                except Exception:
                    vals = np.full(len(true_bias), np.nan)
                trial_results["oracle"].append(vals)

            for s in all_schemes:
                arr = np.array(trial_results[s])
                mean_est  = np.nanmean(arr, axis=0)
                var_est   = np.nanvar(arr, axis=0)
                bias2     = (mean_est - true_bias) ** 2
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
    all_schemes = SCHEMES + ["oracle"]
    split_val = 0.80

    sub = results[results["traffic_split"] == split_val]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in all_schemes:
        d = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        axes[0].plot(d["imbalance_ratio"], d["variance"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
        axes[1].plot(d["imbalance_ratio"], d["mse"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Imbalance ratio", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} vs. Imbalance (split={split_val})", fontsize=13)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Oracle adaptive (true propensities) vs practical schemes", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp_oracle_variance.png"), dpi=150)
    plt.close()

    # Variance reduction relative to harmonic
    fig, ax = plt.subplots(figsize=(10, 6))
    harm = results[results["scheme"] == "harmonic"][
        ["traffic_split", "imbalance_ratio", "variance"]
    ].rename(columns={"variance": "var_harm"})

    for s in ["original", "min", "oracle"]:
        d = results[results["scheme"] == s].merge(harm, on=["traffic_split", "imbalance_ratio"])
        d["reduction_vs_harm"] = (1 - d["variance"] / d["var_harm"]) * 100
        d_fixed = d[d["traffic_split"] == split_val].sort_values("imbalance_ratio")
        ax.plot(d_fixed["imbalance_ratio"], d_fixed["reduction_vs_harm"],
                marker="o", color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])

    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="Harmonic (reference)")
    ax.set_xlabel("Imbalance ratio", fontsize=12)
    ax.set_ylabel("Variance change vs. harmonic (%)", fontsize=12)
    ax.set_title(f"Variance relative to harmonic (split={split_val})", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp_oracle_vs_harmonic.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_summary(results: pd.DataFrame) -> None:
    sub = results[results["traffic_split"] == 0.80]
    all_schemes = SCHEMES + ["oracle"]
    print("\n--- Variance at split=80/20 ---")
    print(f"{'Scheme':<20} " + " ".join(f"r={r:>4}" for r in IMBALANCE_RATIOS))
    for s in all_schemes:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " ".join(f"{v:.5f}" for v in row["variance"].values)
        print(f"{SCHEME_LABELS[s]:<20} {vals}")

    print("\n--- Oracle improvement over harmonic (variance %, split=80/20) ---")
    for ratio in IMBALANCE_RATIOS:
        h = sub[(sub["scheme"] == "harmonic") & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        o = sub[(sub["scheme"] == "oracle")   & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        print(f"  ratio={ratio:>4}: harmonic={h:.5f}  oracle={o:.5f}  "
              f"improvement={(1-o/h)*100:+.1f}%")


if __name__ == "__main__":
    print("Running Oracle Experiment")
    print(f"  Positions: {NUM_POSITIONS}, eta={ETA}, trials={NUM_TRIALS}")
    print(f"  Traffic splits: {TRAFFIC_SPLITS}")
    print(f"  Imbalance ratios: {IMBALANCE_RATIOS}")

    results = run_sweep()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp_oracle_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_results(results, RESULTS_DIR)
    print_summary(results)
    print("\nOracle experiment done.")
