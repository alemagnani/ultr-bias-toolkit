"""
Experiment 2: PBM Violations (Synthetic)

Generates clicks under three models that violate the Position-Based Model:
  (a) Trust bias (Vardasbi et al., CIKM 2020):
      P(C=1|q,d,k) = p_k * rel(q,d) + epsilon_k * (1 - rel(q,d))
  (b) Cascade model:
      User scans top-down, stops after first click.
  (c) Position-dependent relevance:
      P(C=1|q,d,k) = p_k * rel(q,d,k)  where rel varies with position.

Key insight: ALL intervention harvesting methods share the same bias under
model misspecification; our harmonic weighting has lower VARIANCE -> lower MSE.

Usage:
    python paper/experiment2_pbm_violations.py

Outputs:
    images/exp2_pbm_violations/ with PNG plots and a summary CSV.
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
NUM_DOCS = 50
BASE_IMPRESSIONS = 20
IMBALANCE_RATIO = 50.0   # N_k / N_kp for anchor docs
TRAFFIC_SPLIT = 0.80     # fraction of anchor docs

RESULTS_DIR = "images/exp2_pbm_violations"

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

VIOLATION_MODELS = ["trust", "cascade", "pos_dependent_rel"]
MODEL_LABELS = {
    "trust": "Trust Bias",
    "cascade": "Cascade",
    "pos_dependent_rel": "Pos-Dep. Relevance",
}


# ---------------------------------------------------------------------------
# True propensities (PBM baseline for reference)
# ---------------------------------------------------------------------------

def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


# ---------------------------------------------------------------------------
# Click models
# ---------------------------------------------------------------------------

def _assign_impressions(
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    num_docs: int,
    rng: np.random.Generator,
):
    """Anchor-item impression model.

    Returns lists of (n_k, n_kp) per doc.  ``traffic_split`` fraction are
    anchor docs with large N_k and small N_kp; the rest are regular docs with
    balanced impressions.  Identical model to Experiment 1.
    """
    n_anchor = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor
    pairs = []
    for _ in range(n_anchor):
        n_k = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                     base_impressions * imbalance_ratio * 2.0)))
        n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
        pairs.append((n_k, n_kp))
    for _ in range(n_regular):
        n = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
        pairs.append((n, n))
    return pairs


def generate_trust_bias(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
    epsilon_scale: float = 0.05,
) -> pd.DataFrame:
    """Trust bias: P(C=1|q,d,k) = p_k*rel + epsilon_k*(1-rel).

    epsilon_k = epsilon_scale * p_k (trust effect decays with position).
    Uses anchor-item impression model for imbalance.
    """
    num_positions = len(true_bias)
    epsilon = epsilon_scale * true_bias

    rows = []
    doc_id = 0
    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        eps_k = epsilon[pos - 1]
        eps_kp = epsilon[pos]

        imp_pairs = _assign_impressions(base_impressions, imbalance_ratio,
                                        traffic_split, num_docs, rng)
        for n_k, n_kp in imp_pairs:
            rel = rng.uniform(0.1, 0.5)
            prob_k = p_k * rel + eps_k * (1 - rel)
            prob_kp = p_kp * rel + eps_kp * (1 - rel)
            c_k = int(rng.binomial(n_k, min(prob_k, 1.0)))
            c_kp = int(rng.binomial(n_kp, min(prob_kp, 1.0)))
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos, "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos + 1, "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


def generate_cascade(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Cascade model: user scans top-down, stops after first click.

    Uses anchor-item impression model for imbalance.
    """
    num_positions = len(true_bias)
    rows = []
    doc_id = 0

    for pos in range(1, num_positions):
        imp_pairs = _assign_impressions(base_impressions, imbalance_ratio,
                                        traffic_split, num_docs, rng)
        for n_k, n_kp in imp_pairs:
            rel = rng.uniform(0.1, 0.5)
            reach_k = np.prod([1 - true_bias[j] * rel for j in range(pos - 1)])
            reach_kp = np.prod([1 - true_bias[j] * rel for j in range(pos)])
            prob_k = true_bias[pos - 1] * rel * reach_k
            prob_kp = true_bias[pos] * rel * reach_kp
            c_k = int(rng.binomial(n_k, min(prob_k, 1.0)))
            c_kp = int(rng.binomial(n_kp, min(prob_kp, 1.0)))
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos, "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos + 1, "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


def generate_pos_dependent_rel(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
    pos_rel_scale: float = 0.3,
) -> pd.DataFrame:
    """Position-dependent relevance: rel(q,d,k) = rel0 * (1 + noise_k).

    Uses anchor-item impression model for imbalance.
    """
    num_positions = len(true_bias)
    pos_multipliers = 1.0 + rng.normal(0, pos_rel_scale, num_positions)
    pos_multipliers = np.clip(pos_multipliers, 0.1, 3.0)

    rows = []
    doc_id = 0
    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        mult_k = pos_multipliers[pos - 1]
        mult_kp = pos_multipliers[pos]

        imp_pairs = _assign_impressions(base_impressions, imbalance_ratio,
                                        traffic_split, num_docs, rng)
        for n_k, n_kp in imp_pairs:
            rel0 = rng.uniform(0.1, 0.5)
            prob_k = p_k * rel0 * mult_k
            prob_kp = p_kp * rel0 * mult_kp
            c_k = int(rng.binomial(n_k, min(prob_k, 1.0)))
            c_kp = int(rng.binomial(n_kp, min(prob_kp, 1.0)))
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos, "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id,
                         "position": pos + 1, "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


DATA_GENERATORS = {
    "trust": generate_trust_bias,
    "cascade": generate_cascade,
    "pos_dependent_rel": generate_pos_dependent_rel,
}


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trials(
    model: str,
    true_bias: np.ndarray,
    num_trials: int,
    seed: int = 0,
) -> dict:
    """Run `num_trials` and return per-scheme estimate arrays."""
    rng = np.random.default_rng(seed)
    gen_fn = DATA_GENERATORS[model]
    results = {s: [] for s in SCHEMES}

    for _ in range(num_trials):
        df = gen_fn(true_bias, NUM_DOCS, BASE_IMPRESSIONS,
                    IMBALANCE_RATIO, TRAFFIC_SPLIT, rng)
        for s in SCHEMES:
            est = AdjacentChainEstimator(weighting=s)
            try:
                r = est(df, query_col="query_id", doc_col="doc_id",
                        imps_col="impressions", clicks_col="clicks")
                vals = r.set_index("position")["examination"].values
                if len(vals) != len(true_bias):
                    vals = np.full(len(true_bias), np.nan)
            except Exception:
                vals = np.full(len(true_bias), np.nan)
            results[s].append(vals)

    return {s: np.array(v) for s, v in results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []

    for model in tqdm(VIOLATION_MODELS, desc="PBM violation models"):
        trial_results = run_trials(model, true_bias, NUM_TRIALS)

        for s in SCHEMES:
            arr = trial_results[s]
            mean_est = np.nanmean(arr, axis=0)
            var_est = np.nanvar(arr, axis=0)
            bias2 = (mean_est - true_bias) ** 2
            mse = var_est + bias2

            records.append({
                "model": model,
                "scheme": s,
                "mse": float(np.nanmean(mse)),
                "variance": float(np.nanmean(var_est)),
                "bias2": float(np.nanmean(bias2)),
            })

    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    metrics = ["mse", "variance", "bias2"]
    metric_labels = {"mse": "MSE", "variance": "Variance", "bias2": r"Bias$^2$"}

    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for ax, model in zip(axes, VIOLATION_MODELS):
            sub = results[results["model"] == model]
            vals = [sub[sub["scheme"] == s][metric].values[0] for s in SCHEMES]
            colors = [SCHEME_COLORS[s] for s in SCHEMES]
            bars = ax.bar(range(len(SCHEMES)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(SCHEMES)))
            ax.set_xticklabels([SCHEME_LABELS[s] for s in SCHEMES], rotation=30, ha="right",
                               fontsize=9)
            ax.set_title(MODEL_LABELS[model], fontsize=12)
            ax.set_ylabel(metric_labels[metric], fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"{metric_labels[metric]} under PBM violations "
            f"(imbalance={int(IMBALANCE_RATIO)}:1, split={int(TRAFFIC_SPLIT*100)}/{int((1-TRAFFIC_SPLIT)*100)})",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"exp2_{metric}.png"), dpi=150)
        plt.close()

    # MSE decomposition: stacked bar (bias^2 + variance)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, model in zip(axes, VIOLATION_MODELS):
        sub = results[results["model"] == model]
        vars_ = [sub[sub["scheme"] == s]["variance"].values[0] for s in SCHEMES]
        biases = [sub[sub["scheme"] == s]["bias2"].values[0] for s in SCHEMES]
        x = range(len(SCHEMES))
        ax.bar(x, biases, color="steelblue", alpha=0.8, label=r"Bias$^2$")
        ax.bar(x, vars_, bottom=biases, color="salmon", alpha=0.8, label="Variance")
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels([SCHEME_LABELS[s] for s in SCHEMES], rotation=30, ha="right",
                           fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_ylabel("MSE decomposition", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("MSE = Bias² + Variance (all methods share same bias under misspecification)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp2_mse_decomposition.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_latex_table(results: pd.DataFrame) -> None:
    print("\n--- LaTeX Table: MSE / Variance / Bias² by model and scheme ---")
    print(r"\begin{tabular}{ll" + "r" * 3 + "}")
    print(r"\hline")
    print(r"Model & Scheme & MSE & Variance & Bias$^2$ \\")
    print(r"\hline")
    for model in VIOLATION_MODELS:
        for i, s in enumerate(SCHEMES):
            row = results[(results["model"] == model) & (results["scheme"] == s)].iloc[0]
            prefix = MODEL_LABELS[model] if i == 0 else ""
            print(f"{prefix:<20} & {SCHEME_LABELS[s]:<15} & "
                  f"{row['mse']:.5f} & {row['variance']:.5f} & {row['bias2']:.5f} \\\\")
        print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    print("Running Experiment 2: PBM Violations")
    print(f"  Models: {VIOLATION_MODELS}")
    print(f"  Trials: {NUM_TRIALS}, imbalance ratio: {IMBALANCE_RATIO}, split: {TRAFFIC_SPLIT}")

    results = run_experiment()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp2_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_results(results, RESULTS_DIR)
    print_latex_table(results)
    print("\nExperiment 2 done.")
