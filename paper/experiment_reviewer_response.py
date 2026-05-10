"""
Reviewer-response experiments. Three additions:

  1. Paired bootstrap CIs on the harmonic-vs-original and harmonic-vs-min
     margins for each headline configuration. Per-trial seed alignment makes
     the comparison paired.

  2. Adversarial (alpha, beta) search. The full delta-method variance is
     alpha * V_k + beta * V_{k'}. Theorem 4 (cs guarantee) bounds V_k+V_{k'};
     for asymmetric (alpha, beta) harmonic could in principle lose. We
     search a grid of count configurations and (alpha, beta) and report any
     case where harmonic > original on the actual surrogate.

  3. Plug-in oracle. Two-stage adaptive: first half of trials fits harmonic
     to get \hat p_k; compute \hat alpha, \hat beta; second half uses
     \hat omega_i^* = N_k^i N_{k'}^i / (\hat alpha N_{k'}^i + \hat beta N_k^i).
     Compare to harmonic and to the true-propensity oracle.

All outputs land in `paper/reviewer_response/`. Closed-form analyses are exact
(no Monte Carlo). Monte Carlo blocks use 500 trials per cell.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultr_bias_toolkit.bias.intervention_harvesting.adjacent_chain import (
    AdjacentChainEstimator,
)

OUT = Path(__file__).resolve().parent / "reviewer_response"
OUT.mkdir(exist_ok=True)

NUM_POSITIONS = 10
ETA = 1.0
NUM_DOCS = 50
BASE_IMP = 20
SCHEMES = ["original", "min", "harmonic"]


# ---------------------------------------------------------------------------
# Data generation (matches experiment1)
# ---------------------------------------------------------------------------

def true_propensities(num_positions=NUM_POSITIONS, eta=ETA):
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(true_bias, num_docs, base_imp, ratio, split, rng):
    rows = []
    for k in range(1, NUM_POSITIONS):
        kp = k + 1
        n_anchor = int(round(num_docs * split))
        for d in range(num_docs):
            if d < n_anchor:
                Nk = rng.integers(low=int(base_imp * ratio * 0.5),
                                   high=int(base_imp * ratio * 2.0) + 1)
                Nkp = rng.integers(low=1, high=int(base_imp * 2.5) + 1)
            else:
                Nk = rng.integers(low=int(base_imp * 0.5),
                                   high=int(base_imp * 1.5) + 1)
                Nkp = rng.integers(low=int(base_imp * 0.5),
                                   high=int(base_imp * 1.5) + 1)
            rel = rng.uniform(0.1, 0.9)
            ck = rng.binomial(Nk, true_bias[k - 1] * rel)
            ckp = rng.binomial(Nkp, true_bias[kp - 1] * rel)
            rows.append((0, f"d{k}_{d}", k, Nk, ck))
            rows.append((0, f"d{k}_{d}", kp, Nkp, ckp))
    return pd.DataFrame(rows, columns=[
        "query_id", "doc_id", "position", "impressions", "clicks"])


def run_one_trial(true_bias, ratio, split, rng, schemes=SCHEMES):
    df = generate_dataset(true_bias, NUM_DOCS, BASE_IMP, ratio, split, rng)
    out = {}
    for s in schemes:
        try:
            est = AdjacentChainEstimator(weighting=s)(
                df, query_col="query_id", doc_col="doc_id",
                imps_col="impressions", clicks_col="clicks"
            )
            vals = est.set_index("position")["examination"].values
            if len(vals) != len(true_bias):
                vals = np.full(len(true_bias), np.nan)
        except Exception:
            vals = np.full(len(true_bias), np.nan)
        out[s] = vals
    return out


# ===========================================================================
# (1) Paired bootstrap CIs
# ===========================================================================

def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
):
    """Paired bootstrap on the difference a - b (per-trial paired)."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(np.mean(a[idx] - b[idx]))
    diffs = np.sort(diffs)
    point = float(np.mean(a - b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def experiment_paired_ci(num_trials: int = 5000):
    """Re-run the headline (split=0.80, ratio in {1, 10, 50, 100}) with
    seed-aligned RNG so per-trial estimates are paired across schemes."""
    print("\n[1] Paired bootstrap CIs on per-trial MSE")
    true_bias = true_propensities()
    cells = [(0.80, 1), (0.80, 10), (0.80, 50), (0.80, 100)]
    results = []
    for split, ratio in cells:
        per_trial = {s: [] for s in SCHEMES}
        for t in tqdm(range(num_trials), desc=f"split={split} r={ratio}"):
            rng = np.random.default_rng(seed=10_000 + t)
            est = run_one_trial(true_bias, ratio, split, rng)
            for s in SCHEMES:
                # per-trial squared error averaged over positions
                err = (est[s] - true_bias) ** 2
                per_trial[s].append(np.nanmean(err))
        per_trial = {s: np.array(v) for s, v in per_trial.items()}

        h_minus_o = paired_bootstrap_ci(per_trial["harmonic"], per_trial["original"])
        h_minus_m = paired_bootstrap_ci(per_trial["harmonic"], per_trial["min"])
        for s in SCHEMES:
            arr = per_trial[s]
            results.append({
                "split": split, "ratio": ratio, "scheme": s,
                "mean_MSE": float(np.mean(arr)),
                "SE_MSE": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
            })
        results.append({
            "split": split, "ratio": ratio, "scheme": "harmonic-original (paired)",
            "delta_MSE": h_minus_o[0],
            "CI95_lo": h_minus_o[1], "CI95_hi": h_minus_o[2],
        })
        results.append({
            "split": split, "ratio": ratio, "scheme": "harmonic-min (paired)",
            "delta_MSE": h_minus_m[0],
            "CI95_lo": h_minus_m[1], "CI95_hi": h_minus_m[2],
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT / "paired_ci.csv", index=False)
    df.to_json(OUT / "paired_ci.json", orient="records", indent=2)
    print(df.to_string(index=False))
    print(f"\n  saved {OUT / 'paired_ci.csv'}\n")
    return df


# ===========================================================================
# (2) Adversarial (alpha, beta) search
# ===========================================================================

def vk_term(omega: np.ndarray, N: np.ndarray) -> float:
    """V_r(omega) = sum_i omega_i^2 / N_r^i / (sum_i omega_i)^2."""
    s = float(np.sum(omega))
    if s <= 0:
        return float("inf")
    return float(np.sum(omega**2 / N) / s**2)


def harmonic_w(Nk, Nkp):
    return Nk * Nkp / (Nk + Nkp)


def original_w(Nk, Nkp):
    return np.ones_like(Nk, dtype=float)


def min_w(Nk, Nkp):
    return np.minimum(Nk, Nkp).astype(float)


def adversarial_search(num_grid_per_axis=5, num_count_configs=200, seed=0):
    """For a grid of (alpha, beta) and a sample of count configurations,
    compare harmonic vs original on the surrogate alpha V_k + beta V_{k'}."""
    print("\n[2] Adversarial (alpha, beta) search")
    rng = np.random.default_rng(seed)
    # grid: alpha in [0.05, 50], beta in [0.05, 50] (covers extreme imbalances)
    grid = np.geomspace(0.05, 50, num_grid_per_axis)
    losses = []
    worst = {"ratio": 0.0, "config": None, "alpha": 0, "beta": 0}
    n_h_loses = 0
    n_total = 0
    for alpha in grid:
        for beta in grid:
            for cfg in range(num_count_configs):
                Nk = rng.integers(1, 1000, size=NUM_DOCS).astype(float)
                Nkp = rng.integers(1, 1000, size=NUM_DOCS).astype(float)
                wh = harmonic_w(Nk, Nkp)
                wo = original_w(Nk, Nkp)
                surrogate_h = alpha * vk_term(wh, Nk) + beta * vk_term(wh, Nkp)
                surrogate_o = alpha * vk_term(wo, Nk) + beta * vk_term(wo, Nkp)
                ratio = surrogate_h / surrogate_o
                n_total += 1
                if ratio > 1.0:
                    n_h_loses += 1
                if ratio > worst["ratio"]:
                    worst = {"ratio": float(ratio), "alpha": float(alpha),
                             "beta": float(beta),
                             "Nk_min": float(Nk.min()), "Nk_max": float(Nk.max()),
                             "Nkp_min": float(Nkp.min()), "Nkp_max": float(Nkp.max())}
                losses.append({"alpha": alpha, "beta": beta,
                               "surrogate_h": surrogate_h, "surrogate_o": surrogate_o,
                               "ratio_h_over_o": ratio})

    df = pd.DataFrame(losses)
    df.to_csv(OUT / "adversarial_search.csv", index=False)

    # Aggregate per (alpha, beta) cell
    summary = df.groupby(["alpha", "beta"]).agg(
        mean_ratio=("ratio_h_over_o", "mean"),
        max_ratio=("ratio_h_over_o", "max"),
        frac_h_loses=("ratio_h_over_o", lambda x: float((x > 1.0).mean())),
    ).reset_index()
    summary.to_csv(OUT / "adversarial_summary.csv", index=False)

    print(f"  n configurations tested: {n_total}")
    print(f"  configurations where harmonic > original (surrogate): {n_h_loses}/{n_total}")
    print(f"  worst ratio observed: {worst['ratio']:.4f} at "
          f"(alpha={worst['alpha']:.3f}, beta={worst['beta']:.3f})")
    print(f"  per-cell summary saved to {OUT / 'adversarial_summary.csv'}")

    # Now construct an *intentionally adversarial* count config: try to
    # exploit asymmetry. Set Nk small at indices where Nkp is large.
    print("\n  Constructed worst-case attempt:")
    Nk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100, 100, 100, 100, 100])
    Nkp = np.array([100.0, 100, 100, 100, 100, 1.0, 1.0, 1.0, 1.0, 1.0])
    wh = harmonic_w(Nk, Nkp); wo = original_w(Nk, Nkp)
    for alpha, beta in [(0.05, 50), (50, 0.05), (1, 1)]:
        sh = alpha * vk_term(wh, Nk) + beta * vk_term(wh, Nkp)
        so = alpha * vk_term(wo, Nk) + beta * vk_term(wo, Nkp)
        print(f"    alpha={alpha:.2f}, beta={beta:.2f}: harmonic/original = "
              f"{sh/so:.4f}  (sh={sh:.4f}, so={so:.4f})")

    Path(OUT / "adversarial_worst.json").write_text(json.dumps(worst, indent=2))
    return df, worst


# ===========================================================================
# (3) Plug-in oracle
# ===========================================================================

def plugin_oracle_weights(p_hat: np.ndarray, k: int, kp: int):
    """Build a count-only weighting function using estimated propensities."""
    pk = max(p_hat[k - 1], 1e-6)
    pkp = max(p_hat[kp - 1], 1e-6)
    alpha = (1 - pk) / pk
    beta = (1 - pkp) / pkp
    def w(Nk, Nkp):
        denom = alpha * Nkp + beta * Nk
        denom = np.where(denom <= 0, 1.0, denom)
        omega = Nk * Nkp / denom
        return omega, omega.copy()
    return w, alpha, beta


def experiment_plugin_oracle(num_trials: int = 500):
    """Two-stage plug-in: stage A fits harmonic, stage B uses plug-in
    asymmetric weights using the resulting propensity estimate. Compared
    against original, harmonic, and the true-propensity oracle."""
    print("\n[3] Plug-in oracle (two-stage adaptive)")
    true_bias = true_propensities()
    rows = []
    for split, ratio in [(0.80, 10), (0.80, 50), (0.80, 100)]:
        per_trial = {s: [] for s in ["original", "harmonic", "plugin", "true_oracle"]}
        n_complete = 0
        for t in tqdm(range(num_trials), desc=f"split={split} r={ratio}"):
            rng = np.random.default_rng(seed=20_000 + t)
            df = generate_dataset(true_bias, NUM_DOCS, BASE_IMP, ratio, split, rng)
            # Stage A: original (baseline)
            try:
                p_orig = (AdjacentChainEstimator(weighting="original")(
                    df, query_col="query_id", doc_col="doc_id",
                    imps_col="impressions", clicks_col="clicks"
                ).set_index("position")["examination"].values)
                # Stage A': harmonic preliminary
                p_hat = (AdjacentChainEstimator(weighting="harmonic")(
                    df, query_col="query_id", doc_col="doc_id",
                    imps_col="impressions", clicks_col="clicks"
                ).set_index("position")["examination"].values)
                if len(p_hat) != len(true_bias) or len(p_orig) != len(true_bias):
                    continue
                # Stage B: plug-in weights from p_hat
                p_plugin = (AdjacentChainEstimator(
                    weighting=_plugin_weight_factory(p_hat)
                )(df, query_col="query_id", doc_col="doc_id",
                  imps_col="impressions", clicks_col="clicks"
                ).set_index("position")["examination"].values)
                # Stage B': true-propensity oracle (cheats; uses true p)
                p_true = (AdjacentChainEstimator(
                    weighting=_plugin_weight_factory(true_bias)
                )(df, query_col="query_id", doc_col="doc_id",
                  imps_col="impressions", clicks_col="clicks"
                ).set_index("position")["examination"].values)
                if len(p_plugin) != len(true_bias) or len(p_true) != len(true_bias):
                    continue
            except Exception as e:
                continue
            per_trial["original"].append(np.mean((p_orig - true_bias) ** 2))
            per_trial["harmonic"].append(np.mean((p_hat - true_bias) ** 2))
            per_trial["plugin"].append(np.mean((p_plugin - true_bias) ** 2))
            per_trial["true_oracle"].append(np.mean((p_true - true_bias) ** 2))
            n_complete += 1
        results = per_trial

        for name, arr in results.items():
            arr = np.array(arr)
            rows.append({
                "split": split, "ratio": ratio, "scheme": name,
                "mean_MSE": float(np.mean(arr)),
                "SE_MSE": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                "n_trials_complete": len(arr),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "plugin_oracle.csv", index=False)
    print(df.to_string(index=False))
    print(f"  saved {OUT / 'plugin_oracle.csv'}\n")
    return df


def _plugin_weight_factory(p_hat):
    """Returns a callable plug-in weighting using a single global (alpha,
    beta) derived from p_hat (averaged across position pairs).
    Pair-specific (alpha, beta) would require knowing which (k, k') we're
    evaluating; here we use the average for simplicity."""
    p_arr = np.asarray(p_hat, dtype=float)
    p_arr = np.clip(p_arr, 1e-6, 1.0)
    alpha_arr = (1 - p_arr) / p_arr
    alpha_avg = float(np.mean(alpha_arr[:-1]))
    beta_avg = float(np.mean(alpha_arr[1:]))

    def w(Nk, Nkp):
        Nk = np.asarray(Nk, dtype=float)
        Nkp = np.asarray(Nkp, dtype=float)
        denom = alpha_avg * Nkp + beta_avg * Nk
        denom = np.where(denom <= 0, 1.0, denom)
        omega = Nk * Nkp / denom
        return omega, omega.copy()
    return w


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use 100 trials instead of 500 for fast iteration")
    ap.add_argument("--skip-paired", action="store_true")
    ap.add_argument("--skip-adversarial", action="store_true")
    ap.add_argument("--skip-plugin", action="store_true")
    args = ap.parse_args()
    nt = 100 if args.quick else 500

    if not args.skip_paired:
        experiment_paired_ci(num_trials=nt)
    if not args.skip_adversarial:
        adversarial_search()
    if not args.skip_plugin:
        experiment_plugin_oracle(num_trials=nt)

    print("\n--- DONE ---")
