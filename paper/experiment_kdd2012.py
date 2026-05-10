"""
KDD Cup 2012 Track 2 (Tencent SOSO sponsored search) -- real-log experiment.

Source: SamueChan/CTR-Predition GitHub mirror of the scaled-down KDD Cup
2012 Track 2 training file (1M rows, ~64 MB). Each row is one impression
with columns:
  click, displayURL, adID, advertiserID, depth, position,
  queryID, keywordID, titleID, descriptionID, userID

We aggregate to (queryID, adID, position) -> (impressions, clicks).

What we test:
  (1) Imbalance distribution. Real impression-ratio distribution
      N_k / N_{k+1} on adjacent-position pairs of the same (q, ad).
  (2) Variance surrogate per scheme. For each adjacent (k, k+1) we
      compute V_k(omega) and V_{k+1}(omega) for omega in {1, min, harmonic}
      and report alpha V_k + beta V_{k+1} for the realistic
      alpha/beta values (under PBM with eta=1, p_k = 1/k).
  (3) Empirical bootstrap variance of $\\hat{R}_{k,k+1}$. We resample with
      replacement at the (q, ad) level and compute $\\hat{R}_{k,k+1}$ for
      each scheme; the spread of bootstrap estimates is the empirical
      variance (no ground truth needed).
  (4) Mean $\\hat{R}_{k,k+1}$ on the full data. Schemes should agree
      asymptotically (all ratio-unbiased under PBM); variance differs.

Outputs: paper/reviewer_response/kdd2012_*.{csv,json,png}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

OUT = Path(__file__).resolve().parent / "reviewer_response"
OUT.mkdir(exist_ok=True)
DATA = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012/train")

COLS = ["click", "displayURL", "adID", "advertiserID", "depth", "position",
        "queryID", "keywordID", "titleID", "descriptionID", "userID"]


def load_aggregate() -> pd.DataFrame:
    print(f"Loading {DATA} ...")
    df = pd.read_csv(DATA, sep="\t", header=None, names=COLS,
                     usecols=["click", "queryID", "adID", "position"])
    print(f"  {len(df):,} impressions")
    agg = (df.groupby(["queryID", "adID", "position"], sort=False)
             .agg(impressions=("click", "size"), clicks=("click", "sum"))
             .reset_index())
    print(f"  {len(agg):,} (q, a, position) cells")
    return agg


def adjacent_pair_table(agg: pd.DataFrame, k: int, kp: int) -> pd.DataFrame:
    """For each (q, a) appearing at both positions k and kp, return
    a row with (Nk, Ck, Nkp, Ckp)."""
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]]
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]]
    a = a.rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = b.rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    m = a.merge(b, on=["queryID", "adID"], how="inner")
    return m


def weight_schemes(Nk, Nkp):
    return {
        "uniform":  np.ones_like(Nk, dtype=float),
        "min":      np.minimum(Nk, Nkp).astype(float),
        "harmonic": Nk * Nkp / (Nk + Nkp),
    }


def vk_term(omega, N):
    s = float(np.sum(omega))
    if s <= 0:
        return float("inf")
    return float(np.sum(omega ** 2 / N) / s ** 2)


def ratio_estimator(omega, Nk, Ck, Nkp, Ckp):
    A = float(np.sum(omega * Ck / Nk))
    B = float(np.sum(omega * Ckp / Nkp))
    return A / B if B > 0 else float("nan")


def bootstrap_std(omega_fn, m: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    """Bootstrap std of $\\hat{R}_{k,k+1}$ resampling (q, ad) rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(m)
    Nk = m["Nk"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        nk, ck, nkp, ckp = Nk[idx], Ck[idx], Nkp[idx], Ckp[idx]
        omega = omega_fn(nk, nkp)
        estimates.append(ratio_estimator(omega, nk, ck, nkp, ckp))
    e = np.asarray(estimates, dtype=float)
    e = e[np.isfinite(e)]
    return {"mean": float(np.mean(e)),
            "std":  float(np.std(e, ddof=1)),
            "lo":   float(np.percentile(e, 2.5)),
            "hi":   float(np.percentile(e, 97.5))}


def main():
    agg = load_aggregate()

    # ---- (1) Imbalance distribution ----
    print("\n[1] Imbalance distribution N_k/N_{k+1} on adjacent pairs")
    rows = []
    pairs = [(1, 2), (2, 3)]
    pair_tables = {p: adjacent_pair_table(agg, p[0], p[1]) for p in pairs}
    for (k, kp), m in pair_tables.items():
        if m.empty:
            continue
        ratio = m["Nk"].to_numpy() / m["Nkp"].to_numpy()
        bal = float(((ratio >= 0.5) & (ratio <= 2.0)).mean())
        gt10 = float((ratio > 10).mean())
        gt100 = float((ratio > 100).mean())
        rows.append({"k": k, "kp": kp, "n_pairs": int(len(m)),
                     "frac_in_[0.5,2]": bal,
                     "frac_>10": gt10, "frac_>100": gt100,
                     "median_ratio": float(np.median(ratio)),
                     "p90_ratio": float(np.percentile(ratio, 90)),
                     "p99_ratio": float(np.percentile(ratio, 99)),
                     "max_ratio": float(np.max(ratio))})
        print(f"  ({k},{kp}): n={len(m):,} pairs, "
              f"frac in [0.5,2]={bal:.3f}, frac>10={gt10:.3f}, "
              f"median ratio={np.median(ratio):.2f}, p90={np.percentile(ratio,90):.2f}")
    pd.DataFrame(rows).to_csv(OUT / "kdd2012_imbalance.csv", index=False)

    # plot
    fig, axes = plt.subplots(1, len(pair_tables), figsize=(10, 4))
    if len(pair_tables) == 1:
        axes = [axes]
    for ax, ((k, kp), m) in zip(axes, pair_tables.items()):
        if m.empty:
            continue
        r = m["Nk"].to_numpy() / m["Nkp"].to_numpy()
        ax.hist(np.log10(r), bins=40, color="steelblue", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\log_{10}(N_k / N_{k+1})$")
        ax.set_ylabel("# (q, ad) pairs")
        ax.set_title(f"KDD Cup 2012 Track 2: positions {k} vs {kp}\n(n={len(m):,})")
    plt.tight_layout()
    plt.savefig(OUT / "kdd2012_imbalance.png", dpi=150)
    plt.close()

    # ---- (2) Variance surrogate per scheme ----
    print("\n[2] Variance surrogate alpha V_k + beta V_{k+1} per scheme")
    surrogate_rows = []
    # PBM with eta=1, p_k=1/k -> alpha_k = k - 1.
    for (k, kp), m in pair_tables.items():
        if m.empty or len(m) < 2:
            continue
        alpha = k - 1 if k > 1 else 0.5  # avoid alpha=0 trivializing for k=1
        beta = kp - 1
        Nk = m["Nk"].to_numpy(dtype=float)
        Nkp = m["Nkp"].to_numpy(dtype=float)
        for name, omega in weight_schemes(Nk, Nkp).items():
            Vk = vk_term(omega, Nk)
            Vkp = vk_term(omega, Nkp)
            surrogate_rows.append({
                "k": k, "kp": kp, "alpha": alpha, "beta": beta,
                "scheme": name, "n_pairs": int(len(m)),
                "Vk": Vk, "Vkp": Vkp,
                "symmetric_Vk_plus_Vkp": Vk + Vkp,
                "asymmetric_aVk_plus_bVkp": alpha * Vk + beta * Vkp,
            })
    sdf = pd.DataFrame(surrogate_rows)
    sdf.to_csv(OUT / "kdd2012_surrogate.csv", index=False)
    print(sdf.to_string(index=False))

    # Reduction vs uniform (as % reduction)
    pivot_sym = sdf.pivot_table(index=["k", "kp"], columns="scheme",
                                 values="symmetric_Vk_plus_Vkp")
    pivot_asym = sdf.pivot_table(index=["k", "kp"], columns="scheme",
                                  values="asymmetric_aVk_plus_bVkp")
    print("\n  Symmetric surrogate reduction vs uniform:")
    print(((pivot_sym["uniform"].values[:, None] - pivot_sym[["min", "harmonic"]].values)
           / pivot_sym["uniform"].values[:, None] * 100))
    print("\n  Asymmetric surrogate reduction vs uniform:")
    print(((pivot_asym["uniform"].values[:, None] - pivot_asym[["min", "harmonic"]].values)
           / pivot_asym["uniform"].values[:, None] * 100))

    # ---- (3) Bootstrap variance of $\hat{R}$ ----
    print("\n[3] Bootstrap variance of R_{k,k+1} per scheme (1000 resamples)")
    boot_rows = []
    for (k, kp), m in pair_tables.items():
        if m.empty or len(m) < 30:
            continue
        for name, fn in [("uniform", lambda nk, nkp: np.ones_like(nk, dtype=float)),
                          ("min",     lambda nk, nkp: np.minimum(nk, nkp).astype(float)),
                          ("harmonic", lambda nk, nkp: nk * nkp / (nk + nkp))]:
            res = bootstrap_std(fn, m, n_boot=1000, seed=k * 100 + kp)
            boot_rows.append({"k": k, "kp": kp, "scheme": name,
                              "n_pairs": int(len(m)),
                              **res})
            print(f"  ({k},{kp}) {name:8s}  R_hat={res['mean']:.4f}  "
                  f"std={res['std']:.4f}  CI95=[{res['lo']:.4f}, {res['hi']:.4f}]")
    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(OUT / "kdd2012_bootstrap.csv", index=False)

    # bootstrap variance reduction summary
    pv = bdf.pivot_table(index=["k", "kp"], columns="scheme", values="std")
    print("\n  Bootstrap-std reduction (%) vs uniform:")
    for col in ["min", "harmonic"]:
        red = (pv["uniform"] - pv[col]) / pv["uniform"] * 100
        print(f"    {col}: {red.to_dict()}")

    # ---- summary JSON ----
    summary = {
        "n_impressions": int(agg["impressions"].sum()),
        "n_qad_pos_cells": int(len(agg)),
        "n_adjacent_pairs": {f"{k}-{kp}": int(len(m)) for (k, kp), m in pair_tables.items()},
        "imbalance": rows,
        "surrogate": surrogate_rows,
        "bootstrap": boot_rows,
    }
    (OUT / "kdd2012_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved outputs under {OUT}/kdd2012_*")


if __name__ == "__main__":
    main()
