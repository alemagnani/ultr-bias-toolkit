"""KDD Cup 2012 Track 2 -- FULL dataset (235M impressions, 63.8M cells).

Reads pre-aggregated parquet (built by aggregate_kdd2012_full.py), then runs
the same analysis as experiment_kdd2012.py:
  (1) impression-imbalance distribution N_k/N_{k+1}
  (2) variance surrogate per scheme
  (3) bootstrap empirical std of $\\hat{R}_{k,k+1}$
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT  = Path(__file__).resolve().parent / "reviewer_response"
OUT.mkdir(exist_ok=True)
DATA = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")


def adjacent_pair_table(agg: pd.DataFrame, k: int, kp: int) -> pd.DataFrame:
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def vk_term(omega, N):
    s = float(np.sum(omega))
    return float(np.sum(omega**2 / N) / s**2) if s > 0 else float("inf")


def ratio_estimator(omega, Nk, Ck, Nkp, Ckp):
    A = float(np.sum(omega * Ck / Nk))
    B = float(np.sum(omega * Ckp / Nkp))
    return A / B if B > 0 else float("nan")


def bootstrap_std(name, m: pd.DataFrame, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(m)
    Nk  = m["Nk"].to_numpy(dtype=float)
    Ck  = m["Ck"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    fns = {"uniform":  lambda nk, nkp: np.ones_like(nk, dtype=float),
           "min":      lambda nk, nkp: np.minimum(nk, nkp).astype(float),
           "harmonic": lambda nk, nkp: nk * nkp / (nk + nkp)}
    fn = fns[name]
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        nk, ck, nkp, ckp = Nk[idx], Ck[idx], Nkp[idx], Ckp[idx]
        omega = fn(nk, nkp)
        estimates[b] = ratio_estimator(omega, nk, ck, nkp, ckp)
    e = estimates[np.isfinite(estimates)]
    return {"mean": float(np.mean(e)),
            "std":  float(np.std(e, ddof=1)),
            "lo":   float(np.percentile(e, 2.5)),
            "hi":   float(np.percentile(e, 97.5))}


def main():
    print(f"Loading {DATA} ...")
    agg = pd.read_parquet(DATA)
    print(f"  {len(agg):,} cells; total impressions {agg['impressions'].sum():,}; "
          f"clicks {agg['clicks'].sum():,}; CTR {agg['clicks'].sum()/agg['impressions'].sum():.4%}")
    print(f"  positions present: {sorted(agg['position'].unique().tolist())}")

    pairs = [(1, 2), (2, 3)]
    pair_tables = {p: adjacent_pair_table(agg, p[0], p[1]) for p in pairs}

    # ---- (1) imbalance ----
    print("\n[1] Imbalance distribution N_k / N_{k+1}")
    rows = []
    for (k, kp), m in pair_tables.items():
        if m.empty:
            continue
        ratio = m["Nk"].to_numpy(dtype=float) / m["Nkp"].to_numpy(dtype=float)
        rows.append({"k": k, "kp": kp, "n_pairs": int(len(m)),
                      "frac_in_[0.5,2]": float(((ratio >= 0.5) & (ratio <= 2.0)).mean()),
                      "frac_>10":  float((ratio > 10).mean()),
                      "frac_>100": float((ratio > 100).mean()),
                      "median_ratio": float(np.median(ratio)),
                      "p90_ratio":    float(np.percentile(ratio, 90)),
                      "p99_ratio":    float(np.percentile(ratio, 99)),
                      "max_ratio":    float(np.max(ratio))})
        print(f"  ({k},{kp}): n={len(m):,}, "
              f"in[0.5,2]={rows[-1]['frac_in_[0.5,2]']:.3f}, >10={rows[-1]['frac_>10']:.3f}, "
              f"median={rows[-1]['median_ratio']:.2f}, p90={rows[-1]['p90_ratio']:.2f}, "
              f"p99={rows[-1]['p99_ratio']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "kdd2012_full_imbalance.csv", index=False)

    fig, axes = plt.subplots(1, len(pair_tables), figsize=(10, 4))
    if len(pair_tables) == 1:
        axes = [axes]
    for ax, ((k, kp), m) in zip(axes, pair_tables.items()):
        if m.empty:
            continue
        r = m["Nk"].to_numpy(dtype=float) / m["Nkp"].to_numpy(dtype=float)
        ax.hist(np.log10(r), bins=60, color="steelblue", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\log_{10}(N_k / N_{k+1})$")
        ax.set_ylabel("# (q, ad) pairs")
        ax.set_title(f"KDD Cup 2012 Track 2 (full): positions {k} vs {kp}\n(n={len(m):,})")
    plt.tight_layout()
    plt.savefig(OUT / "kdd2012_full_imbalance.png", dpi=150)
    plt.close()

    # ---- (2) variance surrogate ----
    print("\n[2] Variance surrogate per scheme")
    surrogate_rows = []
    for (k, kp), m in pair_tables.items():
        if m.empty or len(m) < 2:
            continue
        alpha = k - 1 if k > 1 else 0.5
        beta = kp - 1
        Nk = m["Nk"].to_numpy(dtype=float)
        Nkp = m["Nkp"].to_numpy(dtype=float)
        for name, omega in [
            ("uniform",  np.ones_like(Nk, dtype=float)),
            ("min",      np.minimum(Nk, Nkp).astype(float)),
            ("harmonic", Nk * Nkp / (Nk + Nkp)),
        ]:
            Vk = vk_term(omega, Nk)
            Vkp = vk_term(omega, Nkp)
            surrogate_rows.append({
                "k": k, "kp": kp, "alpha": alpha, "beta": beta, "scheme": name,
                "n_pairs": int(len(m)), "Vk": Vk, "Vkp": Vkp,
                "symmetric_Vk_plus_Vkp": Vk + Vkp,
                "asymmetric_aVk_plus_bVkp": alpha * Vk + beta * Vkp,
            })
    sdf = pd.DataFrame(surrogate_rows)
    sdf.to_csv(OUT / "kdd2012_full_surrogate.csv", index=False)
    print(sdf.to_string(index=False))

    pivot_sym  = sdf.pivot_table(index=["k","kp"], columns="scheme", values="symmetric_Vk_plus_Vkp")
    pivot_asym = sdf.pivot_table(index=["k","kp"], columns="scheme", values="asymmetric_aVk_plus_bVkp")
    print("\n  Symmetric  reduction vs uniform (%):")
    red_sym  = (pivot_sym["uniform"].values[:, None] - pivot_sym[["min","harmonic"]].values) / pivot_sym["uniform"].values[:, None] * 100
    print(red_sym)
    print("  Asymmetric reduction vs uniform (%):")
    red_asym = (pivot_asym["uniform"].values[:, None] - pivot_asym[["min","harmonic"]].values) / pivot_asym["uniform"].values[:, None] * 100
    print(red_asym)

    # ---- (3) bootstrap variance ----
    print("\n[3] Bootstrap std of R_hat per scheme (1000 resamples)")
    boot_rows = []
    for (k, kp), m in pair_tables.items():
        if m.empty or len(m) < 30:
            continue
        for name in ["uniform", "min", "harmonic"]:
            res = bootstrap_std(name, m, n_boot=1000, seed=k * 100 + kp)
            boot_rows.append({"k": k, "kp": kp, "scheme": name, "n_pairs": int(len(m)), **res})
            print(f"  ({k},{kp}) {name:8s} R_hat={res['mean']:.4f}  std={res['std']:.5f}  CI95=[{res['lo']:.4f}, {res['hi']:.4f}]")
    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(OUT / "kdd2012_full_bootstrap.csv", index=False)
    pv = bdf.pivot_table(index=["k","kp"], columns="scheme", values="std")
    print("\n  Bootstrap-std reduction (%) vs uniform:")
    for col in ["min", "harmonic"]:
        red = (pv["uniform"] - pv[col]) / pv["uniform"] * 100
        print(f"    {col}: {red.to_dict()}")

    summary = {
        "n_cells":        int(len(agg)),
        "n_impressions":  int(agg["impressions"].sum()),
        "n_clicks":       int(agg["clicks"].sum()),
        "ctr":            float(agg["clicks"].sum() / agg["impressions"].sum()),
        "n_adjacent_pairs": {f"{k}-{kp}": int(len(m)) for (k, kp), m in pair_tables.items()},
        "imbalance":      rows,
        "surrogate":      surrogate_rows,
        "bootstrap":      boot_rows,
    }
    (OUT / "kdd2012_full_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved outputs under {OUT}/kdd2012_full_*")


if __name__ == "__main__":
    main()
