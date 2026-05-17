"""Diagnose why harmonic helps on synthetic but hurts click-noise on real KDD.

Three angles:
  (A) (N, theta) correlation — is high-impression also high-click-rate?
  (B) Position bias curve — does the estimator give sensible p_k?
  (C) Compare AdjacentChain vs AllPairs vs Pivot estimators.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
SUBSET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012/train")

SCHEMES = ["uniform", "min", "harmonic"]


def weights(scheme, Nk, Nkp):
    if scheme == "uniform":
        return np.ones_like(Nk, dtype=float)
    if scheme == "min":
        return np.minimum(Nk, Nkp).astype(float)
    if scheme == "harmonic":
        return Nk * Nkp / (Nk + Nkp)


def adjacent_pair(agg, k, kp, keys):
    a = agg[agg["position"] == k][keys + ["impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][keys + ["impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=keys, how="inner")


def ratio(omega, Nk, Ck, Nkp, Ckp):
    A = float(np.sum(omega * Ck / Nk))
    B = float(np.sum(omega * Ckp / Nkp))
    return A / B if B > 0 else float("nan")


def diag_n_theta_correlation(m: pd.DataFrame, label: str):
    """Per-cell N vs theta correlation, separately for each position."""
    print(f"\n--- (A) (N, theta) correlation: {label} ---")
    for col_N, col_C, pos_label in [("Nk", "Ck", "k"), ("Nkp", "Ckp", "k+1")]:
        N = m[col_N].to_numpy(dtype=float)
        theta = m[col_C].to_numpy(dtype=float) / N
        # only cells with at least one click contribute meaningful theta
        mask = N >= 5  # focus on cells we can estimate theta from
        Nm = N[mask]
        thetam = theta[mask]
        if len(Nm) < 10:
            continue
        # Spearman rho: rank correlation
        rN = pd.Series(Nm).rank().to_numpy()
        rT = pd.Series(thetam).rank().to_numpy()
        rho = np.corrcoef(rN, rT)[0, 1]
        # also: compare theta in low-N vs high-N quartiles
        q1 = np.percentile(Nm, 25)
        q4 = np.percentile(Nm, 75)
        theta_low = thetam[Nm <= q1].mean()
        theta_high = thetam[Nm >= q4].mean()
        print(f"  position {pos_label}: cells with N>=5 = {len(Nm):,}")
        print(f"    Spearman rank corr(N, theta) = {rho:+.3f}")
        print(f"    mean theta in N-quartile 1: {theta_low:.4f}")
        print(f"    mean theta in N-quartile 4: {theta_high:.4f}")
        print(f"    quartile 4 / quartile 1: {theta_high/theta_low:.2f}x"
              if theta_low > 0 else "")


def diag_position_curve(name, agg, keys, scheme: str = "harmonic"):
    """Compute p_k via AdjacentChain for each k. Normalize p_1 = 1."""
    positions = sorted(agg["position"].unique().tolist())
    p = {positions[0]: 1.0}
    R_chain = {}
    for k, kp in zip(positions[:-1], positions[1:]):
        m = adjacent_pair(agg, k, kp, keys)
        if m.empty:
            continue
        Nk = m["Nk"].to_numpy(dtype=float)
        Nkp = m["Nkp"].to_numpy(dtype=float)
        Ck = m["Ck"].to_numpy(dtype=float)
        Ckp = m["Ckp"].to_numpy(dtype=float)
        omega = weights(scheme, Nk, Nkp)
        R = ratio(omega, Nk, Ck, Nkp, Ckp)
        R_chain[(k, kp)] = R
        p[kp] = p[k] / R  # p_{k+1} = p_k / R since R = p_k / p_{k+1}
    return positions, p, R_chain


def diag_allpairs_estimator(agg, keys, scheme: str = "harmonic"):
    """Compute p_k by computing R_{1,k} directly for every k>=2 (Pivot @ 1)."""
    positions = sorted(agg["position"].unique().tolist())
    p = {positions[0]: 1.0}
    for kp in positions[1:]:
        m = adjacent_pair(agg, positions[0], kp, keys)
        if m.empty:
            continue
        Nk = m["Nk"].to_numpy(dtype=float)
        Nkp = m["Nkp"].to_numpy(dtype=float)
        Ck = m["Ck"].to_numpy(dtype=float)
        Ckp = m["Ckp"].to_numpy(dtype=float)
        omega = weights(scheme, Nk, Nkp)
        R = ratio(omega, Nk, Ck, Nkp, Ckp)
        p[kp] = 1.0 / R
    return p


def main():
    print("=" * 60)
    print("Full KDD Cup 2012")
    print("=" * 60)
    full = pd.read_parquet(KDD)
    print(f"  cells: {len(full):,}; impressions: {full['impressions'].sum():,}; "
          f"CTR: {full['clicks'].sum()/full['impressions'].sum():.4%}")

    # (A) per-pair (N, theta) correlation
    for k, kp in [(1, 2), (2, 3)]:
        m = adjacent_pair(full, k, kp, ["queryID", "adID"])
        diag_n_theta_correlation(m, f"FULL pair ({k},{kp})")

    # (B) position curve
    print("\n--- (B) Position bias curve (full KDD) ---")
    for scheme in SCHEMES:
        positions, p, R_chain = diag_position_curve("kdd_full", full,
                                                   ["queryID", "adID"], scheme)
        print(f"  AdjacentChain ({scheme}): " +
              ", ".join([f"p_{k}={p[k]:.3f}" for k in positions]) +
              "   |  R_chain = " +
              ", ".join([f"({k},{kp})={R:.3f}" for (k, kp), R in R_chain.items()]))
    for scheme in SCHEMES:
        p = diag_allpairs_estimator(full, ["queryID", "adID"], scheme)
        print(f"  Pivot @ 1   ({scheme}): " +
              ", ".join([f"p_{k}={p[k]:.3f}" for k in sorted(p.keys())]))

    # save & plot
    plt.figure(figsize=(7, 4))
    for scheme in SCHEMES:
        positions, p, _ = diag_position_curve("kdd_full", full,
                                             ["queryID", "adID"], scheme)
        plt.plot(positions, [p[k] for k in positions], "o-", label=f"AdjChain {scheme}")
    for scheme in SCHEMES:
        p = diag_allpairs_estimator(full, ["queryID", "adID"], scheme)
        plt.plot(sorted(p.keys()), [p[k] for k in sorted(p.keys())], "x--",
                 alpha=0.6, label=f"Pivot@1 {scheme}")
    plt.xlabel("position")
    plt.ylabel(r"$\hat{p}_k$ (normalized to $p_1=1$)")
    plt.title("KDD Cup 2012 Track 2 (full): position-bias estimates by scheme & estimator")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = OUT / "kdd2012_full_position_curve.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\n  saved {out_png}")

    # --- subset comparison ---
    print("\n" + "=" * 60)
    print("KDD Cup 2012 SCALED SUBSET (1M rows, no Impression column)")
    print("=" * 60)
    sub = pd.read_csv(SUBSET, sep="\t", header=None,
                      names=["click", "displayURL", "adID", "advertiserID",
                             "depth", "position", "queryID", "keywordID",
                             "titleID", "descriptionID", "userID"])
    sub_agg = (sub.groupby(["queryID", "adID", "position"], sort=False)
                  .agg(impressions=("click", "size"), clicks=("click", "sum"))
                  .reset_index())
    print(f"  cells: {len(sub_agg):,}; rows: {len(sub):,}; "
          f"CTR: {sub['click'].sum()/len(sub):.4%}")

    for k, kp in [(1, 2), (2, 3)]:
        m = adjacent_pair(sub_agg, k, kp, ["queryID", "adID"])
        if not m.empty:
            diag_n_theta_correlation(m, f"SUBSET pair ({k},{kp})")

    print("\n--- Position curve on SUBSET ---")
    for scheme in SCHEMES:
        positions, p, R_chain = diag_position_curve("kdd_subset", sub_agg,
                                                   ["queryID", "adID"], scheme)
        print(f"  AdjacentChain ({scheme}): " +
              ", ".join([f"p_{k}={p[k]:.3f}" for k in positions]))


if __name__ == "__main__":
    main()
