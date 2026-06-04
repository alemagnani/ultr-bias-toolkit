"""Regularised theta-aware oracle on full KDD pair (1,2).

The pathology of the theta-aware oracle (ARE = 0.002) comes from per-cell
plug-in noise: theta_hat = C/N is wildly off when N is small. Shrinkage
toward the marginal CTR controls this.

Shrinkage estimator:
    theta_hat_k^i = (C_k^i + a * theta_bar_k) / (N_k^i + a)
where theta_bar_k = sum_i C_k^i / sum_i N_k^i is the marginal position-k CTR
and a is a pseudo-count "shrinkage strength" (a = 0 -> raw; a -> inf ->
marginal).

We sweep a in {0, 1, 10, 100, 1000} and report ARE vs uniform.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
OUT = Path("/home/alessandro/workspace/ultr-bias-toolkit/paper/reviewer_response")
PAIR = (1, 2)


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def shrunk(C, N, marg, a):
    return (C + a * marg) / (N + a)


def vtilde(omega, Nk, Nkp, th_k, th_kp):
    A = float(np.sum(omega * th_k)); B = float(np.sum(omega * th_kp))
    vA = float(np.sum(omega**2 * th_k * (1 - th_k) / Nk))
    vB = float(np.sum(omega**2 * th_kp * (1 - th_kp) / Nkp))
    R = A / B
    V = vA / A**2 + vB / B**2
    return R, V, abs(R) * np.sqrt(V), (omega.sum())**2 / (omega**2).sum()


def main():
    agg = pd.read_parquet(KDD)
    m = build_pair(agg, *PAIR)
    Nk = m["Nk"].to_numpy(float); Nkp = m["Nkp"].to_numpy(float)
    Ck = m["Ck"].to_numpy(float); Ckp = m["Ckp"].to_numpy(float)
    bar_k = float(Ck.sum() / Nk.sum()); bar_kp = float(Ckp.sum() / Nkp.sum())
    print(f"pairs={len(m):,}, bar_theta_1={bar_k:.4f}, bar_theta_2={bar_kp:.4f}", flush=True)

    # Uniform baseline (always uses shrunk theta with a=1 -- consistent with paper's plug-in)
    th_k_base = shrunk(Ck, Nk, bar_k, 1.0)
    th_kp_base = shrunk(Ckp, Nkp, bar_kp, 1.0)
    _, V_u, _, _ = vtilde(np.ones_like(Nk), Nk, Nkp, th_k_base, th_kp_base)

    rows = []
    for a in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        th_k = shrunk(Ck, Nk, bar_k, a) if a > 0 else (Ck / np.maximum(Nk, 1.0))
        th_kp = shrunk(Ckp, Nkp, bar_kp, a) if a > 0 else (Ckp / np.maximum(Nkp, 1.0))
        # avoid 0/0 for a=0
        if a == 0:
            th_k = np.where(Nk > 0, Ck / np.maximum(Nk, 1.0), bar_k)
            th_kp = np.where(Nkp > 0, Ckp / np.maximum(Nkp, 1.0), bar_kp)
        # theta-aware oracle weight: omega ∝ 1/[α θ(1-θ)/N + β θ'(1-θ')/N']
        # use a/b = 0.5, 1.0 as in the existing paper
        alpha, beta = 0.5, 1.0
        s = alpha * th_k * (1 - th_k) / Nk + beta * th_kp * (1 - th_kp) / Nkp
        # cap denominator to avoid div by zero (rare since theta>0 after shrinkage)
        omega = 1.0 / np.maximum(s, 1e-30)
        R, V, SE, ess = vtilde(omega, Nk, Nkp, th_k, th_kp)
        are = V_u / V if V > 0 else float("inf")
        rows.append({"shrinkage_a": a, "R_hat": R, "SE": SE, "V": V,
                     "ARE_vs_uniform": are, "ESS": ess})
        print(f"a={a:7.1f}: R={R:.4f}  SE={SE:.3e}  ARE={are:6.2f}  ESS={ess:.2e}", flush=True)

    # Reference values from paper for context
    print(f"\nReference (from paper):")
    print(f"  uniform        : R=1.6834, SE=4.24e-3, ARE=1.00")
    print(f"  harmonic       : R=1.7095, SE=2.02e-3, ARE=4.53")
    print(f"  theta-oracle*  : R=2.3435, SE=0.148,   ARE=0.002   (* unregularised; same as a=1 here)")

    pd.DataFrame(rows).to_csv(OUT / "kdd_regularised_theta_oracle.csv", index=False)
    print(f"\nWrote {OUT / 'kdd_regularised_theta_oracle.csv'}")


if __name__ == "__main__":
    main()
