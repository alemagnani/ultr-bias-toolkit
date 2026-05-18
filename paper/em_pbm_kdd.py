"""PBM EM (Wang et al. 2018-style) for pair (1,2) on full KDD Cup 2012.

We jointly estimate per-cell relevance r_i and global propensity p_2
(with p_1 = 1) by EM.

Click model under PBM:
    P(C_k^i = 1 | impression) = p_k * r_i

Small-rate ML update for r_i given p_2:
    r_i = (C_1^i + C_2^i) / (N_1^i + p_2 N_2^i)
Update for p_2 given r_i:
    p_2 = sum_i C_2^i / sum_i (r_i * N_2^i)

Iterate to convergence; report R_hat = p_1 / p_2 = 1/p_2.
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


def em_pbm(Nk, Nkp, Ck, Ckp, max_iter=200, tol=1e-9):
    """EM with p_1 = 1. Returns p_2 estimate and trace."""
    p2 = 0.5  # init
    trace = []
    for it in range(max_iter):
        # E-step (small-rate): r_i = (C_1+C_2) / (N_1 + p_2 N_2)
        denom = Nk + p2 * Nkp
        r = (Ck + Ckp) / np.maximum(denom, 1.0)
        # M-step: p_2 = sum(C_2) / sum(r * N_2)
        num = float(np.sum(Ckp))
        den = float(np.sum(r * Nkp))
        new_p2 = num / max(den, 1e-30)
        trace.append((it, p2, new_p2))
        if abs(new_p2 - p2) < tol:
            p2 = new_p2
            break
        p2 = new_p2
    return p2, trace


def main():
    print(f"Loading {KDD}", flush=True)
    agg = pd.read_parquet(KDD)
    m = build_pair(agg, *PAIR)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    print(f"pairs={len(m):,}, total imps={int(Nk.sum() + Nkp.sum()):,}", flush=True)

    p2, trace = em_pbm(Nk, Nkp, Ck, Ckp)
    R_em = 1.0 / p2
    print(f"\nEM converged: p_1=1.0, p_2={p2:.6f}, R_hat_EM = 1/p_2 = {R_em:.4f}")
    print(f"iterations: {len(trace)}, last delta = {abs(trace[-1][2]-trace[-1][1]):.2e}")

    # Comparison with count-only schemes (from prior runs)
    print("\n--- Comparison of R_hat across schemes (full KDD pair (1,2)) ---")
    print(f"{'Scheme':<24s} {'R_hat':>8s}")
    print(f"{'-'*34}")
    print(f"{'uniform':<24s} {1.6834:>8.4f}")
    print(f"{'min':<24s} {1.7122:>8.4f}")
    print(f"{'harmonic':<24s} {1.7095:>8.4f}")
    print(f"{'count-only oracle (.5,1)':<24s} {1.7147:>8.4f}")
    print(f"{'theta-aware oracle':<24s} {2.3435:>8.4f}")
    print(f"{'PBM-EM (this run)':<24s} {R_em:>8.4f}")

    pd.DataFrame({
        "scheme": ["uniform", "min", "harmonic", "count_oracle", "theta_oracle", "pbm_em"],
        "R_hat": [1.6834, 1.7122, 1.7095, 1.7147, 2.3435, R_em],
    }).to_csv(OUT / "kdd_em_comparison.csv", index=False)
    print(f"\nWrote {OUT / 'kdd_em_comparison.csv'}")


if __name__ == "__main__":
    main()
