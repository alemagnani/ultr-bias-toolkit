"""Plug-in delta-method ARE on KDD pairs (1,2), (2,3), (3,4), (4,5) for
uniform / min / harmonic.

Same machinery as paper/plugin_variance_kdd2012.py but applied to multiple
adjacent pairs, to broaden the real-log evidence beyond the (1,2) headline.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
OUT = Path("/home/alessandro/workspace/ultr-bias-toolkit/paper/reviewer_response")
PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5)]
SHRINK = 1.0


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def shrunk(C, N, m):
    return (C + SHRINK * m) / (N + SHRINK)


def vtilde(omega, Nk, Nkp, th_k, th_kp):
    A = float(np.sum(omega * th_k)); B = float(np.sum(omega * th_kp))
    vA = float(np.sum(omega**2 * th_k * (1 - th_k) / Nk))
    vB = float(np.sum(omega**2 * th_kp * (1 - th_kp) / Nkp))
    R = A / B
    V = vA / A**2 + vB / B**2
    return R, V, abs(R) * np.sqrt(V), (omega.sum())**2 / (omega**2).sum()


def main():
    agg = pd.read_parquet(KDD)
    rows = []
    for (k, kp) in PAIRS:
        m = build_pair(agg, k, kp)
        if len(m) < 2:
            print(f"pair ({k},{kp}): no data, skipping")
            continue
        Nk = m["Nk"].to_numpy(float); Nkp = m["Nkp"].to_numpy(float)
        Ck = m["Ck"].to_numpy(float); Ckp = m["Ckp"].to_numpy(float)
        bar_k = float(Ck.sum() / max(Nk.sum(), 1))
        bar_kp = float(Ckp.sum() / max(Nkp.sum(), 1))
        th_k = shrunk(Ck, Nk, bar_k); th_kp = shrunk(Ckp, Nkp, bar_kp)
        n_imp = int(Nk.sum() + Nkp.sum())
        print(f"\npair ({k},{kp}): pairs={len(m):,}, imps={n_imp:,}, "
              f"bar_theta=({bar_k:.4f},{bar_kp:.4f})")
        # Three schemes
        schemes = {
            "uniform":  np.ones_like(Nk),
            "min":      np.minimum(Nk, Nkp),
            "harmonic": Nk * Nkp / (Nk + Nkp),
        }
        _, V_u, _, _ = vtilde(schemes["uniform"], Nk, Nkp, th_k, th_kp)
        for name, w in schemes.items():
            R, V, SE, ess = vtilde(w, Nk, Nkp, th_k, th_kp)
            are = V_u / V if V > 0 else float("inf")
            print(f"  {name:9s}  R={R:.4f}  SE={SE:.3e}  ARE={are:6.2f}  ESS={ess:.2e}")
            rows.append({"pair_k": k, "pair_kp": kp, "n_pairs": len(m),
                         "n_imps": n_imp, "scheme": name,
                         "R_hat": R, "SE": SE, "V": V,
                         "ARE_vs_uniform": are, "ESS": ess})

    pd.DataFrame(rows).to_csv(OUT / "kdd_adjacent_chain.csv", index=False)
    print(f"\nWrote {OUT / 'kdd_adjacent_chain.csv'}")


if __name__ == "__main__":
    main()
