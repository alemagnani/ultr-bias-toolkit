"""Compare additional weighting schemes vs harmonic on full KDD (1,2) pair.

Adds geometric mean sqrt(N_k N_k'), log-count log(1+min(N_k,N_k')),
capped harmonic at 99th percentile, and reports plug-in Vtilde + ARE + ESS.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")


def shrunk(C, N, m):
    return (C + m) / (N + 1.0)


def V(omega, Nk, Nkp, th_k, th_kp):
    A = float(np.sum(omega * th_k)); B = float(np.sum(omega * th_kp))
    vA = float(np.sum(omega**2 * th_k * (1 - th_k) / Nk))
    vB = float(np.sum(omega**2 * th_kp * (1 - th_kp) / Nkp))
    R = A / B
    Vt = vA / A**2 + vB / B**2
    return R, Vt, abs(R) * np.sqrt(Vt), (omega.sum())**2 / (omega**2).sum()


def main():
    agg = pd.read_parquet(KDD)
    a = agg[agg.position == 1][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nk","clicks":"Ck"})
    b = agg[agg.position == 2][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nkp","clicks":"Ckp"})
    m = a.merge(b, on=["queryID","adID"], how="inner")
    Nk = m.Nk.to_numpy(float); Nkp = m.Nkp.to_numpy(float)
    Ck = m.Ck.to_numpy(float); Ckp = m.Ckp.to_numpy(float)
    bk = Ck.sum()/Nk.sum(); bkp = Ckp.sum()/Nkp.sum()
    th_k = shrunk(Ck, Nk, bk); th_kp = shrunk(Ckp, Nkp, bkp)

    schemes = {
        "uniform":           np.ones_like(Nk),
        "min":               np.minimum(Nk, Nkp),
        "geometric":         np.sqrt(Nk * Nkp),
        "log_min":           np.log1p(np.minimum(Nk, Nkp)),
        "harmonic":          Nk * Nkp / (Nk + Nkp),
    }
    h = schemes["harmonic"]
    schemes["harmonic_cap99"] = np.minimum(h, np.percentile(h, 99.0))

    _, V_u, _, _ = V(schemes["uniform"], Nk, Nkp, th_k, th_kp)
    rows = []
    for name, w in schemes.items():
        R, Vt, SE, ess = V(w, Nk, Nkp, th_k, th_kp)
        rows.append({"scheme": name, "R": R, "SE": SE, "V": Vt,
                     "ARE_vs_U": V_u / Vt, "ESS": ess})
        print(f"{name:18s}  R={R:.4f}  SE={SE:.3e}  ARE={V_u/Vt:6.2f}  ESS={ess:.3e}")
    pd.DataFrame(rows).to_csv(OUT / "kdd_scheme_comparison.csv", index=False)


if __name__ == "__main__":
    main()
