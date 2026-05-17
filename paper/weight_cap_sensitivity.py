"""Weight-cap sensitivity for harmonic on full KDD.

Addresses the reviewer concern that ARE = 4.5x with ESS = 1849 out of
4.61M pairs is fragile. We cap omega_h at percentiles {99.99, 99.9, 99, 95, 90}
and re-compute plug-in Vtilde, ARE-vs-uniform, R_hat, and ESS.

A robust result: ARE stays in the same ballpark across caps.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
PAIR = (1, 2)
PCTS = [99.99, 99.9, 99.0, 95.0, 90.0]
SHRINK = 1.0


def shrunk(C, N, marg):
    a, b = marg * SHRINK, (1 - marg) * SHRINK
    return (C + a) / (N + a + b)


def vtilde(omega, Nk, Nkp, th_k, th_kp):
    A = float(np.sum(omega * th_k)); B = float(np.sum(omega * th_kp))
    vA = float(np.sum(omega**2 * th_k * (1 - th_k) / Nk))
    vB = float(np.sum(omega**2 * th_kp * (1 - th_kp) / Nkp))
    R = A / B
    V = vA / A**2 + vB / B**2
    return R, V, abs(R) * np.sqrt(V)


def main():
    agg = pd.read_parquet(KDD_PARQUET)
    a = agg[agg.position == PAIR[0]][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nk","clicks":"Ck"})
    b = agg[agg.position == PAIR[1]][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nkp","clicks":"Ckp"})
    m = a.merge(b, on=["queryID","adID"], how="inner")
    Nk = m.Nk.to_numpy(float); Nkp = m.Nkp.to_numpy(float)
    Ck = m.Ck.to_numpy(float); Ckp = m.Ckp.to_numpy(float)
    bk = Ck.sum() / Nk.sum(); bkp = Ckp.sum() / Nkp.sum()
    th_k = shrunk(Ck, Nk, bk); th_kp = shrunk(Ckp, Nkp, bkp)

    omega_u = np.ones_like(Nk)
    omega_h = Nk * Nkp / (Nk + Nkp)
    R_u, V_u, SE_u = vtilde(omega_u, Nk, Nkp, th_k, th_kp)
    R_h, V_h, SE_h = vtilde(omega_h, Nk, Nkp, th_k, th_kp)
    print(f"baseline uniform : R={R_u:.4f} SE={SE_u:.3e}")
    print(f"baseline harmonic: R={R_h:.4f} SE={SE_h:.3e} ARE={V_u/V_h:.2f} "
          f"ESS={(omega_h.sum())**2/(omega_h**2).sum():.3e}")

    rows = [{"cap_pct": None, "scheme": "uniform", "R": R_u, "SE": SE_u,
             "V": V_u, "ARE": 1.0, "ESS": float(len(omega_u))}]
    rows.append({"cap_pct": None, "scheme": "harmonic", "R": R_h, "SE": SE_h,
                 "V": V_h, "ARE": V_u / V_h,
                 "ESS": (omega_h.sum())**2 / (omega_h**2).sum()})

    for p in PCTS:
        cap = float(np.percentile(omega_h, p))
        omega_c = np.minimum(omega_h, cap)
        R_c, V_c, SE_c = vtilde(omega_c, Nk, Nkp, th_k, th_kp)
        ess = (omega_c.sum())**2 / (omega_c**2).sum()
        n_capped = int((omega_h > cap).sum())
        share_before = float((omega_h**2).sum())
        share_after  = float((omega_c**2).sum())
        rows.append({"cap_pct": p, "scheme": f"harmonic@{p}", "cap": cap,
                     "n_capped": n_capped, "R": R_c, "SE": SE_c, "V": V_c,
                     "ARE": V_u / V_c, "ESS": ess,
                     "weight_share_reduction": 1 - share_after/share_before})
        print(f"cap@{p:5.2f}%: thresh={cap:.2f} n_capped={n_capped:>7d} "
              f"R={R_c:.4f} SE={SE_c:.3e} ARE={V_u/V_c:5.2f} ESS={ess:.2e}")

    pd.DataFrame(rows).to_csv(OUT / "kdd_weight_cap_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
