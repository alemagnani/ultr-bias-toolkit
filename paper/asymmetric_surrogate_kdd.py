"""Compute asymmetric surrogate alpha V_k + beta V_{k'} for harmonic vs
uniform on full KDD. The unconditional CS bound (Theorem 2 of the short
paper) is on V_k + V_{k'}; whether the asymmetric version also reduces is
an empirical question. We estimate alpha,beta from observed marginal CTRs
p_hat_k = sum C_k / sum N_k.
"""
from pathlib import Path
import numpy as np
import pandas as pd

KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")


def main():
    agg = pd.read_parquet(KDD)
    a = agg[agg.position == 1][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nk","clicks":"Ck"})
    b = agg[agg.position == 2][["queryID","adID","impressions","clicks"]]\
        .rename(columns={"impressions":"Nkp","clicks":"Ckp"})
    m = a.merge(b, on=["queryID","adID"], how="inner")
    Nk = m.Nk.to_numpy(float); Nkp = m.Nkp.to_numpy(float)
    Ck = m.Ck.to_numpy(float); Ckp = m.Ckp.to_numpy(float)
    p_k = Ck.sum() / Nk.sum(); p_kp = Ckp.sum() / Nkp.sum()
    alpha = (1 - p_k) / p_k
    beta = (1 - p_kp) / p_kp
    print(f"p_hat_1 = {p_k:.4f}   p_hat_2 = {p_kp:.4f}")
    print(f"alpha = (1-p_1)/p_1 = {alpha:.4f}")
    print(f"beta  = (1-p_2)/p_2 = {beta:.4f}")
    print(f"alpha/beta = {alpha/beta:.4f}")

    schemes = {
        "uniform":  np.ones_like(Nk),
        "min":      np.minimum(Nk, Nkp),
        "harmonic": Nk * Nkp / (Nk + Nkp),
    }
    print(f"\n{'scheme':10s}  {'V_k':>10s}  {'V_kp':>10s}  {'V_k+V_kp':>10s}  "
          f"{'aV+bV':>10s}  {'norm:V':>10s}  {'norm:aV+bV':>10s}")
    base_sym = None; base_asym = None
    for name, w in schemes.items():
        S = w.sum(); S2 = (w**2).sum()
        V_k  = float(np.sum(w**2 / Nk)  / S**2)
        V_kp = float(np.sum(w**2 / Nkp) / S**2)
        sym = V_k + V_kp
        asym = alpha * V_k + beta * V_kp
        if name == "uniform":
            base_sym, base_asym = sym, asym
        print(f"{name:10s}  {V_k:.3e}  {V_kp:.3e}  {sym:.3e}  {asym:.3e}  "
              f"{sym/base_sym:8.4f}  {asym/base_asym:8.4f}")


if __name__ == "__main__":
    main()
