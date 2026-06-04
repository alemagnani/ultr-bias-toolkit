"""Query-cluster bootstrap on full KDD pair (1,2).

Resample queries with replacement; for each sampled query include all its
(q, ad) pairs. Compute R_hat under uniform and harmonic for each bootstrap
sample. Report SE; compare to the plug-in delta-method SE reported in the
short paper (uniform 4.24e-3, harmonic 2.02e-3).

This is the "stronger probe of independence" the paper flags as future
work in section 3.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import time

KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
OUT = Path("/home/alessandro/workspace/ultr-bias-toolkit/paper/reviewer_response")
B = 200            # bootstrap iterations
SEED = 2026


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]]\
        .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def ratio(omega, Nk, Nkp, Ck, Ckp):
    th_k = Ck / np.maximum(Nk, 1.0)
    th_kp = Ckp / np.maximum(Nkp, 1.0)
    A = float(np.sum(omega * th_k))
    B = float(np.sum(omega * th_kp))
    return A / B if B > 0 else np.nan


def main():
    t0 = time.time()
    agg = pd.read_parquet(KDD)
    m = build_pair(agg, 1, 2)
    queries = m["queryID"].to_numpy()
    Nk = m["Nk"].to_numpy(float); Nkp = m["Nkp"].to_numpy(float)
    Ck = m["Ck"].to_numpy(float); Ckp = m["Ckp"].to_numpy(float)
    omega_h = Nk * Nkp / (Nk + Nkp)
    omega_u = np.ones_like(Nk)

    unique_qs, q_inverse = np.unique(queries, return_inverse=True)
    n_q = len(unique_qs)
    print(f"pairs={len(m):,}, unique queries={n_q:,}, bootstrap B={B}", flush=True)
    print(f"baseline (no resampling): R_U={ratio(omega_u, Nk, Nkp, Ck, Ckp):.4f}, "
          f"R_H={ratio(omega_h, Nk, Nkp, Ck, Ckp):.4f}")

    # Pre-group rows per query for fast bootstrap
    order = np.argsort(q_inverse, kind="stable")
    boundaries = np.r_[0, np.cumsum(np.bincount(q_inverse))]
    sorted_idx = order  # rows of (Nk, Nkp, Ck, Ckp) in query-grouped order

    rng = np.random.default_rng(SEED)
    R_u_boot = np.empty(B)
    R_h_boot = np.empty(B)
    for b in range(B):
        q_sample = rng.integers(0, n_q, size=n_q)
        # Gather rows for sampled queries
        # For each sampled query q, take rows boundaries[q]:boundaries[q+1]
        rows_idx = np.concatenate([
            sorted_idx[boundaries[q]:boundaries[q+1]] for q in q_sample
        ])
        nk = Nk[rows_idx]; nkp = Nkp[rows_idx]
        ck = Ck[rows_idx]; ckp = Ckp[rows_idx]
        w_u = np.ones_like(nk)
        w_h = nk * nkp / (nk + nkp)
        R_u_boot[b] = ratio(w_u, nk, nkp, ck, ckp)
        R_h_boot[b] = ratio(w_h, nk, nkp, ck, ckp)
        if (b + 1) % 20 == 0:
            print(f"  boot {b+1}/{B}  R_U_se={np.std(R_u_boot[:b+1], ddof=1):.4f}  "
                  f"R_H_se={np.std(R_h_boot[:b+1], ddof=1):.4f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)

    SE_u = float(np.std(R_u_boot, ddof=1))
    SE_h = float(np.std(R_h_boot, ddof=1))
    print(f"\nQuery-cluster bootstrap SE on full KDD ({B} iterations):")
    print(f"  Uniform  : SE = {SE_u:.4f}    (plug-in SE was 0.0042; ratio {SE_u/0.0042:.2f}x)")
    print(f"  Harmonic : SE = {SE_h:.4f}    (plug-in SE was 0.0020; ratio {SE_h/0.0020:.2f}x)")
    are = (SE_u / SE_h) ** 2
    print(f"  Implied ARE(harmonic, uniform) under query-cluster bootstrap = {are:.2f}")
    print(f"  (paper's plug-in ARE was 4.53)")

    pd.DataFrame({
        "scheme": ["uniform", "harmonic"],
        "qcluster_boot_SE": [SE_u, SE_h],
        "plugin_SE": [0.0042, 0.0020],
        "qcluster_ARE_vs_uniform": [1.0, are],
        "plugin_ARE_vs_uniform": [1.0, 4.53],
    }).to_csv(OUT / "kdd_qcluster_bootstrap.csv", index=False)
    print(f"\nWrote {OUT / 'kdd_qcluster_bootstrap.csv'}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
