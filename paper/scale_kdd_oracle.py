"""KDD scaling experiment: theorem's asymmetric oracle vs harmonic on real data.

Asymmetric oracle (Theorem 5):  omega_i* = N_k^i N_{k'}^i / (alpha N_{k'}^i + beta N_k^i)
which minimises the count-only surrogate alpha V_k + beta V_{k'}.

We try several (alpha, beta) values bracketing what KDD's converged R suggests
about the real propensity ratio:

  (alpha, beta) = (1, 1)         -> reduces to harmonic (sanity check)
  (alpha, beta) = (0.5, 1)       -> the value we used elsewhere in the paper
  (alpha, beta) = (0.1, 0.7)     -> implied by p_1 = 0.9, p_2 = 0.535
                                    (from converged R_hat ~ 1.68)
  (alpha, beta) = (0.05, 1.0)    -> stronger asymmetry, p_1 ~ 0.95

Plus we include a "full empirical-optimal" scheme that knows per-cell
click rates and minimises the actual click-noise variance:
  omega_i^** = 1 / [alpha theta_k(1-theta_k)/N_k + beta theta_{k'}(1-theta_{k'})/N_{k'}]

This is the *true* variance-optimal weight on real data; it's not deployable
(needs unknown propensities AND per-cell rates) but tells us whether the
theorem's count-only oracle is leaving variance on the table.

For each scheme we report click bootstrap, (q,ad)-row bootstrap, and query
cluster bootstrap stds, plus the count-only surrogate.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

FRACTIONS = [0.001, 0.01, 0.1, 1.0]
REPS = 2
PAIR = (1, 2)
N_BOOT = 200

# Schemes: (label, alpha, beta) for oracle variants; harmonic is (1, 1)
ORACLE_VARIANTS = {
    "oracle_a0.5_b1.0":   (0.5, 1.0),
    "oracle_a0.1_b0.7":   (0.1, 0.7),
    "oracle_a0.05_b1.0":  (0.05, 1.0),
}
SCHEMES = ["uniform", "harmonic"] + list(ORACLE_VARIANTS.keys()) + ["full_oracle"]


def weight(scheme, Nk, Nkp, Ck=None, Ckp=None):
    if scheme == "uniform":
        return np.ones_like(Nk, dtype=float)
    if scheme == "harmonic":
        return Nk * Nkp / (Nk + Nkp)
    if scheme.startswith("oracle_"):
        alpha, beta = ORACLE_VARIANTS[scheme]
        return Nk * Nkp / (alpha * Nkp + beta * Nk)
    if scheme == "full_oracle":
        # uses per-cell click rates and (alpha=0.5, beta=1)
        a, b = 0.5, 1.0
        th_k  = np.clip(Ck  / Nk,  1e-9, 1 - 1e-9)
        th_kp = np.clip(Ckp / Nkp, 1e-9, 1 - 1e-9)
        # variance per cell when omega = 1: a*theta(1-theta)/N at each pos
        # (proportional to actual variance contribution)
        s = a * th_k * (1 - th_k) / Nk + b * th_kp * (1 - th_kp) / Nkp
        return 1.0 / np.maximum(s, 1e-30)
    raise ValueError(scheme)


def vk(omega, N):
    s = float(np.sum(omega))
    return float(np.sum(omega**2 / N) / s**2) if s > 0 else float("inf")


def click_boot(m, omega, n_iter, seed):
    rng = np.random.default_rng(seed)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    th_k  = np.clip(m["Ck"].to_numpy(dtype=float)  / Nk,  0.0, 1.0)
    th_kp = np.clip(m["Ckp"].to_numpy(dtype=float) / Nkp, 0.0, 1.0)
    Nk_int = Nk.astype(np.int64); Nkp_int = Nkp.astype(np.int64)
    est = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, th_k).astype(float)
        Ckp = rng.binomial(Nkp_int, th_kp).astype(float)
        A = float(np.sum(omega * Ck / Nk))
        B = float(np.sum(omega * Ckp / Nkp))
        est[it] = A / B if B > 0 else np.nan
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def qad_boot_scheme(m, scheme, n_iter, seed):
    rng = np.random.default_rng(seed)
    n = len(m)
    Nk  = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck  = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    est = np.empty(n_iter)
    for it in range(n_iter):
        idx = rng.integers(0, n, size=n)
        nk, ck, nkp, ckp = Nk[idx], Ck[idx], Nkp[idx], Ckp[idx]
        omega = weight(scheme, nk, nkp, Ck=ck, Ckp=ckp)
        A = float(np.sum(omega * ck / nk))
        B = float(np.sum(omega * ckp / nkp))
        est[it] = A / B if B > 0 else np.nan
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def query_boot(m, omega, n_iter, seed):
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    a_row = omega * Ck / Nk
    b_row = omega * Ckp / Nkp
    queries = m["queryID"].to_numpy()
    g = pd.DataFrame({"q": queries, "a": a_row, "b": b_row})
    sums = g.groupby("q", sort=False)[["a", "b"]].sum()
    a_q = sums["a"].to_numpy(); b_q = sums["b"].to_numpy()
    n_q = len(a_q)
    rng = np.random.default_rng(seed)
    est = np.empty(n_iter)
    for it in range(n_iter):
        idx = rng.integers(0, n_q, size=n_q)
        A = float(a_q[idx].sum())
        B = float(b_q[idx].sum())
        est[it] = A / B if B > 0 else np.nan
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1)), int(n_q)


def subsample(agg, f, rng):
    N = agg["impressions"].to_numpy(dtype=np.int64)
    C = agg["clicks"].to_numpy(dtype=np.int64)
    tilde_N = rng.binomial(N, f)
    tilde_C = rng.hypergeometric(C, N - C, tilde_N)
    out = agg.copy()
    out["impressions"] = tilde_N
    out["clicks"] = tilde_C
    return out[out["impressions"] > 0]


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def main():
    print(f"Loading {KDD_PARQUET}", flush=True)
    agg = pd.read_parquet(KDD_PARQUET)
    rows = []
    for f in FRACTIONS:
        for rep in range(REPS):
            seed = int(1e6 * f) + rep
            rng = np.random.default_rng(seed)
            t0 = time.time()
            sub = subsample(agg, f, rng)
            m = build_pair(sub, *PAIR)
            n_imp = int(sub["impressions"].sum())
            if len(m) < 2:
                continue
            Nk = m["Nk"].to_numpy(dtype=float)
            Nkp = m["Nkp"].to_numpy(dtype=float)
            Ck = m["Ck"].to_numpy(dtype=float)
            Ckp = m["Ckp"].to_numpy(dtype=float)
            print(f"\nf={f}, rep={rep}: pairs={len(m):,}, imps={n_imp:,}", flush=True)
            for scheme in SCHEMES:
                omega = weight(scheme, Nk, Nkp, Ck=Ck, Ckp=Ckp)
                Vk_ = vk(omega, Nk); Vkp_ = vk(omega, Nkp)
                Rh = float(np.sum(omega * Ck / Nk)) / float(np.sum(omega * Ckp / Nkp))
                cstd = click_boot(m, omega, N_BOOT, seed)
                qstd = qad_boot_scheme(m, scheme, N_BOOT, seed + 1)
                Qstd, n_q = query_boot(m, omega, N_BOOT, seed + 2)
                rows.append({
                    "f": f, "rep": rep, "n_imp": n_imp, "n_pairs": int(len(m)),
                    "n_queries": n_q, "scheme": scheme,
                    "Vk_plus_Vkp": Vk_ + Vkp_,
                    "asym_aVk_bVkp_at_05_1": 0.5 * Vk_ + 1.0 * Vkp_,
                    "R_hat": Rh,
                    "click_std": cstd, "qad_std": qstd, "query_std": Qstd,
                })
                print(f"  {scheme:24s}  Vk+V'={Vk_+Vkp_:.3e}  R={Rh:.3f}  "
                      f"c={cstd:.4f}  q={qstd:.4f}  Q={Qstd:.4f}", flush=True)
            print(f"  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kdd_oracle_scaling.csv", index=False)
    summary = df.groupby(["f", "scheme"]).agg(
        n_imp=("n_imp", "mean"),
        Vk_plus_Vkp=("Vk_plus_Vkp", "mean"),
        R_hat=("R_hat", "mean"),
        click_std=("click_std", "mean"),
        qad_std=("qad_std", "mean"),
        query_std=("query_std", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "kdd_oracle_scaling_summary.csv", index=False)
    print("\n--- Summary ---", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
