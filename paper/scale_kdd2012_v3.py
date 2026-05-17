"""KDD scaling experiment v3: add query-cluster bootstrap.

Three bootstrap schemes are computed at every (f, scheme):

  click-bootstrap : keep counts, redraw clicks ~ Bin(N, C/N) per cell.
                    Theorem-4-relevant. No population randomness.
  (q,ad) bootstrap: resample (q, ad) rows iid with replacement.
                    Conflates click + per-row population variance, treats
                    rows as independent (they're not — same query rows are
                    correlated).
  query bootstrap : cluster-resample queryIDs with replacement; all
                    (q, ad) rows for a sampled query move together.
                    Standard cluster bootstrap; respects the natural
                    independence unit of the data.

Pre-aggregate each row's contribution to A and B once, then per-query
sum once, so each bootstrap iter is just an array indexing + sum.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

FRACTIONS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
REPS = 3
PAIR = (1, 2)
N_BOOT = 200
SCHEMES = ["uniform", "min", "harmonic"]


def weights(scheme, Nk, Nkp):
    if scheme == "uniform":
        return np.ones_like(Nk, dtype=float)
    if scheme == "min":
        return np.minimum(Nk, Nkp).astype(float)
    if scheme == "harmonic":
        return Nk * Nkp / (Nk + Nkp)


def vk(omega, N):
    s = float(np.sum(omega))
    return float(np.sum(omega**2 / N) / s**2) if s > 0 else float("inf")


def ratio_from_AB(A: np.ndarray, B: np.ndarray):
    return A / B if B > 0 else float("nan")


def click_boot(m, scheme, n_iter, seed):
    rng = np.random.default_rng(seed)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    th_k  = np.clip(m["Ck"].to_numpy(dtype=float)  / Nk,  0.0, 1.0)
    th_kp = np.clip(m["Ckp"].to_numpy(dtype=float) / Nkp, 0.0, 1.0)
    omega = weights(scheme, Nk, Nkp)
    Nk_int = Nk.astype(np.int64)
    Nkp_int = Nkp.astype(np.int64)
    est = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, th_k).astype(float)
        Ckp = rng.binomial(Nkp_int, th_kp).astype(float)
        A = float(np.sum(omega * Ck / Nk))
        B = float(np.sum(omega * Ckp / Nkp))
        est[it] = A / B if B > 0 else np.nan
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def qad_boot(m, scheme, n_iter, seed):
    """Resample (q,ad) rows iid with replacement, recompute weights, ratio."""
    rng = np.random.default_rng(seed)
    n = len(m)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    est = np.empty(n_iter)
    for it in range(n_iter):
        idx = rng.integers(0, n, size=n)
        nk, ck, nkp, ckp = Nk[idx], Ck[idx], Nkp[idx], Ckp[idx]
        omega = weights(scheme, nk, nkp)
        A = float(np.sum(omega * ck / nk))
        B = float(np.sum(omega * ckp / nkp))
        est[it] = A / B if B > 0 else np.nan
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def query_boot(m, scheme, n_iter, seed):
    """Cluster-resample queries with replacement; keep within-query rows together."""
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    omega = weights(scheme, Nk, Nkp)
    a_row = omega * Ck / Nk
    b_row = omega * Ckp / Nkp
    queries = m["queryID"].to_numpy()
    g = pd.DataFrame({"q": queries, "a": a_row, "b": b_row})
    sums = g.groupby("q", sort=False)[["a", "b"]].sum()
    a_q = sums["a"].to_numpy()
    b_q = sums["b"].to_numpy()
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


def subsample_impressions(agg, f, rng):
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


def evaluate(m, alpha, beta, seed):
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    rows = []
    n_q_global = None
    for scheme in SCHEMES:
        omega = weights(scheme, Nk, Nkp)
        Vk_ = vk(omega, Nk); Vkp_ = vk(omega, Nkp)
        Rh = float(np.sum(omega * m["Ck"].to_numpy(dtype=float) / Nk)) / \
             float(np.sum(omega * m["Ckp"].to_numpy(dtype=float) / Nkp))
        c_std = click_boot(m, scheme, N_BOOT, seed)
        q_std = qad_boot(m, scheme, N_BOOT, seed + 1)
        qry_std, n_q = query_boot(m, scheme, N_BOOT, seed + 2)
        n_q_global = n_q
        rows.append({"scheme": scheme,
                      "n_pairs": int(len(m)),
                      "n_queries": int(n_q),
                      "Vk_plus_Vkp": Vk_ + Vkp_,
                      "asym": alpha * Vk_ + beta * Vkp_,
                      "R_hat": Rh,
                      "click_std": c_std,
                      "qad_std":   q_std,
                      "query_std": qry_std})
    return rows


def main():
    print(f"Loading {KDD_PARQUET}", flush=True)
    agg = pd.read_parquet(KDD_PARQUET)
    print(f"  cells {len(agg):,}, impressions {agg['impressions'].sum():,}", flush=True)

    rows = []
    for f in FRACTIONS:
        for rep in range(REPS):
            seed = int(1e6 * f) + rep
            rng = np.random.default_rng(seed)
            t0 = time.time()
            sub = subsample_impressions(agg, f, rng)
            m = build_pair(sub, *PAIR)
            n_imp = int(sub["impressions"].sum())
            if len(m) < 2:
                continue
            res = evaluate(m, alpha=0.5, beta=1.0, seed=seed)
            for r in res:
                r.update(f=f, rep=rep, n_imp=n_imp, seed=seed)
            rows.extend(res)
            n_q = res[0]["n_queries"]
            print(f"  f={f:.3f} rep={rep}: pairs={len(m):>9,}, "
                  f"imps={n_imp:>11,}, queries={n_q:>9,}, " +
                  ", ".join([f"{r['scheme'][:3]}:c={r['click_std']:.4f},q={r['qad_std']:.4f},Q={r['query_std']:.4f}"
                            for r in res]) +
                  f"  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kdd2012_scaling_v3.csv", index=False)

    summary = df.groupby(["f", "scheme"]).agg(
        n_imp=("n_imp", "mean"),
        n_pairs=("n_pairs", "mean"),
        n_queries=("n_queries", "mean"),
        click_std=("click_std", "mean"),
        qad_std=("qad_std", "mean"),
        query_std=("query_std", "mean"),
        Vk_plus_Vkp=("Vk_plus_Vkp", "mean"),
        R_hat=("R_hat", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "kdd2012_scaling_v3_summary.csv", index=False)
    print("\n--- Summary ---")
    print(summary.to_string(index=False))

    # plot reduction vs uniform across the three bootstraps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title in [
        (axes[0], "click_std", "Click bootstrap (Theorem 4)"),
        (axes[1], "qad_std",   "(q, ad)-row bootstrap"),
        (axes[2], "query_std", "Query cluster bootstrap")]:
        u = summary[summary["scheme"] == "uniform"].set_index("f")
        for scheme, color in [("min", "blue"), ("harmonic", "red")]:
            s = summary[summary["scheme"] == scheme].set_index("f")
            f = sorted(u.index.intersection(s.index))
            red = (u.loc[f][col].values - s.loc[f][col].values) / u.loc[f][col].values * 100
            ax.plot(s.loc[f]["n_imp"].values, red, "o-", color=color, label=scheme)
        ax.axhline(0, color="grey", linestyle=":")
        ax.set_xlabel("Total impressions in subsample")
        ax.set_ylabel("% reduction vs uniform")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kdd2012_scaling_v3.png", dpi=150)
    plt.close()
    print(f"\nSaved {OUT / 'kdd2012_scaling_v3.png'}")


if __name__ == "__main__":
    main()
