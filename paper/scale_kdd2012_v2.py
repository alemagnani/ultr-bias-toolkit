"""KDD scaling experiment v2: BOTH bootstrap methods at every scale + SamueChan replication.

Differences from scale_kdd2012.py:
  (a) For each subsample we now compute *both* the (q,ad)-row bootstrap
      (resamples the population of pairs with replacement, keeping click
      counts fixed) and the parametric click-noise bootstrap (resamples
      C ~ Bin(N, C/N) per cell). The two measure different sources of
      randomness and can disagree.
  (b) Same impression-level subsampling as v1.
  (c) Replicate SamueChan-style (row-level subsample + drop Impression
      column) at three matched scales to verify whether the dramatic
      "harmonic helps" we saw on the SamueChan subset is reproducible
      under that exact procedure.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
KDD_RAW     = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/track2/training.txt")

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


def ratio(omega, Nk, Ck, Nkp, Ckp):
    A = float(np.sum(omega * Ck / Nk))
    B = float(np.sum(omega * Ckp / Nkp))
    return A / B if B > 0 else float("nan")


def vk(omega, N):
    s = float(np.sum(omega))
    return float(np.sum(omega**2 / N) / s**2) if s > 0 else float("inf")


def click_boot(m: pd.DataFrame, scheme: str, n_iter: int, seed: int):
    rng = np.random.default_rng(seed)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    th_k = np.clip(m["Ck"].to_numpy(dtype=float) / Nk, 0.0, 1.0)
    th_kp = np.clip(m["Ckp"].to_numpy(dtype=float) / Nkp, 0.0, 1.0)
    omega = weights(scheme, Nk, Nkp)
    Nk_int = Nk.astype(np.int64)
    Nkp_int = Nkp.astype(np.int64)
    est = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, th_k).astype(float)
        Ckp = rng.binomial(Nkp_int, th_kp).astype(float)
        est[it] = ratio(omega, Nk, Ck, Nkp, Ckp)
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def qad_boot(m: pd.DataFrame, scheme: str, n_iter: int, seed: int):
    """Bootstrap by resampling (q,ad) rows with replacement, keeping clicks fixed."""
    rng = np.random.default_rng(seed)
    n = len(m)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck = m["Ck"].to_numpy(dtype=float)
    Ckp = m["Ckp"].to_numpy(dtype=float)
    fn = lambda nk, nkp: weights(scheme, nk, nkp)
    est = np.empty(n_iter)
    for it in range(n_iter):
        idx = rng.integers(0, n, size=n)
        omega = fn(Nk[idx], Nkp[idx])
        est[it] = ratio(omega, Nk[idx], Ck[idx], Nkp[idx], Ckp[idx])
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1))


def subsample_impressions(agg: pd.DataFrame, f: float, rng) -> pd.DataFrame:
    """Bin/Hyper subsample at impression level: each impression kept iid with prob f."""
    N = agg["impressions"].to_numpy(dtype=np.int64)
    C = agg["clicks"].to_numpy(dtype=np.int64)
    tilde_N = rng.binomial(N, f)
    tilde_C = rng.hypergeometric(C, N - C, tilde_N)
    out = agg.copy()
    out["impressions"] = tilde_N
    out["clicks"] = tilde_C
    return out[out["impressions"] > 0]


def subsample_rows_drop_impression(target_rows: int, seed: int) -> pd.DataFrame:
    """SamueChan replication: random row-subsample of training.txt, then
    treat each row as 1 impression (drop Impression column)."""
    rng = np.random.default_rng(seed)
    print(f"  [SamueChan] streaming, target {target_rows:,} rows")
    parts = []
    total_kept = 0
    chunk_size = 5_000_000
    # First count rows? Use known 149,639,105
    total_rows = 149_639_105
    p_keep = target_rows / total_rows
    for chunk in pd.read_csv(KDD_RAW, sep="\t", header=None,
                              usecols=[0, 3, 6, 7],
                              names=["click", "adID", "position", "queryID"],
                              dtype={"click": np.int8, "adID": np.int64,
                                     "position": np.int8, "queryID": np.int64},
                              chunksize=chunk_size):
        keep = rng.random(len(chunk)) < p_keep
        sub = chunk[keep]
        if len(sub) == 0:
            continue
        # Aggregate at cell level: row count, click sum
        agg = (sub.groupby(["queryID", "adID", "position"], sort=False)
                  .agg(impressions=("click", "size"),
                       clicks=("click", "sum"))
                  .reset_index())
        parts.append(agg)
        total_kept += len(sub)
    big = pd.concat(parts, ignore_index=True)
    final = (big.groupby(["queryID", "adID", "position"], sort=False)
                .agg(impressions=("impressions", "sum"),
                     clicks=("clicks", "sum"))
                .reset_index())
    print(f"  [SamueChan] kept {total_kept:,} rows -> {len(final):,} cells")
    return final


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def evaluate(m, alpha, beta, seed) -> list[dict]:
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    rows = []
    for scheme in SCHEMES:
        omega = weights(scheme, Nk, Nkp)
        Vk_ = vk(omega, Nk)
        Vkp_ = vk(omega, Nkp)
        Rh = ratio(omega, Nk, m["Ck"].to_numpy(dtype=float),
                   Nkp, m["Ckp"].to_numpy(dtype=float))
        c_std = click_boot(m, scheme, N_BOOT, seed)
        q_std = qad_boot(m, scheme, N_BOOT, seed + 1)
        rows.append({
            "scheme": scheme,
            "n_pairs": int(len(m)),
            "Vk_plus_Vkp": Vk_ + Vkp_,
            "asym": alpha * Vk_ + beta * Vkp_,
            "R_hat": Rh,
            "click_boot_std": c_std,
            "qad_boot_std":   q_std,
        })
    return rows


def main():
    print(f"Loading {KDD_PARQUET}")
    agg = pd.read_parquet(KDD_PARQUET)
    print(f"  cells {len(agg):,}, impressions {agg['impressions'].sum():,}")

    # --- impression-level scaling ---
    rows_imp = []
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
                r.update(f=f, rep=rep, mode="impression", n_imp=n_imp, seed=seed)
            rows_imp.extend(res)
            print(f"  [imp] f={f} rep={rep}: pairs={len(m):>9,}, "
                  f"imps={n_imp:>11,}, "
                  + ", ".join([f"{r['scheme'][:3]}: c={r['click_boot_std']:.4f} q={r['qad_boot_std']:.4f}"
                              for r in res])
                  + f"  ({time.time()-t0:.1f}s)")

    df_imp = pd.DataFrame(rows_imp)
    df_imp.to_csv(OUT / "kdd2012_scaling_v2_impression.csv", index=False)

    # --- SamueChan-style (row-level + drop Impression) at three target row counts ---
    rows_sc = []
    sc_targets = [200_000, 1_000_000, 5_000_000]
    for trg in sc_targets:
        for rep in range(REPS):
            seed = int(1e7 + trg + rep)
            sub = subsample_rows_drop_impression(trg, seed)
            m = build_pair(sub, *PAIR)
            n_imp = int(sub["impressions"].sum())
            if len(m) < 2:
                continue
            res = evaluate(m, alpha=0.5, beta=1.0, seed=seed)
            for r in res:
                r.update(f=trg / 149_639_105, rep=rep, mode="samuechan",
                         n_imp=n_imp, seed=seed, target_rows=trg)
            rows_sc.extend(res)
            print(f"  [sc ] trg={trg} rep={rep}: pairs={len(m):>9,}, imps={n_imp:>11,}, "
                  + ", ".join([f"{r['scheme'][:3]}: c={r['click_boot_std']:.4f} q={r['qad_boot_std']:.4f}"
                              for r in res]))

    df_sc = pd.DataFrame(rows_sc)
    df_sc.to_csv(OUT / "kdd2012_scaling_v2_samuechan.csv", index=False)

    # ---- merged plot ----
    df_all = pd.concat([df_imp, df_sc], ignore_index=True)
    df_all.to_csv(OUT / "kdd2012_scaling_v2.csv", index=False)
    summary_imp = df_imp.groupby(["f", "scheme"]).agg(
        n_imp=("n_imp", "mean"),
        n_pairs=("n_pairs", "mean"),
        click_std=("click_boot_std", "mean"),
        qad_std=("qad_boot_std", "mean"),
        Vk_plus_Vkp=("Vk_plus_Vkp", "mean"),
        R_hat=("R_hat", "mean"),
    ).reset_index()
    summary_sc = df_sc.groupby(["f", "scheme"]).agg(
        n_imp=("n_imp", "mean"),
        n_pairs=("n_pairs", "mean"),
        click_std=("click_boot_std", "mean"),
        qad_std=("qad_boot_std", "mean"),
        Vk_plus_Vkp=("Vk_plus_Vkp", "mean"),
        R_hat=("R_hat", "mean"),
    ).reset_index()
    summary_imp.to_csv(OUT / "kdd2012_scaling_v2_summary_imp.csv", index=False)
    summary_sc.to_csv(OUT / "kdd2012_scaling_v2_summary_sc.csv", index=False)

    print("\n--- Impression-level subsample summary ---")
    print(summary_imp.to_string(index=False))
    print("\n--- SamueChan-style row-level subsample summary ---")
    print(summary_sc.to_string(index=False))

    # plot: harmonic % vs uniform on click and qad bootstrap, both subsampling modes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def reduction_curve(summary, scheme, col):
        u = summary[summary["scheme"] == "uniform"].set_index("f")[col]
        s = summary[summary["scheme"] == scheme].set_index("f")[col]
        f = sorted(u.index.intersection(s.index))
        red = (u.loc[f].values - s.loc[f].values) / u.loc[f].values * 100
        return summary[summary["scheme"] == scheme].set_index("f").loc[f]["n_imp"].values, red

    for ax, col, title in [
        (axes[0], "click_std", "Click-noise bootstrap (matches Theorem 4)"),
        (axes[1], "qad_std",   "(q, ad)-row bootstrap (used in earlier draft)")]:
        for scheme, color in [("min", "blue"), ("harmonic", "red")]:
            n, red = reduction_curve(summary_imp, scheme, col)
            ax.plot(n, red, "o-", color=color, label=f"{scheme} (impression-level)")
            n, red = reduction_curve(summary_sc, scheme, col)
            ax.plot(n, red, "s--", color=color, alpha=0.5,
                    label=f"{scheme} (SamueChan-style)")
        ax.axhline(0, color="grey", linestyle=":")
        ax.set_xlabel("Total impressions in subsample")
        ax.set_ylabel("% reduction vs uniform (positive = harmonic better)")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kdd2012_scaling_v2_compare.png", dpi=150)
    plt.close()
    print(f"\nSaved {OUT / 'kdd2012_scaling_v2_compare.png'}")


if __name__ == "__main__":
    main()
