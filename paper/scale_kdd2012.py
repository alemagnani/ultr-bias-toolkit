"""KDD Cup 2012 scaling experiment: harmonic vs uniform across sample sizes.

We start from the full aggregated cells (63.8M (q, ad, pos) cells, 235.6M
impressions). For each sample fraction f in {0.001, 0.003, 0.01, 0.03, 0.1,
0.3, 1.0} we draw an impression-level subsample by

    tilde_N_i ~ Binomial(N_i, f)
    tilde_C_i ~ Hypergeometric(N_i, C_i, tilde_N_i)

(each impression kept independently with prob f). Re-aggregate, build the
adjacent (1, 2) interventional pair table, and compute for each scheme:

  * variance surrogate  V_k(omega) + V_{k+1}(omega)
  * effective sample size  (sum w)^2 / sum w^2
  * point estimate  R_hat
  * parametric click-bootstrap std of R_hat (n_iter=100)

REPS subsamples per fraction give error bars. Plot all four quantities
versus effective n on log axes to show: at small n harmonic helps, at full
scale the actual variance becomes irrelevantly small and popularity
correlation eats some of the surrogate gain.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

FRACTIONS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
REPS = 3
PAIR = (1, 2)
N_BOOT = 100
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


def ess(omega):
    s1 = float(np.sum(omega))
    s2 = float(np.sum(omega * omega))
    return s1 * s1 / s2 if s2 > 0 else 0.0


def click_boot_std(m, scheme, n_iter, seed):
    rng = np.random.default_rng(seed)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    theta_k = np.clip(m["Ck"].to_numpy(dtype=float) / Nk, 0.0, 1.0)
    theta_kp = np.clip(m["Ckp"].to_numpy(dtype=float) / Nkp, 0.0, 1.0)
    omega = weights(scheme, Nk, Nkp)
    Nk_int = Nk.astype(np.int64)
    Nkp_int = Nkp.astype(np.int64)
    est = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, theta_k).astype(float)
        Ckp = rng.binomial(Nkp_int, theta_kp).astype(float)
        est[it] = ratio(omega, Nk, Ck, Nkp, Ckp)
    e = est[np.isfinite(est)]
    return float(np.std(e, ddof=1)), float(np.mean(e))


def subsample_cells(agg: pd.DataFrame, f: float, rng: np.random.Generator) -> pd.DataFrame:
    """Impression-level subsample: tilde_N ~ Bin(N, f), tilde_C ~ Hyper(N, C, tilde_N)."""
    N = agg["impressions"].to_numpy(dtype=np.int64)
    C = agg["clicks"].to_numpy(dtype=np.int64)
    tilde_N = rng.binomial(N, f)
    # Hypergeometric per cell: draw tilde_C successes given C successes, N-C failures, tilde_N draws
    # numpy hypergeometric: hypergeometric(ngood, nbad, nsample)
    tilde_C = rng.hypergeometric(C, N - C, tilde_N)
    out = agg.copy()
    out["impressions"] = tilde_N
    out["clicks"] = tilde_C
    out = out[out["impressions"] > 0]
    return out


def build_pair(agg, k, kp):
    a = agg[agg["position"] == k][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][["queryID", "adID", "impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=["queryID", "adID"], how="inner")


def evaluate(m, alpha=0.5, beta=1.0, seed=0):
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    rows = []
    for scheme in SCHEMES:
        omega = weights(scheme, Nk, Nkp)
        Vk = vk(omega, Nk)
        Vkp = vk(omega, Nkp)
        e = ess(omega)
        Rh_full = ratio(omega, Nk, m["Ck"].to_numpy(dtype=float),
                        Nkp, m["Ckp"].to_numpy(dtype=float))
        boot_std, boot_R = click_boot_std(m, scheme, N_BOOT, seed)
        rows.append({"scheme": scheme,
                      "n_pairs": int(len(m)),
                      "ESS": e,
                      "Vk_plus_Vkp": Vk + Vkp,
                      "asym": alpha * Vk + beta * Vkp,
                      "R_hat": Rh_full,
                      "boot_R": boot_R,
                      "boot_std": boot_std})
    return rows


def main():
    print(f"Loading {KDD} ...")
    agg = pd.read_parquet(KDD)
    print(f"  cells: {len(agg):,}, impressions: {agg['impressions'].sum():,}")

    rows_all = []
    for f in FRACTIONS:
        for rep in range(REPS):
            seed = int(1e6 * f) + rep
            rng = np.random.default_rng(seed)
            t0 = time.time()
            sub = subsample_cells(agg, f, rng)
            m = build_pair(sub, *PAIR)
            n_imp_sub = int(sub["impressions"].sum())
            n_pairs = int(len(m))
            if n_pairs < 2:
                print(f"  f={f}, rep={rep}: too few pairs ({n_pairs}), skip")
                continue
            res = evaluate(m, alpha=0.5, beta=1.0, seed=seed)
            for r in res:
                r["f"] = f
                r["rep"] = rep
                r["n_imp_subsample"] = n_imp_sub
                r["seed"] = seed
            rows_all.extend(res)
            print(f"  f={f:.3f} rep={rep}: pairs={n_pairs:>9,}, "
                  f"impressions={n_imp_sub:>11,}, "
                  f"R_hat: " + ", ".join([f"{s[:3]}={r['R_hat']:.3f}" for s, r in zip(SCHEMES, res)]) +
                  f"  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows_all)
    df.to_csv(OUT / "kdd2012_scaling.csv", index=False)
    print(f"\nSaved {OUT / 'kdd2012_scaling.csv'}")

    # Aggregate over reps: mean and std per (f, scheme)
    agg_cols = ["n_pairs", "n_imp_subsample", "ESS", "Vk_plus_Vkp",
                "asym", "R_hat", "boot_std"]
    summary = (df.groupby(["f", "scheme"])[agg_cols].mean().reset_index()
               .merge(df.groupby(["f", "scheme"])[agg_cols].std().reset_index()
                          .rename(columns={c: f"{c}_std" for c in agg_cols}),
                      on=["f", "scheme"]))
    summary.to_csv(OUT / "kdd2012_scaling_summary.csv", index=False)

    # Plot panel
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    color = {"uniform": "black", "min": "blue", "harmonic": "red"}
    label = {"uniform": "Uniform", "min": "Min", "harmonic": "Harmonic"}

    # Panel 1: surrogate
    for s in SCHEMES:
        d = summary[summary["scheme"] == s].sort_values("n_imp_subsample")
        axes[0, 0].plot(d["n_imp_subsample"], d["Vk_plus_Vkp"], "o-",
                         color=color[s], label=label[s])
    axes[0, 0].set_xlabel("Total impressions in subsample")
    axes[0, 0].set_ylabel(r"$V_k + V_{k+1}$  (surrogate)")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Variance surrogate vs sample size")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: bootstrap std (actual click-noise variance)
    for s in SCHEMES:
        d = summary[summary["scheme"] == s].sort_values("n_imp_subsample")
        axes[0, 1].errorbar(d["n_imp_subsample"], d["boot_std"],
                             yerr=d["boot_std_std"],
                             fmt="o-", color=color[s], label=label[s])
    axes[0, 1].set_xlabel("Total impressions in subsample")
    axes[0, 1].set_ylabel(r"std($\hat{R}$) under click bootstrap")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Actual click-noise std vs sample size")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: relative reduction (% lower than uniform)
    base = summary[summary["scheme"] == "uniform"].set_index("f")
    for s in ["min", "harmonic"]:
        d = summary[summary["scheme"] == s].sort_values("f")
        red_surr = (base.loc[d["f"].values, "Vk_plus_Vkp"].to_numpy()
                    - d["Vk_plus_Vkp"].to_numpy()) \
                   / base.loc[d["f"].values, "Vk_plus_Vkp"].to_numpy() * 100
        red_boot = (base.loc[d["f"].values, "boot_std"].to_numpy()
                    - d["boot_std"].to_numpy()) \
                   / base.loc[d["f"].values, "boot_std"].to_numpy() * 100
        axes[1, 0].plot(d["n_imp_subsample"], red_surr, "o-",
                         color=color[s], label=f"{label[s]} surrogate")
        axes[1, 0].plot(d["n_imp_subsample"], red_boot, "x--",
                         color=color[s], alpha=0.7,
                         label=f"{label[s]} click-boot")
    axes[1, 0].axhline(0, color="grey", linestyle=":")
    axes[1, 0].set_xlabel("Total impressions in subsample")
    axes[1, 0].set_ylabel("% reduction vs uniform")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Relative reduction (positive = harmonic better)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: R_hat convergence
    for s in SCHEMES:
        d = summary[summary["scheme"] == s].sort_values("n_imp_subsample")
        axes[1, 1].errorbar(d["n_imp_subsample"], d["R_hat"],
                             yerr=d["R_hat_std"],
                             fmt="o-", color=color[s], label=label[s])
    axes[1, 1].set_xlabel("Total impressions in subsample")
    axes[1, 1].set_ylabel(r"$\hat{R}_{1,2}$")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(r"$\hat{R}$ convergence across data scales")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = OUT / "kdd2012_scaling.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
