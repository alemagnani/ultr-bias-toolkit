"""
Asymmetric Cauchy--Schwarz: counterexample + structured adversarial search.

Investigates whether harmonic weighting always reduces the asymmetric
delta-method surrogate

    f_{alpha,beta}(omega) = (alpha V_k(omega) + beta V_{k'}(omega))
                          = sum_i omega_i^2 s_i / (sum_i omega_i)^2,
    s_i = alpha / N_k^i + beta / N_{k'}^i.

Theorem~\\ref{thm:cs} only proves f_{1,1}(h) <= f_{1,1}(1). The asymmetric
inequality f_{alpha,beta}(h) <= f_{alpha,beta}(1) was previously checked
empirically on 5,000 random uniform configs and held in all of them.

This script demonstrates the conjecture is FALSE in general:
  (1) Concrete n=2 counterexample with closed-form check.
  (2) n=50 structured counterexample (49 imbalanced + 1 balanced doc).
  (3) Structured adversarial search over count "types" (not uniform random):
      mixtures of "imbalanced" (a >> b) and "balanced" (a ~ b) docs, sweeping
      mixture fraction and (alpha, beta).

Outputs:
  paper/reviewer_response/asymmetric_counterexample.json
  paper/reviewer_response/asymmetric_structured_search.csv
  paper/reviewer_response/asymmetric_worst_ratio_heatmap.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
OUT.mkdir(exist_ok=True)


def harmonic(a, b):
    return a * b / (a + b)


def f_surr(omega, a, b, alpha, beta):
    """alpha V_k(omega) + beta V_{k'}(omega)."""
    s = alpha / a + beta / b
    num = np.sum(omega ** 2 * s)
    den = np.sum(omega) ** 2
    return num / den


def closed_form_check():
    print("=" * 70)
    print("(1) Closed-form n=2 counterexample")
    print("=" * 70)
    a = np.array([100.0, 2.0])
    b = np.array([1.0, 2.0])
    alpha, beta = 100.0, 0.01
    h = harmonic(a, b)
    one = np.ones_like(a)
    fh = f_surr(h, a, b, alpha, beta)
    fo = f_surr(one, a, b, alpha, beta)
    print(f"  counts: a={a.tolist()}, b={b.tolist()}")
    print(f"  alpha={alpha}, beta={beta}  (rho=alpha/beta={alpha/beta:g})")
    print(f"  h = {h.tolist()}")
    print(f"  f(harmonic) = {fh:.6f}")
    print(f"  f(uniform)  = {fo:.6f}")
    print(f"  ratio f(h)/f(1) = {fh/fo:.6f}  ({'HARMONIC LOSES' if fh > fo else 'harmonic wins'})")
    return {"n2": {"fh": fh, "fo": fo, "ratio": fh / fo}}


def structured_n50():
    print("\n" + "=" * 70)
    print("(2) Structured n=50 counterexample")
    print("=" * 70)
    n_imb, n_bal = 49, 1
    a = np.concatenate([np.full(n_imb, 100.0), np.full(n_bal, 2.0)])
    b = np.concatenate([np.full(n_imb, 1.0), np.full(n_bal, 2.0)])
    alpha, beta = 100.0, 0.01
    h = harmonic(a, b)
    fh = f_surr(h, a, b, alpha, beta)
    fo = f_surr(np.ones_like(a), a, b, alpha, beta)
    print(f"  {n_imb} docs at (a,b)=(100,1), {n_bal} doc at (2,2)")
    print(f"  alpha={alpha}, beta={beta}")
    print(f"  f(harmonic) = {fh:.6f}, f(uniform) = {fo:.6f}")
    print(f"  ratio = {fh/fo:.6f}  ({'HARMONIC LOSES' if fh > fo else 'harmonic wins'})")
    return {"n50": {"fh": fh, "fo": fo, "ratio": fh / fo}}


def structured_search(seed=0, n_per_axis=15, mixture_grid=None, n_docs=50,
                      alpha_range=(0.05, 100.0), beta_range=(0.05, 100.0),
                      tag=""):
    """Search over structured (rather than uniform-random) count configs.

    For each (alpha, beta) pair, build documents as a mixture of two
    archetypes:
      type-A ("imbalanced"): (a, b) = (M, 1) with M in {2, 5, 10, 50, 200}
      type-B ("balanced"):   (a, b) = (b0, b0) with b0 in {2, 5, 20}
    Sweep the fraction of type-A docs from 0 to 1.

    Report worst-case f(h)/f(1) over the structured configs, per (alpha,beta).
    This finds the regimes where harmonic loses; previous random-uniform
    search smoothed them out.
    """
    print("\n" + "=" * 70)
    print("(3) Structured adversarial search")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    alpha_grid = np.geomspace(alpha_range[0], alpha_range[1], n_per_axis)
    beta_grid = np.geomspace(beta_range[0], beta_range[1], n_per_axis)
    M_values = [2, 5, 10, 50, 200]
    b0_values = [2, 5, 20]
    if mixture_grid is None:
        mixture_grid = np.linspace(0.0, 1.0, 21)  # 0%, 5%, ..., 100% type-A

    rows = []
    for alpha in alpha_grid:
        for beta in beta_grid:
            best_ratio = 0.0
            best_cfg = None
            for M in M_values:
                for b0 in b0_values:
                    for frac in mixture_grid:
                        n_imb = int(round(n_docs * frac))
                        n_bal = n_docs - n_imb
                        if n_imb == 0 and n_bal == 0:
                            continue
                        a = np.concatenate([
                            np.full(n_imb, float(M)),
                            np.full(n_bal, float(b0)),
                        ])
                        b = np.concatenate([
                            np.full(n_imb, 1.0),
                            np.full(n_bal, float(b0)),
                        ])
                        h = harmonic(a, b)
                        fh = f_surr(h, a, b, alpha, beta)
                        fo = f_surr(np.ones_like(a), a, b, alpha, beta)
                        r = fh / fo
                        if r > best_ratio:
                            best_ratio = r
                            best_cfg = {"M": M, "b0": b0, "frac": frac,
                                         "n_imb": n_imb, "n_bal": n_bal,
                                         "fh": fh, "fo": fo}
            rows.append({
                "alpha": alpha, "beta": beta, "rho": alpha / beta,
                "worst_ratio": best_ratio,
                "M": best_cfg["M"] if best_cfg else None,
                "b0": best_cfg["b0"] if best_cfg else None,
                "frac": best_cfg["frac"] if best_cfg else None,
                "harmonic_loses": int(best_ratio > 1.0),
            })
    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    df.to_csv(OUT / f"asymmetric_structured_search{suffix}.csv", index=False)
    n_lose = int(df["harmonic_loses"].sum())
    print(f"  Tested {len(df)} (alpha, beta) cells with structured count search.")
    print(f"  Cells where harmonic loses to uniform on the asymmetric surrogate: "
          f"{n_lose}/{len(df)} ({100*n_lose/len(df):.1f}%)")
    print(f"  Worst observed ratio f(h)/f(1) = {df['worst_ratio'].max():.4f}")
    worst_row = df.loc[df["worst_ratio"].idxmax()]
    print(f"    at (alpha, beta) = ({worst_row['alpha']:.3g}, {worst_row['beta']:.3g}), "
          f"M={worst_row['M']}, b0={worst_row['b0']}, frac={worst_row['frac']:.2f}")
    return df


def heatmap(df: pd.DataFrame, tag: str = ""):
    pivot = df.pivot(index="alpha", columns="beta", values="worst_ratio")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=0.9, vmax=1.1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"worst-case $f(h)/f(1)$ over structured counts")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2g}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.2g}" for x in pivot.index])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"Asymmetric surrogate: worst $f(h)/f(1)$ over structured counts"
                 + "\n(red=harmonic loses, blue=harmonic wins; bound is small)")
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    out_png = OUT / f"asymmetric_worst_ratio_heatmap{suffix}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  heatmap saved to {out_png}")


def main():
    summary = {}
    summary.update(closed_form_check())
    summary.update(structured_n50())
    # Wide range (extreme): alpha,beta in [0.05, 100]
    df_wide = structured_search(alpha_range=(0.05, 100.0),
                                 beta_range=(0.05, 100.0), tag="wide")
    heatmap(df_wide, tag="wide")
    # Realistic range for PBM with eta=1, M=10 positions:
    # p_k = 1/k, alpha_k = (1-p_k)/p_k = k - 1, so alpha ranges over [0, 9].
    # For adjacent pairs (k, k+1): alpha = k-1, beta = k. ratio in [1, ~9].
    # For non-adjacent: alpha/beta in [1/9, 9].
    df_real = structured_search(alpha_range=(0.1, 10.0),
                                 beta_range=(0.1, 10.0), tag="realistic")
    heatmap(df_real, tag="realistic")
    summary["structured_search_wide"] = {
        "n_cells": int(len(df_wide)),
        "n_harmonic_loses": int(df_wide["harmonic_loses"].sum()),
        "frac_harmonic_loses": float(df_wide["harmonic_loses"].mean()),
        "worst_ratio": float(df_wide["worst_ratio"].max()),
    }
    summary["structured_search_realistic"] = {
        "n_cells": int(len(df_real)),
        "n_harmonic_loses": int(df_real["harmonic_loses"].sum()),
        "frac_harmonic_loses": float(df_real["harmonic_loses"].mean()),
        "worst_ratio": float(df_real["worst_ratio"].max()),
        "median_worst_ratio": float(df_real["worst_ratio"].median()),
    }
    (OUT / "asymmetric_counterexample.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {OUT / 'asymmetric_counterexample.json'}")


if __name__ == "__main__":
    main()
