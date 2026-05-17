"""Real-data validation for harmonic weighting -- click-noise variance + ESS.

For each dataset (KDD Cup 2012 full, Open Bandit), we report:

  (1) Variance surrogate V_k(omega) and alpha V_k + beta V_{k'} (deterministic
      on counts; what Theorem~\\ref{thm:cs} bounds).
  (2) Effective sample size  ESS(omega) = (sum omega)^2 / sum omega^2,
      a standard importance-sampling diagnostic for weight concentration.
  (3) Parametric click-bootstrap variance of R_hat conditional on counts:
      resample C_k^i ~ Binomial(N_k^i, C_k^i / N_k^i) for each cell, compute
      R_hat per scheme, std across iterations. This avoids the population-
      resampling artifact of the (q, d)-level bootstrap; it directly measures
      the click-noise variance the surrogate predicts.

We do NOT report the (q, d)-level bootstrap: at large n it conflates the
click-noise component with population-resampling fluctuations dominated by
heavy-tailed weights (effectively the IS sample-size effect).
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
OUT.mkdir(exist_ok=True)

KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")
OBD_ZIP = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/openbandit/open_bandit_dataset.zip")
OBD_INNER = "open_bandit_dataset/random/all/all.csv"


# ---------------------------------------------------------------------------
# Estimators and diagnostics
# ---------------------------------------------------------------------------

SCHEMES = ["uniform", "min", "harmonic"]


def weights(scheme: str, Nk: np.ndarray, Nkp: np.ndarray) -> np.ndarray:
    if scheme == "uniform":
        return np.ones_like(Nk, dtype=float)
    if scheme == "min":
        return np.minimum(Nk, Nkp).astype(float)
    if scheme == "harmonic":
        return Nk * Nkp / (Nk + Nkp)
    raise ValueError(scheme)


def ratio_estimator(omega, Nk, Ck, Nkp, Ckp):
    A = float(np.sum(omega * Ck / Nk))
    B = float(np.sum(omega * Ckp / Nkp))
    return A / B if B > 0 else float("nan")


def vk_term(omega, N):
    s = float(np.sum(omega))
    return float(np.sum(omega**2 / N) / s**2) if s > 0 else float("inf")


def ess(omega: np.ndarray) -> float:
    """Effective sample size: (sum w)^2 / sum w^2, in [1, n]."""
    s1 = float(np.sum(omega))
    s2 = float(np.sum(omega * omega))
    return s1 * s1 / s2 if s2 > 0 else 0.0


def click_bootstrap_std(m: pd.DataFrame, scheme: str,
                         n_iter: int = 200, seed: int = 0):
    """Parametric click-bootstrap conditional on counts.

    For each cell, resample clicks via Binomial(N, C/N) given the empirical
    click rate. Compute R_hat across iterations. Returns std + percentile CI
    of R_hat. Does NOT resample (q, ad) pairs --- counts and population are
    fixed, only click realizations vary. This isolates the click-noise
    variance that V_k(omega) bounds.
    """
    rng = np.random.default_rng(seed)
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    Ck0 = m["Ck"].to_numpy(dtype=float)
    Ckp0 = m["Ckp"].to_numpy(dtype=float)
    theta_k = np.clip(Ck0 / Nk, 0.0, 1.0)
    theta_kp = np.clip(Ckp0 / Nkp, 0.0, 1.0)
    omega = weights(scheme, Nk, Nkp)
    Nk_int = Nk.astype(np.int64)
    Nkp_int = Nkp.astype(np.int64)

    estimates = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, theta_k).astype(float)
        Ckp = rng.binomial(Nkp_int, theta_kp).astype(float)
        estimates[it] = ratio_estimator(omega, Nk, Ck, Nkp, Ckp)
    e = estimates[np.isfinite(estimates)]
    return {"mean": float(np.mean(e)),
            "std":  float(np.std(e, ddof=1)),
            "lo":   float(np.percentile(e, 2.5)),
            "hi":   float(np.percentile(e, 97.5)),
            "n_iter": int(len(e))}


def evaluate_pair(m: pd.DataFrame, alpha: float, beta: float,
                  n_iter: int, seed: int) -> list[dict]:
    Nk = m["Nk"].to_numpy(dtype=float)
    Nkp = m["Nkp"].to_numpy(dtype=float)
    rows = []
    for scheme in SCHEMES:
        omega = weights(scheme, Nk, Nkp)
        Vk = vk_term(omega, Nk)
        Vkp = vk_term(omega, Nkp)
        e = ess(omega)
        cb = click_bootstrap_std(m, scheme, n_iter=n_iter, seed=seed)
        rows.append({
            "scheme": scheme,
            "n_pairs": int(len(m)),
            "ESS": e,
            "ESS_frac": e / len(m),
            "Vk": Vk, "Vkp": Vkp,
            "symmetric_Vk_plus_Vkp": Vk + Vkp,
            "asymmetric_aVk_plus_bVkp": alpha * Vk + beta * Vkp,
            "click_boot_R": cb["mean"],
            "click_boot_std": cb["std"],
            "click_boot_lo": cb["lo"],
            "click_boot_hi": cb["hi"],
            "click_boot_n_iter": cb["n_iter"],
        })
    return rows


def report_table(name: str, k: int, kp: int, rows: list[dict]):
    print(f"\n=== {name}: pair ({k}, {kp})  n={rows[0]['n_pairs']:,} ===")
    df = pd.DataFrame(rows)
    cols_show = ["scheme", "ESS", "ESS_frac",
                 "symmetric_Vk_plus_Vkp", "asymmetric_aVk_plus_bVkp",
                 "click_boot_R", "click_boot_std"]
    print(df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    base = df[df["scheme"] == "uniform"].iloc[0]
    print("\n  Reduction vs uniform (%):")
    for col in ["symmetric_Vk_plus_Vkp", "asymmetric_aVk_plus_bVkp",
                "click_boot_std"]:
        for s in ["min", "harmonic"]:
            v = df[df["scheme"] == s].iloc[0][col]
            r = (base[col] - v) / base[col] * 100
            print(f"    {s:8s} {col:30s} {r:+6.1f}%")


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

def load_kdd_full() -> pd.DataFrame:
    print(f"Loading {KDD_PARQUET} ...")
    return pd.read_parquet(KDD_PARQUET)


def stream_aggregate_obd() -> pd.DataFrame:
    print(f"Streaming {OBD_INNER} from {OBD_ZIP} ...")
    parts = []
    with zipfile.ZipFile(OBD_ZIP) as zf:
        with zf.open(OBD_INNER) as raw:
            for chunk in pd.read_csv(raw,
                                     usecols=["item_id", "position", "click"],
                                     dtype={"item_id": np.int32,
                                            "position": np.int8,
                                            "click": np.int8},
                                     chunksize=2_000_000):
                agg = (chunk.groupby(["item_id", "position"], sort=False)
                            .agg(impressions=("click", "size"),
                                 clicks=("click", "sum"))
                            .reset_index())
                parts.append(agg)
                print(f"  partial agg: {len(agg):,} cells (chunk {len(chunk):,})")
    big = pd.concat(parts, ignore_index=True)
    final = (big.groupby(["item_id", "position"], sort=False)
                .agg(impressions=("impressions", "sum"),
                     clicks=("clicks", "sum"))
                .reset_index())
    print(f"  total cells: {len(final):,}, "
          f"impressions: {final['impressions'].sum():,}, "
          f"clicks: {final['clicks'].sum():,}, "
          f"CTR {final['clicks'].sum()/final['impressions'].sum():.4%}")
    return final


def adjacent_pair_table(agg: pd.DataFrame, k: int, kp: int,
                         join_keys: list[str]) -> pd.DataFrame:
    a = agg[agg["position"] == k][join_keys + ["impressions", "clicks"]] \
            .rename(columns={"impressions": "Nk", "clicks": "Ck"})
    b = agg[agg["position"] == kp][join_keys + ["impressions", "clicks"]] \
            .rename(columns={"impressions": "Nkp", "clicks": "Ckp"})
    return a.merge(b, on=join_keys, how="inner")


def run_dataset(name: str, agg: pd.DataFrame, join_keys: list[str],
                pairs: list[tuple[int, int]], n_iter: int) -> dict:
    summary = {"dataset": name,
               "n_cells": int(len(agg)),
               "n_impressions": int(agg["impressions"].sum()),
               "n_clicks": int(agg["clicks"].sum()),
               "ctr": float(agg["clicks"].sum() / agg["impressions"].sum()),
               "pairs": []}
    for k, kp in pairs:
        m = adjacent_pair_table(agg, k, kp, join_keys)
        if m.empty or len(m) < 2:
            continue
        alpha = (k - 1) if k > 1 else 0.5
        beta = float(kp - 1)
        t0 = time.time()
        rows = evaluate_pair(m, alpha=alpha, beta=beta,
                             n_iter=n_iter, seed=k * 100 + kp)
        elapsed = time.time() - t0
        report_table(name, k, kp, rows)
        print(f"  ({k},{kp}) eval done in {elapsed:.1f}s")
        summary["pairs"].append({"k": k, "kp": kp, "alpha": alpha, "beta": beta,
                                  "n_pairs": int(len(m)), "rows": rows})
    out = OUT / f"{name}_validation.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {out}")
    return summary


def main():
    n_iter_kdd = 200
    n_iter_obd = 500

    # ---- KDD Cup 2012 Track 2 (full) ----
    kdd_agg = load_kdd_full()
    run_dataset("kdd2012_full", kdd_agg,
                join_keys=["queryID", "adID"],
                pairs=[(1, 2), (2, 3)],
                n_iter=n_iter_kdd)

    # ---- Open Bandit (random/all) ----
    obd_agg = stream_aggregate_obd()
    run_dataset("openbandit_random_all", obd_agg,
                join_keys=["item_id"],
                pairs=[(1, 2), (2, 3)],
                n_iter=n_iter_obd)


if __name__ == "__main__":
    main()
