"""Plug-in delta-method asymptotic variance for harmonic-vs-uniform on KDD.

Replaces the bootstrap framing on KDD with the classical plug-in
asymptotic-variance estimator for a self-normalized weighted ratio
estimator.

References (full list — these anchor each part of the analysis):

Self-normalized weighted ratio estimator + ESS-based variance:
  - Owen, A. B. (2013). Monte Carlo theory, methods and examples, Ch. 9
    (Importance sampling), Sections 9.2, 9.4. Primary reference for
    Vtilde(omega) and ESS = (sum w)^2 / sum w^2.
  - Kong, A. (1992). A note on importance sampling using standardized
    weights. Tech. Report 348, U. Chicago Statistics. ESS derivation.
  - Hesterberg, T. (1995). Weighted average importance sampling and
    defensive mixture distributions. Technometrics 37(2), 185-194.

Delta method:
  - van der Vaart, A. W. (1998). Asymptotic Statistics, Ch. 3.
  - Wasserman, L. All of Statistics, Ch. 9; 36-705 lecture notes 3-4.

Inverse-variance weighting / BLUE / meta-analysis:
  - Aitken, A. C. (1934). On least squares and linear combination of
    observations. Proc. Royal Soc. Edinburgh 55, 42-48. (BLUE under
    heteroscedasticity.)
  - Cochran, W. G. (1937). Problems arising in the analysis of a series
    of similar experiments. JRSS Suppl. 4, 102-118.
  - Cochran, W. G. (1954). The combination of estimates from different
    experiments. Biometrics 10, 101-129. (Cochran's Q heterogeneity
    test, used in Section 3 below.)
  - Hedges, L. V., & Olkin, I. (1985). Statistical Methods for
    Meta-Analysis. Academic Press, Ch. 5-6, §6.2.
  - DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical
    trials. Controlled Clinical Trials 7(3), 177-188.
  - Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R.
    (2009). Introduction to Meta-Analysis, Ch. 9.

Survey sampling / design effect:
  - Cochran, W. G. (1977). Sampling Techniques (3rd ed.), Ch. 5, 11.
  - Lohr, S. L. (2010). Sampling: Design and Analysis (2nd ed.), Ch. 7.

Off-policy / counterfactual evaluation (closest in spirit):
  - Bottou, L., Peters, J., Quinonero-Candela, J., et al. (2013).
    Counterfactual reasoning and learning systems. JMLR 14, 3207-3260,
    §4 (variance of self-normalized IPS).
  - Dudik, M., Langford, J., & Li, L. (2011). Doubly robust policy
    evaluation and learning. ICML.
  - Swaminathan, A., & Joachims, T. (2015). The self-normalized
    estimator for counterfactual learning. NeurIPS. (Exact Vtilde
    framework for SNIPS.)

Click models / ULTR / PBM violations on real logs:
  - Craswell, N., Zoeter, O., Taylor, M., & Ramsey, B. (2008). An
    experimental comparison of click position-bias models. WSDM.
  - Yue, Y., Patel, R., & Roehrig, H. (2010). Beyond position bias:
    examining result attractiveness as a source of presentation bias
    in clickthrough data. WWW. (Documents PBM violations.)
  - Chuklin, A., Markov, I., & de Rijke, M. (2015). Click Models for
    Web Search. Morgan & Claypool.
  - Joachims, T., Swaminathan, A., & Schnabel, T. (2017). Unbiased
    learning-to-rank with biased feedback. WSDM.
  - Wang, X., Bendersky, M., Metzler, D., & Najork, M. (2018).
    Position bias estimation for unbiased learning to rank in personal
    search. WSDM. (Section 6 documents PBM violation on Gmail/Drive.)
  - Agarwal, A., Wang, X., Li, C., Bendersky, M., & Najork, M. (2019).
    Estimating position bias without intrusive interventions. WSDM.
    (Intervention-harvesting estimator we extend.)

For the pair (k, k') and cells i = 1..n with counts (N_k^i, N_{k'}^i)
and click rates theta_k^i = C_k^i / N_k^i, the estimator is

    R_hat = A / B,
    A = sum_i omega_i * theta_k^i,
    B = sum_i omega_i * theta_{k'}^i.

Under conditional independence of clicks given (N_k^i, N_{k'}^i), the
delta-method asymptotic variance of log(R_hat) is

    Vtilde(omega) = sum_i omega_i^2 * theta_k^i (1-theta_k^i) / N_k^i / A^2
                  + sum_i omega_i^2 * theta_{k'}^i (1-theta_{k'}^i) / N_{k'}^i / B^2

The standard error of R_hat itself is |R_hat| * sqrt(Vtilde).

We compute Vtilde for each scheme on KDD at f in {0.001, 0.01, 0.1, 1.0},
report the **Asymptotic Relative Efficiency** ARE(scheme, uniform) =
Vtilde_uniform / Vtilde_scheme as the headline, plus Cochran's Q for
PBM-heterogeneity and a stratified ARE by impression-count quartile.

We also do a calibration check: empirical std under click-only resampling
should match the plug-in SE within Monte Carlo error.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "reviewer_response"
KDD_PARQUET = Path("/home/alessandro/workspace/ultr-bias-toolkit/data/kdd2012_full/agg_qad_pos.parquet")

FRACTIONS = [0.001, 0.01, 0.1, 1.0]
PAIR = (1, 2)
N_CALIB_BOOT = 500  # for calibration check only
SHRINK = 1.0  # additive shrinkage strength on counts toward marginal theta_bar


def weight(scheme, Nk, Nkp, th_k=None, th_kp=None):
    if scheme == "uniform":
        return np.ones_like(Nk, dtype=float)
    if scheme == "min":
        return np.minimum(Nk, Nkp).astype(float)
    if scheme == "harmonic":
        return Nk * Nkp / (Nk + Nkp)
    if scheme == "count_oracle_a0.5_b1":
        a, b = 0.5, 1.0
        return Nk * Nkp / (a * Nkp + b * Nk)
    if scheme == "full_theta_oracle":
        a, b = 0.5, 1.0
        s = (a * th_k * (1.0 - th_k) / Nk
             + b * th_kp * (1.0 - th_kp) / Nkp)
        return 1.0 / np.maximum(s, 1e-30)
    raise ValueError(scheme)


SCHEMES = ["uniform", "min", "harmonic", "count_oracle_a0.5_b1", "full_theta_oracle"]


def shrunk_theta(C, N, marginal):
    """Beta(marginal*SHRINK, (1-marginal)*SHRINK) posterior mean."""
    a = marginal * SHRINK
    b = (1.0 - marginal) * SHRINK
    return (C + a) / (N + a + b)


def plugin_variance(omega, Nk, Nkp, th_k, th_kp):
    """Delta-method asymptotic variance of R_hat = A/B."""
    A = float(np.sum(omega * th_k))
    B = float(np.sum(omega * th_kp))
    varA = float(np.sum(omega**2 * th_k * (1.0 - th_k) / Nk))
    varB = float(np.sum(omega**2 * th_kp * (1.0 - th_kp) / Nkp))
    Vtilde = varA / A**2 + varB / B**2
    R = A / B if B > 0 else np.nan
    SE = abs(R) * np.sqrt(max(Vtilde, 0.0))
    return R, Vtilde, SE, A, B


def cochran_Q_high_n(Nk, Nkp, th_k, th_kp, R_hat, min_clicks=5):
    """Cochran's Q restricted to cells with at least `min_clicks` at each position.
    Drops singleton noise; tests whether high-information cells share one R."""
    mask = ((th_k * Nk) >= min_clicks) & ((th_kp * Nkp) >= min_clicks) \
           & (th_k < 1) & (th_kp < 1)
    if mask.sum() < 2:
        return float("nan"), 0
    th_k_, th_kp_, Nk_, Nkp_ = th_k[mask], th_kp[mask], Nk[mask], Nkp[mask]
    var_log = (1.0 - th_k_) / (Nk_ * th_k_) + (1.0 - th_kp_) / (Nkp_ * th_kp_)
    w = 1.0 / np.maximum(var_log, 1e-30)
    log_rho = np.log(th_k_) - np.log(th_kp_)
    # IVW-pooled log R from the high-N cells themselves
    log_R_pool = float(np.sum(w * log_rho) / np.sum(w))
    Q = float(np.sum(w * (log_rho - log_R_pool) ** 2))
    return Q, int(mask.sum())


def cochran_Q(omega, Nk, Nkp, th_k, th_kp, R_hat):
    """Heterogeneity test on per-cell pseudo-ratios under PBM null.

    Per-cell estimate rho_i = theta_k^i / theta_{k'}^i. Variance of log(rho_i)
    via delta method: (1-th_k)/(N_k th_k) + (1-th_kp)/(N_kp th_kp).
    Use IVW weights w_i = 1/var(log rho_i); Q = sum w_i (log rho_i - log R)^2.
    Cells with zero clicks at either position contribute nothing reliable;
    we drop them (they have infinite variance).
    """
    mask = (th_k > 0) & (th_k < 1) & (th_kp > 0) & (th_kp < 1)
    th_k_, th_kp_, Nk_, Nkp_ = th_k[mask], th_kp[mask], Nk[mask], Nkp[mask]
    if len(th_k_) < 2:
        return float("nan"), 0
    var_log = (1.0 - th_k_) / (Nk_ * th_k_) + (1.0 - th_kp_) / (Nkp_ * th_kp_)
    w = 1.0 / np.maximum(var_log, 1e-30)
    log_rho = np.log(th_k_) - np.log(th_kp_)
    log_R = np.log(R_hat)
    Q = float(np.sum(w * (log_rho - log_R) ** 2))
    return Q, int(mask.sum())


def calibration_click_resample(omega, Nk, Nkp, th_k, th_kp, n_iter, seed):
    """Empirical std of R under click-only resampling, for calibration."""
    rng = np.random.default_rng(seed)
    Nk_int = Nk.astype(np.int64)
    Nkp_int = Nkp.astype(np.int64)
    est = np.empty(n_iter)
    for it in range(n_iter):
        Ck = rng.binomial(Nk_int, np.clip(th_k, 0, 1)).astype(float)
        Ckp = rng.binomial(Nkp_int, np.clip(th_kp, 0, 1)).astype(float)
        A = float(np.sum(omega * Ck / Nk))
        B = float(np.sum(omega * Ckp / Nkp))
        est[it] = A / B if B > 0 else np.nan
    return float(np.std(est[np.isfinite(est)], ddof=1))


def subsample(agg, f, rng):
    if f >= 1.0:
        return agg
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


def stratified_are(omega_u, omega_h, Nk, Nkp, th_k, th_kp):
    """ARE(uniform, harmonic) within total-impression quartiles."""
    Ntot = Nk + Nkp
    qs = np.quantile(Ntot, [0.25, 0.5, 0.75])
    bins = np.digitize(Ntot, qs)  # 0..3
    rows = []
    for q in range(4):
        mask = bins == q
        if mask.sum() < 2:
            continue
        _, Vu, _, _, _ = plugin_variance(omega_u[mask], Nk[mask], Nkp[mask],
                                         th_k[mask], th_kp[mask])
        _, Vh, _, _, _ = plugin_variance(omega_h[mask], Nk[mask], Nkp[mask],
                                         th_k[mask], th_kp[mask])
        rows.append({"quartile": q, "n": int(mask.sum()),
                     "V_uniform": Vu, "V_harmonic": Vh,
                     "ARE_h_over_u": Vu / Vh if Vh > 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    print(f"Loading {KDD_PARQUET}", flush=True)
    agg = pd.read_parquet(KDD_PARQUET)
    rows = []
    strat_rows = []
    k, kp = PAIR

    for f in FRACTIONS:
        seed = int(1e6 * f) + 7
        rng = np.random.default_rng(seed)
        t0 = time.time()
        sub = subsample(agg, f, rng)
        m = build_pair(sub, k, kp)
        n_imp = int(sub["impressions"].sum())
        if len(m) < 2:
            continue

        Nk = m["Nk"].to_numpy(dtype=float)
        Nkp = m["Nkp"].to_numpy(dtype=float)
        Ck = m["Ck"].to_numpy(dtype=float)
        Ckp = m["Ckp"].to_numpy(dtype=float)

        # marginal theta for shrinkage
        bar_k = float(Ck.sum() / Nk.sum())
        bar_kp = float(Ckp.sum() / Nkp.sum())
        th_k = shrunk_theta(Ck, Nk, bar_k)
        th_kp = shrunk_theta(Ckp, Nkp, bar_kp)

        print(f"\nf={f}: pairs={len(m):,}, imps={n_imp:,}, "
              f"bar_theta=({bar_k:.4f},{bar_kp:.4f})", flush=True)

        # baseline uniform for ARE denominator
        omega_uni = weight("uniform", Nk, Nkp)
        _, V_uniform, _, _, _ = plugin_variance(omega_uni, Nk, Nkp, th_k, th_kp)
        omega_harm = weight("harmonic", Nk, Nkp)

        for scheme in SCHEMES:
            omega = weight(scheme, Nk, Nkp, th_k=th_k, th_kp=th_kp)
            R, Vtilde, SE, A, B = plugin_variance(omega, Nk, Nkp, th_k, th_kp)
            ess = float((omega.sum())**2 / (omega**2).sum())
            ARE_vs_uniform = V_uniform / Vtilde if Vtilde > 0 else np.nan
            Q, n_eff = cochran_Q(omega, Nk, Nkp, th_k, th_kp, R)
            Q_hi, n_hi = cochran_Q_high_n(Nk, Nkp, th_k, th_kp, R, min_clicks=5)

            # calibration only at smaller f to keep runtime bounded
            do_calib = (f <= 0.01) and (scheme in ("uniform", "harmonic"))
            calib_std = (calibration_click_resample(omega, Nk, Nkp, th_k, th_kp,
                                                    N_CALIB_BOOT, seed + 11)
                         if do_calib else float("nan"))

            rows.append({
                "f": f, "n_imp": n_imp, "n_pairs": int(len(m)),
                "scheme": scheme, "R_hat": R, "Vtilde": Vtilde, "SE": SE,
                "ESS": ess, "ARE_vs_uniform": ARE_vs_uniform, "n_hi": n_hi,
                "cochran_Q": Q, "Q_df": n_eff - 1 if n_eff > 0 else 0,
                "cochran_Q_hi": Q_hi, "Q_hi_df": n_hi - 1 if n_hi > 0 else 0,
                "calib_empirical_std": calib_std,
            })
            print(f"  {scheme:24s}  R={R:.4f}  SE={SE:.4e}  "
                  f"V={Vtilde:.3e}  ARE_vs_U={ARE_vs_uniform:7.2f}  "
                  f"ESS={ess:.3e}  Q={Q:.2e}  Q_hi(n={n_hi})={Q_hi:.2e}  calib={calib_std}",
                  flush=True)

        # stratified ARE (uniform vs harmonic) only at full data
        if f >= 1.0:
            sdf = stratified_are(omega_uni, omega_harm, Nk, Nkp, th_k, th_kp)
            sdf["f"] = f
            strat_rows.append(sdf)
            print("\n  Stratified ARE(harmonic vs uniform) by N_tot quartile:")
            print(sdf.to_string(index=False))

        print(f"  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kdd_plugin_variance.csv", index=False)

    if strat_rows:
        pd.concat(strat_rows, ignore_index=True).to_csv(
            OUT / "kdd_plugin_stratified_are.csv", index=False)

    print("\n--- Headline table (full data) ---")
    full = df[df["f"] >= 1.0][["scheme", "R_hat", "SE", "Vtilde",
                                "ARE_vs_uniform", "ESS"]]
    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
