"""Ground-truth comparison: PBM-EM vs count-only schemes on synthetic data
where the true position-bias ratio is known.

We simulate the anchor-item PBM scenario from §5 of the paper:
- M=10 positions, p_k = k^{-eta}, eta=1 => true p_1/p_2 = 2.0
- Relevance r(q,d) drawn iid from a per-cell distribution.
- Anchor-item imbalance with traffic split s and ratio r.

For each weighting scheme, report empirical bias and SE of R_hat over
500 trials. PBM-EM is included alongside the count-only schemes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/alessandro/workspace/ultr-bias-toolkit/paper/reviewer_response")
N_TRIALS = 500
SEED = 12345

# True propensities
ETA = 1.0
P_TRUE = lambda k: k ** (-ETA)  # p_1=1, p_2=0.5, ...
TRUE_R = P_TRUE(1) / P_TRUE(2)  # = 2.0


def sim_pair(rng, n_anchor=400, n_balanced=100, r_max=50):
    """Simulate (q,d) cells at positions (1,2) under anchor-item PBM.

    Anchor items: heavy traffic at one position, light at the other.
    Balanced items: equal traffic at both.
    Per-cell relevance r in (0, 0.5) uniform.
    Clicks ~ Bernoulli(p_k * r) -> Binomial(N, p_k r) at aggregated level.
    """
    p1, p2 = P_TRUE(1), P_TRUE(2)
    rows = []
    # anchor items at position 1
    for _ in range(n_anchor // 2):
        r = rng.uniform(0.05, 0.5)
        N1 = rng.integers(50 * r_max, 200 * r_max)
        N2 = rng.integers(1, 51)
        C1 = rng.binomial(int(N1), float(np.clip(p1 * r, 0, 1)))
        C2 = rng.binomial(int(N2), float(np.clip(p2 * r, 0, 1)))
        rows.append((N1, C1, N2, C2))
    # anchor items at position 2
    for _ in range(n_anchor // 2):
        r = rng.uniform(0.05, 0.5)
        N1 = rng.integers(1, 51)
        N2 = rng.integers(50 * r_max, 200 * r_max)
        C1 = rng.binomial(int(N1), float(np.clip(p1 * r, 0, 1)))
        C2 = rng.binomial(int(N2), float(np.clip(p2 * r, 0, 1)))
        rows.append((N1, C1, N2, C2))
    # balanced items
    for _ in range(n_balanced):
        r = rng.uniform(0.05, 0.5)
        N1 = rng.integers(25, 75)
        N2 = rng.integers(25, 75)
        C1 = rng.binomial(int(N1), float(np.clip(p1 * r, 0, 1)))
        C2 = rng.binomial(int(N2), float(np.clip(p2 * r, 0, 1)))
        rows.append((N1, C1, N2, C2))
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 2], arr[:, 1], arr[:, 3]  # Nk, Nkp, Ck, Ckp


def weighted_ratio(omega, Nk, Nkp, Ck, Ckp):
    th_k = np.where(Nk > 0, Ck / np.maximum(Nk, 1), 0.0)
    th_kp = np.where(Nkp > 0, Ckp / np.maximum(Nkp, 1), 0.0)
    A = float(np.sum(omega * th_k))
    B = float(np.sum(omega * th_kp))
    return A / B if B > 0 else np.nan


def em_pbm(Nk, Nkp, Ck, Ckp, max_iter=200, tol=1e-9):
    p2 = 0.5
    for it in range(max_iter):
        denom = Nk + p2 * Nkp
        r = (Ck + Ckp) / np.maximum(denom, 1.0)
        num = float(np.sum(Ckp))
        den = float(np.sum(r * Nkp))
        new_p2 = num / max(den, 1e-30)
        if abs(new_p2 - p2) < tol:
            p2 = new_p2
            break
        p2 = new_p2
    return p2


def trial(rng):
    Nk, Nkp, Ck, Ckp = sim_pair(rng)
    out = {}
    out["uniform"] = weighted_ratio(np.ones_like(Nk), Nk, Nkp, Ck, Ckp)
    out["min"] = weighted_ratio(np.minimum(Nk, Nkp), Nk, Nkp, Ck, Ckp)
    out["harmonic"] = weighted_ratio(Nk * Nkp / (Nk + Nkp), Nk, Nkp, Ck, Ckp)
    p2 = em_pbm(Nk, Nkp, Ck, Ckp)
    out["pbm_em"] = 1.0 / p2 if p2 > 0 else np.nan
    return out


def main():
    rng = np.random.default_rng(SEED)
    results = {k: [] for k in ["uniform", "min", "harmonic", "pbm_em"]}
    print(f"True R = p_1/p_2 = {TRUE_R:.4f}")
    print(f"Running {N_TRIALS} trials...")
    for t in range(N_TRIALS):
        r = trial(rng)
        for k in results:
            results[k].append(r[k])
        if (t + 1) % 100 == 0:
            print(f"  trial {t+1}", flush=True)

    rows = []
    print(f"\n{'Scheme':<12s}  {'mean R':>10s}  {'bias':>10s}  {'SE':>10s}  {'RMSE':>10s}")
    print("-" * 60)
    for k, vals in results.items():
        v = np.array(vals)
        v = v[np.isfinite(v)]
        mean = float(v.mean())
        bias = mean - TRUE_R
        se = float(v.std(ddof=1))
        rmse = float(np.sqrt(np.mean((v - TRUE_R) ** 2)))
        rows.append({"scheme": k, "mean_R": mean, "bias": bias,
                     "SE": se, "RMSE": rmse, "true_R": TRUE_R})
        print(f"{k:<12s}  {mean:>10.4f}  {bias:>+10.4f}  {se:>10.4f}  {rmse:>10.4f}")

    pd.DataFrame(rows).to_csv(OUT / "synth_em_groundtruth.csv", index=False)
    print(f"\nWrote {OUT / 'synth_em_groundtruth.csv'}")


if __name__ == "__main__":
    main()
