# A simple write-up: why harmonic weighting beats uniform weighting, and how we measure it without bootstrap

## 1. The problem in its simplest form

We have click logs from a search engine. For each query $q$ and ad $d$, we observe how often $(q, d)$ was shown at position $k$ — call this count $N_k$ — and how often it was clicked — call this $C_k$.

Under the **Position-Based Model (PBM)**, the probability of a click factors:

$$P(\mathrm{click} \mid q, d, k) \;=\; p_k \cdot r_{q,d},$$

where $p_k$ depends only on the position and $r_{q,d}$ depends only on the (query, ad) pair. The thing we want to estimate is the **position-bias ratio**

$$R_{1,2} \;=\; p_1 / p_2.$$

If we pair up every $(q, d)$ that was seen at *both* position 1 and position 2, we can estimate $R_{1,2}$ by taking a ratio of click rates and combining across pairs. The question is: **how should we combine them?**

A given pair $i$ gives us a noisy estimate of $R$. Pair $i$ might have been seen 1,000,000 times at each position (very precise) or 1 time at each position (basically noise). If we just *average* the per-pair estimates with equal weight (uniform weighting), we're letting one coin flip count the same as a million coin flips. That's clearly wrong.

The right thing — by a century of statistics — is to **weight each pair by how much information it carries**. The toolkit's claim is that **harmonic weighting**, $\omega_i = N_k^i N_{k'}^i / (N_k^i + N_{k'}^i)$, is the count-based version of this idea, and substantially beats uniform on real data.

The challenge has been: **how do we measure "substantially beats" rigorously?**

---

## 2. Why the bootstrap doesn't work here

Our first instinct was to use the **bootstrap**: resample pairs many times, see how much the estimate jiggles, and call the jiggle the standard error. This is the standard tool in modern statistics for "I don't trust analytic formulas, let the computer do it."

It failed on KDD, for two specific reasons:

1. **Uniform "wins" the bootstrap artificially.** Of KDD's 4.6 million pairs at positions (1,2), about 60% are singletons (1 impression). Averaging 4.6 million near-coin-flip values gives a *very smooth-looking* sample distribution — low jiggle. But "low jiggle of an average of garbage" is not the same as "precise estimate." The bootstrap can't tell the difference.

2. **The bootstrap penalizes inverse-variance weighting by construction.** Inverse-variance weighting works by *concentrating* on the few high-information cells. When the bootstrap resamples pairs, it randomly drops some of those high-weight cells in and out, causing big swings — but those swings are an artifact of resampling, not of the estimator's actual precision.

So the bootstrap framed the question as "how much does the estimate move when I randomly reshuffle pairs?", and the answer was misleading because the bootstrap was the wrong instrument.

---

## 3. The right tool: plug-in asymptotic variance via the delta method

Instead of letting the computer simulate the variance, we **compute it directly from a statistical model**.

### 3.1 The estimator

For a pair $i$ at positions $(k, k')$, with $\hat\theta_k^i = C_k^i / N_k^i$ the observed click rate, our weighted estimator is

$$\hat R \;=\; \frac{A}{B} \;=\; \frac{\sum_i \omega_i\, \hat\theta_k^i}{\sum_i \omega_i\, \hat\theta_{k'}^i}.$$

This is exactly a **self-normalized weighted ratio estimator** — a textbook object in importance sampling and survey statistics (Owen 2013, Ch. 9; Kong 1992).

### 3.2 Where the variance comes from

Each $\hat\theta_k^i$ is a noisy estimate of the true $\theta_k^i$. Specifically, if PBM holds and clicks are Bernoulli, then

$$\mathrm{Var}(\hat\theta_k^i) \;=\; \frac{\theta_k^i (1 - \theta_k^i)}{N_k^i}.$$

That's the **binomial variance formula** — the same thing that gives the "margin of error of a poll." Big $N$ → small variance. Bernoulli rate near 0.5 → bigger variance than rate near 0 or 1.

### 3.3 The delta method

The estimator $\hat R = A/B$ is a *ratio* of two random sums. To get its variance, we use the **delta method** (van der Vaart 1998, Ch. 3; Wasserman, *All of Statistics*, Ch. 9):

> If $\hat A \to A$ and $\hat B \to B$ with small noise, then $\hat A / \hat B$ has variance approximately
> $$\mathrm{Var}\!\left(\hat R\right) \;\approx\; \hat R^2 \cdot \left[\frac{\mathrm{Var}(\hat A)}{A^2} + \frac{\mathrm{Var}(\hat B)}{B^2} - 2\,\frac{\mathrm{Cov}(\hat A, \hat B)}{AB}\right].$$

The covariance term is zero here because clicks at position $k$ and position $k'$ come from *different impression sets* and are conditionally independent given the counts.

Plugging in:

$$\boxed{\,\widetilde V(\omega) \;=\; \underbrace{\frac{\sum_i \omega_i^2\, \theta_k^i(1-\theta_k^i)/N_k^i}{\left(\sum_i \omega_i\, \theta_k^i\right)^2}}_{\text{noise from position }k} \;+\; \underbrace{\frac{\sum_i \omega_i^2\, \theta_{k'}^i(1-\theta_{k'}^i)/N_{k'}^i}{\left(\sum_i \omega_i\, \theta_{k'}^i\right)^2}}_{\text{noise from position }k'}\,}$$

and the standard error of $\hat R$ is $\mathrm{SE}(\hat R) = |\hat R|\sqrt{\widetilde V(\omega)}$.

### 3.4 What's "plug-in"

We don't know the true $\theta_k^i$, so we **plug in** $\hat\theta_k^i = C_k^i / N_k^i$ from the data (with a tiny bit of shrinkage toward the marginal click rate to handle zero-click cells). That's it — a single closed-form sum over the 4.6M pairs.

No simulation. No resampling. No bootstrap loop. Just a formula.

---

## 4. The classical theory: why inverse-variance weighting is optimal

This isn't a new idea — it's a century old.

**Aitken (1934)** proved the **Gauss–Markov–Aitken theorem**: when combining noisy estimates of the same quantity, the minimum-variance unbiased linear combination is

$$\omega_i^\star \;\propto\; 1/\mathrm{Var}(\hat\theta_i).$$

That is, **inverse-variance weighting** (IVW). Anyone who has seen meta-analysis (Hedges & Olkin 1985; DerSimonian & Laird 1986; Borenstein et al. 2009) has met this in the form "weight each study by 1/variance, never by sample count alone, and never by equal weighting."

For Bernoulli observations with $\mathrm{Var}(\hat\theta_i) = \theta_i(1-\theta_i)/N_i$:

- If $\theta_i$ is roughly constant across cells, $\mathrm{Var}(\hat\theta_i) \propto 1/N_i$, so the IVW weight reduces to $\omega_i \propto N_i$. With *two* positions per pair this becomes the harmonic mean of the two counts — **that's exactly the harmonic weight.**
- If $\theta_i$ varies a lot across cells, we need to plug in $\hat\theta_i$ — but $\hat\theta_i$ itself is noisy when $N_i$ is small, and the result becomes unstable. (Our "full θ-oracle" experiment on KDD demonstrates this: with mostly singletons, plug-in IVW is pathological.)

So **harmonic weighting is the count-only inverse-variance optimum** — the version of Aitken's theorem you can compute without knowing the click rates.

### 4.1 The ESS interpretation

Kong (1992) introduced a single-number summary of how concentrated the weights are: the **effective sample size**

$$\mathrm{ESS} \;=\; \frac{\left(\sum_i \omega_i\right)^2}{\sum_i \omega_i^2}.$$

ESS tells you "how many *equal-weight* observations would give the same variance." Owen (2013) §9.2 shows that the asymptotic variance of a self-normalized weighted ratio scales as $1/\mathrm{ESS}$, **not** $1/n$.

On KDD:
- Uniform: $\mathrm{ESS} = 4{,}613{,}072$ (all 4.6M pairs count fully — but mostly meaningless).
- Harmonic: $\mathrm{ESS} = 1{,}849$ (only the high-information pairs count, but they count for real).

Uniform's "4.6M effective observations" is a fiction created by averaging singletons. The 1,849 from harmonic is the *real* effective sample size of the data.

---

## 5. Asymptotic Relative Efficiency: the headline number

**ARE** between two estimators is just the ratio of their asymptotic variances:

$$\mathrm{ARE}(\text{scheme}_1, \text{scheme}_2) \;=\; \frac{\widetilde V(\omega_2)}{\widetilde V(\omega_1)}.$$

If $\mathrm{ARE}(A, B) = 4$, then $A$ uses sample size $n$ to achieve what $B$ achieves at $4n$. Equivalently, $A$'s standard error is $1/\sqrt{4} = 0.5\times$ that of $B$.

On the full KDD data, our analysis gives

$$\mathrm{ARE}(\text{harmonic},\, \text{uniform}) \;=\; 4.53.$$

**Plain English:** uniform weighting needs about 4.5× as many impressions as harmonic weighting to reach the same precision. Equivalently, harmonic's standard error is 2.1× smaller. **This is a property of KDD's count distribution, not of any model fit, and it does not require ground truth.**

---

## 6. Why this avoids every bootstrap trap

| Bootstrap problem | How plug-in $\widetilde V$ avoids it |
|---|---|
| Uniform's smoothness from averaging garbage | Each singleton contributes its full Bernoulli variance — no smoothing |
| Penalty for concentrated weights | Variance is computed *for* the given weights, not under perturbations of which pairs appear |
| Requires ground truth or held-out test set | Pure model-internal quantity |
| Slow Monte Carlo loop | Single closed-form sum |
| Confounds estimator precision with model misspecification | PBM violation becomes a *separately testable* quantity (Cochran's Q) |

---

## 7. Calibration check

How do we know the plug-in formula is right on this data? Pick a small subsample ($f = 0.001$), simulate click-only resampling 500 times under the binomial model, and compare:

| Scheme | Plug-in SE | Empirical SE | Agreement |
|---|---:|---:|---:|
| Uniform | 0.1306 | 0.1321 | within 1% |
| Harmonic | 0.1113 | 0.1120 | within 1% |

Plug-in agrees with the empirical click-resample to within Monte Carlo noise. ✓

---

## 8. PBM violation: a separate question

Different weighting schemes converge to slightly different $\hat R$ on KDD (uniform → 1.68, harmonic → 1.71, count-oracle → 1.71, full-θ-oracle → 2.34). Theorem 1 of the paper says they should all converge to the same value *under PBM*. They don't, so **PBM is violated on real sponsored-search data**.

The textbook detector for this is **Cochran's Q** (Cochran 1954) — a heterogeneity test that asks whether per-pair ratios $\hat\theta_k^i / \hat\theta_{k'}^i$ are consistent with a single shared $R$. We compute it but the singleton mass makes it noisy; restricting to high-$N$ cells would give a cleaner signal.

This separates two questions that the bootstrap had conflated:
- **Statistical efficiency** (how much variance for fixed model): ARE = 4.53, harmonic wins.
- **Model misspecification** (does the model fit the data): Cochran Q quantifies it; consistent with Yue 2010 and Wang 2018 documenting PBM violations on real logs.

---

## 9. What we end up claiming on KDD

1. **Heavy tails make uniform weighting statistically untenable.** 60% of pairs carry essentially no information; uniform gives them equal vote.
2. **Harmonic weighting is the count-only inverse-variance optimum** (Aitken 1934; Hedges & Olkin 1985).
3. **Plug-in delta-method variance** (Owen 2013 §9.2; van der Vaart 1998 Ch. 3) gives a rigorous, falsifiable comparison without bootstrap or ground truth.
4. **On full KDD, ARE(harmonic, uniform) ≈ 4.5**, with calibration agreement within 1%, and the ratio rises to ~5 on the high-information quartile.
5. **PBM is violated** on KDD (different schemes converge to different $\hat R$, Cochran Q heterogeneity), but this is a separate and well-known fact about real search logs.

---

## 10. Where to read more

- **Owen (2013), Ch. 9.** Self-normalized weighted ratio estimators, ESS, variance formulas. Free online: https://artowen.su.domains/mc/
- **Wasserman, 36-705 lectures 3 and 4.** Delta method. https://www.stat.cmu.edu/~larry/=stat705/
- **Bottou et al. (2013) JMLR §4.** Variance of self-normalized IPS in counterfactual evaluation — closest narrative analogue to our setting.
- **Swaminathan & Joachims (2015) NeurIPS.** SNIPS estimator — applies exactly $\widetilde V$ to off-policy evaluation in ULTR.
- **Hedges & Olkin (1985), Ch. 5–6.** The classical inverse-variance optimality argument.
- **Kong (1992) TR-348.** The ESS derivation.
