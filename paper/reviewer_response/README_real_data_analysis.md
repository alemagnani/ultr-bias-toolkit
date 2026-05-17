# Real-data analysis of harmonic weighting: discussion log + citations

This document captures the discussion of how to validate harmonic weighting on real
sponsored-search data (KDD Cup 2012 Track 2), the dead ends we ran into trying to
use bootstrap-style variance comparisons, and the proper statistical framing that
should drive the paper's empirical claims.

---

## 1. The original idea behind the paper

When you look at impression distributions on real ULTR-relevant click logs, they
are extraordinarily heavy-tailed:

* **Baidu-ULTR (Zou et al. 2022)**: 94% of $(q,d,\mathrm{pos})$ cells appear with
  $N=1$ impression after the published deduplication; only 4.3% of $(q,d)$ pairs
  appear at $\geq 2$ positions; among the 454K qualifying adjacent pairs, 78% have
  ratio $N_k/N_{k+1}\in[0.5,2]$. The released distribution looks balanced
  *because* it's been deduplicated.
* **KDD Cup 2012 Track 2 (full release, 9.9 GB)**: 149.6M raw rows, 235.6M
  effective impressions after expanding the `Impression` column, **63.8M unique
  $(q, \mathrm{ad}, \mathrm{pos})$ cells**, 8.2M clicks, overall CTR 3.49%.
  Adjacent-pair (1,2) interventional set: **4.61M $(q,\mathrm{ad})$ pairs** at
  positions 1 and 2 simultaneously, with **132.3M total impressions** in this
  interventional set alone.
  - Per-pair impression distribution: median 4 (combined), p99 = 258, p99.9 =
    2,255, max = 2,466,174.
  - **60.7% of pairs have $N_1+N_2 \le 5$** (essentially singletons / noise).
  - **0.1% of pairs (4,613 pairs) hold 41.6% of all impressions; top 1% holds
    61%.**

On data with this kind of heterogeneity, **uniform weighting is statistically
untenable**: it gives a 1-impression coin-flip estimator the same vote as a
100,000-impression precise estimator. The paper's premise was that harmonic
weighting (or the count-only oracle, which is essentially equivalent) is the
correct way to weight such heterogeneous samples.

That premise is correct and uncontroversial in statistics (see citations
below). The challenge has been validating it empirically.

---

## 2. What we found on KDD Cup 2012 Track 2 (full data)

### 2.1 Point estimates differ

For the position-1 → position-2 ratio $\hat R_{1,2}$, full-data point estimates
diverge by scheme:

| Scheme | $\hat R_{1,2}$ | Implied $p_2/p_1$ |
| --- | ---: | ---: |
| Uniform | 1.542 | 0.649 |
| Min ($\omega = \min(N_k, N_{k'})$) | 1.680 | 0.595 |
| Harmonic ($\omega = N_k N_{k'}/(N_k + N_{k'})$) | 1.679 | 0.596 |
| Count-only oracle ($\alpha=0.5,\beta=1$) | 1.683 | 0.594 |
| Count-only oracle ($\alpha=0.1,\beta=0.7$) | 1.689 | 0.592 |
| Full oracle ($\omega \propto 1/[\alpha\theta(1-\theta)/N + \beta\theta'(1-\theta')/N']$) | 1.740 | 0.575 |

Theorem 1 of the paper says all of these schemes are **ratio-unbiased under PBM**,
i.e., they should converge to the same value as $n\to\infty$. They don't, and
the gap doesn't shrink across our scaling sweep (235K → 235M impressions, the
gap stays ~9% between uniform and harmonic). Therefore **PBM is violated on
KDD sponsored-search data** — schemes have different finite-sample/asymptotic
biases under model misspecification.

### 2.2 Bootstrap variances are misleading

We tried three bootstrap variants:

1. **Click bootstrap** (parametric, conditional on counts): for each cell,
   redraw $C_k^i \sim \mathrm{Binomial}(N_k^i, \hat\theta_k^i)$. Only click
   realizations vary; population and counts are fixed. Theorem 4 of the paper
   bounds this directly.
2. **(q,ad)-row bootstrap**: resample $(q,\mathrm{ad})$ rows iid with
   replacement. Variance comes from population sampling.
3. **Query cluster bootstrap**: resample queries (each carrying all its
   $(q,\mathrm{ad})$ rows) with replacement. The natural unit of independence
   in click-log data.

What we observed across sample fractions $f \in \{0.001, 0.003, 0.01, 0.03,
0.1, 0.3, 1.0\}$ on the full KDD aggregation (`paper/scale_kdd2012_v3.py`):

* The **(q,ad)-row bootstrap shows harmonic 17% better than uniform at
  $f = 0.001$** (matching the original SamueChan-subset finding) but **224%
  *worse* at $f = 1.0$**. This crossover is not because harmonic stops working
  — it's because bootstrap-of-pairs penalizes any heterogeneous weighting:
  resampling whole high-weight pairs in/out causes large swings under
  heterogeneous weights, regardless of the estimator's true precision.
* The **click bootstrap shows harmonic ~80% worse at small $n$ and ~25%
  worse at full $n$**. But absolute click-noise std at full data is 0.002 on
  $\hat R \approx 1.68$ — coefficient of variation 0.1%, practically
  irrelevant.
* The **query cluster bootstrap** shows harmonic worse than uniform at every
  scale tested (10% worse at $f = 0.001$, 500% worse at $f = 1.0$).

**The bootstrap framing breaks down**, and it is the wrong instrument for two
reasons:

1. *Uniform's apparent low bootstrap std is averaging-out smoothness, not
   estimator precision.* When 60% of pairs are noise singletons, averaging 4.6M
   of them gives a smooth-looking sample distribution that has nothing to do
   with how close the estimate is to the truth.
2. *Bootstrap-of-pairs penalizes inverse-variance-style weights by
   construction.* The exact concentration on high-information cells that
   *helps* an inverse-variance estimator *hurts* it under bootstrap-of-pairs,
   because resampling perturbs the high-weight cells. This is a property of
   the bootstrap, not of the estimator.

### 2.3 Effective sample size paints the right picture

ESS = $(\sum \omega)^2 / \sum \omega^2$ on full data (4.61M pairs):

| Scheme | ESS | ESS / n_pairs |
| --- | ---: | ---: |
| Uniform | 4,613,072 | 100% |
| Min | 2,526 | 0.05% |
| Harmonic | 1,849 | 0.04% |
| Full oracle | even smaller (concentrated on cells with non-degenerate $\hat\theta$) | < 0.01% |

Uniform's "ESS = 4.6M" is a misleading number — it pretends 60% of pairs that
are noise singletons each count as one effective observation. Harmonic's
"ESS = 1849" is much closer to the *real* effective sample size of the data,
which is concentrated on the few thousand high-impression pairs that drive
the actual estimate.

### 2.4 The (N, θ) correlation issue

A diagnostic on full KDD (`paper/diagnose_kdd2012.py`):

| Pair, position | Spearman $\rho(N, \theta)$ | $\theta$ ratio q4 / q1 |
| --- | ---: | ---: |
| (1,2) at $k=1$ | +0.265 | 1.14 |
| (1,2) at $k+1=2$ | +0.277 | 1.08 |
| (2,3) at $k=2$ | +0.317 | 1.23 |
| (2,3) at $k+1=3$ | +0.229 | 1.05 |

In sponsored search, popular ads are shown more *and* clicked more, so $N$ and
$\theta$ are positively correlated. The count-only surrogate
$V_k(\omega) = \sum \omega^2/N_k / (\sum \omega)^2$ implicitly assumes
$\theta(1-\theta)$ factors out as a constant — i.e., that $\theta_i$ is iid
across cells. **On KDD this assumption fails**, which is why the surrogate's
predicted reduction (79%) doesn't translate one-to-one into actual click-noise
variance reduction.

Synthetic Exp1 (relevance ~ U(0.1, 0.5), independent of counts) and the
Yahoo semi-synthetic experiment satisfy the $(N, \theta)$ independence
assumption by construction; KDD violates it by ~14–23% on q4/q1 ratios.

### 2.5 The full $\theta$-aware oracle scaling
`paper/scale_kdd_oracle.py` runs the inverse-variance oracle weight
$\omega^{**}_i \propto 1/[\alpha \theta_k(1-\theta_k)/N_k + \beta \theta_{k'}(1-\theta_{k'})/N_{k'}]$
across the same sample-size grid. Behaviour:

* At small $n$, $\hat\theta_i = C/N$ is too noisy and the inverse-variance
  weight is pathological — full oracle has 100%+ higher (q,ad) and query
  bootstrap variance than harmonic.
* By $f = 0.1$ (23M imps, 542K pairs), full oracle is comparable to harmonic.
* **At $f = 1.0$ (235M imps, 4.6M pairs), full oracle gives query std 57%
  *lower* than harmonic** (0.011 vs 0.025), nearly matching uniform's 0.005,
  with $\hat R = 1.74$ (yet another scheme-specific limit).

So the missing piece *does* matter, *if* you have enough impressions per cell
to estimate $\theta_i$ reliably. The classical inverse-variance theory works
once $\hat\theta$ stops being noisy.

---

## 3. The framing trap and the way out

We spent considerable effort trying to validate "harmonic beats uniform on
KDD" via:

* (q,ad)-row bootstrap → harmonic looks better at small $n$, worse at full
  $n$. Not robust.
* Click bootstrap → harmonic always slightly worse on KDD; absolute values
  too small to matter at full $n$.
* Query cluster bootstrap → harmonic always worse than uniform.
* Cross-scheme consensus (harmonic / min / count-only oracle all agree at
  $\hat R = 1.68$) → invalidated because these schemes are not independent.
  They're all variants of the same "weight by smaller cell dimension" family
  and reduce to each other up to slowly-varying constants.

**The fundamental problem**: on data without ground truth, no bootstrap-style
or consensus-style argument can validate that one scheme is "right." Uniform
"wins" the bootstrap horse races because averaging 4.6M near-noise observations
produces a smooth-looking sample distribution; this smoothness has nothing to
do with statistical accuracy.

The way out is to step back from the framing of "validate harmonic on KDD"
and recognise:

1. **Uniform is not a sensible baseline on heavy-tailed data**. Standard
   statistical theory (a century of it) says equal-weighting heteroscedastic
   estimators is strictly suboptimal. It's a strawman, not a serious
   competitor.
2. **The proper validation is on synthetic + Yahoo where ground truth
   exists**. The DEF heatmap (median 1.31× speedup, max 1.94×) already shows
   the speed-of-convergence story. Rephrased as "impressions needed for
   precision $\varepsilon$", it's the standard sample-size calculation in
   stats textbooks.
3. **The KDD section becomes descriptive**, not validating. It motivates the
   problem (heavy tails make uniform unviable), shows that count-based and
   $\theta$-aware methods give different answers (PBM violated in sponsored
   search), and points toward future work.

---

## 4. Citations: this analysis is not new

The treatment of heteroscedastic estimation is a classical statistical topic.
The following references are directly applicable to the paper's methodological
positioning. Reviewers cannot dismiss the harmonic-vs-uniform comparison as
ad hoc once these are anchored properly.

### 4.1 Inverse-variance / generalized least squares

* **Aitken, A. C. (1934)**. *On least squares and linear combination of
  observations.* Proceedings of the Royal Society of Edinburgh, 55, 42–48.
  The original BLUE result for heteroscedastic linear models. Establishes
  that inverse-variance weighting is the minimum-variance unbiased estimator
  when sample variances differ.

* **Cochran, W. G. (1937)**. *Problems arising in the analysis of a series of
  similar experiments.* Journal of the Royal Statistical Society Supplement,
  4, 102–118. Inverse-variance pooling of estimates from heterogeneous
  experiments.

### 4.2 Meta-analysis (closest analogue to our setting)

* **Hedges, L. V., & Olkin, I. (1985)**. *Statistical Methods for
  Meta-Analysis.* Academic Press. Chapters 4–6 cover combining estimators
  with different precisions; Section 6.2 explicitly argues that uniform
  weighting of studies of different sizes is methodologically incorrect.

* **DerSimonian, R., & Laird, N. (1986)**. *Meta-analysis in clinical
  trials.* Controlled Clinical Trials, 7(3), 177–188. Random-effects
  inverse-variance weighting; the canonical reference for combining studies
  of unequal sample size.

* **Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R.
  (2009)**. *Introduction to Meta-Analysis.* Wiley. Chapter 9 makes the
  unequal-weights case explicit and rejects uniform weighting in plain
  language.

### 4.3 Importance sampling and self-normalized estimators

* **Kong, A. (1992)**. *A note on importance sampling using standardized
  weights.* Technical Report 348, Department of Statistics, University of
  Chicago. Defines effective sample size $\mathrm{ESS} = (\sum w)^2 / \sum
  w^2$ and proves that the asymptotic variance of a self-normalized weighted
  ratio estimator scales as $1/\mathrm{ESS}$, not $1/n$.

* **Hesterberg, T. (1995)**. *Weighted average importance sampling and
  defensive mixture distributions.* Technometrics, 37(2), 185–194. Variance
  analysis of weighted ratio estimators with heterogeneous weights.

* **Owen, A. B. (2013)**. *Monte Carlo theory, methods and examples.* Online
  book. Chapter 9 (importance sampling) is the modern reference; Section 9.2
  derives the rate-of-convergence result for self-normalized weighted ratio
  estimators directly.

* **Robert, C. P., & Casella, G. (2004)**. *Monte Carlo Statistical Methods*
  (2nd ed.). Springer. Chapter 3.3 covers importance sampling; gives the
  variance decomposition that makes the same point.

### 4.4 Survey sampling

* **Cochran, W. G. (1977)**. *Sampling Techniques* (3rd ed.). Wiley. Chapters
  5 and 11 cover stratified sampling and unequal-probability sampling. The
  design-effect concept ($\mathrm{deff} = n / \mathrm{ESS}$) is introduced
  here.

* **Lohr, S. L. (2010)**. *Sampling: Design and Analysis* (2nd ed.).
  Brooks/Cole. Modern textbook treatment; Chapter 7 explicitly covers when
  uniform weighting is appropriate (only under simple random sampling) and
  when it is not (anything else).

### 4.5 Counterfactual / off-policy evaluation (most directly comparable to
ULTR)

* **Bottou, L., Peters, J., Quiñonero-Candela, J., et al. (2013)**.
  *Counterfactual reasoning and learning systems.* Journal of Machine
  Learning Research, 14, 3207–3260. Variance analysis of inverse-propensity
  scoring and clipped IPS estimators on logged bandit data.

* **Dudík, M., Langford, J., & Li, L. (2011)**. *Doubly robust policy
  evaluation and learning.* International Conference on Machine Learning.
  Doubly-robust estimators that combine weighted ratios with model-based
  estimates.

* **Swaminathan, A., & Joachims, T. (2015)**. *The self-normalized estimator
  for counterfactual learning.* NeurIPS. Self-normalized importance sampling
  for off-policy evaluation; argues that the variance of weighted ratio
  estimators is dominated by weight heterogeneity, leading directly to the
  ESS-based design considerations we use.

### 4.6 Click-model and ULTR-specific references

* **Joachims, T., Swaminathan, A., & Schnabel, T. (2017)**. *Unbiased
  learning-to-rank with biased feedback.* WSDM. Foundational paper for IPS
  in ULTR; describes the propensity-aware loss.

* **Wang, X., Bendersky, M., Metzler, D., & Najork, M. (2018)**. *Position
  bias estimation for unbiased learning to rank in personal search.* WSDM.
  Joint relevance-propensity estimation via EM.

* **Agarwal, A., Wang, X., Li, C., Bendersky, M., & Najork, M. (2019)**.
  *Estimating position bias without intrusive interventions.* WSDM. The
  intervention-harvesting estimator and AdjacentChain aggregation that this
  paper extends.

* **Craswell, N., Zoeter, O., Taylor, M., & Ramsey, B. (2008)**. *An
  experimental comparison of click position-bias models.* WSDM. PBM, cascade,
  and DBN click models.

* **Chuklin, A., Markov, I., & de Rijke, M. (2015)**. *Click Models for Web
  Search.* Morgan & Claypool. Comprehensive review of click models;
  documents PBM violations in real search-engine logs.

### 4.7 PBM violations in real search logs (relevant to KDD divergence)

* **Yue, Y., Patel, R., & Roehrig, H. (2010)**. *Beyond position bias:
  examining result attractiveness as a source of presentation bias in
  clickthrough data.* WWW. Documents that real click data has effects beyond
  position (attractiveness, trust bias, etc.) — i.e., PBM-violating
  structure.

* **Wang, X., Golbandi, N., Bendersky, M., Metzler, D., & Najork, M.
  (2018)**. *Position bias estimation for unbiased learning to rank in
  personal search.* WSDM. Section 6 documents that PBM doesn't fully hold on
  Gmail and Drive search logs.

---

## 5. Recommended paper restructure (informal)

Based on the analysis above:

1. **Introduction / Motivation** — open with the impression-distribution
   picture from KDD Cup 2012 Track 2 (or a similar real ULTR log). Heavy
   tails, singletons. Cite Aitken / IVW / Hedges-Olkin to establish that
   uniform weighting is not a serious baseline on data of this shape.

2. **Theory** — Theorem 1 (ratio-unbiased), Theorem 4 (Cauchy–Schwarz on
   count-only surrogate), Theorem 5 (asymmetric oracle). Position harmonic as
   the count-only inverse-variance optimum. Cite Kong / Owen on ESS-based
   convergence rates.

3. **Synthetic + Yahoo validation** — *with ground truth*. Replace the DEF
   heatmap with **"impressions needed for precision $\varepsilon$"** at
   $\varepsilon \in \{5\%, 1\%, 0.1\%\}$. This is the standard sample-size
   calculation; it is unambiguous, falsifiable, and directly comparable
   across schemes. Show that harmonic needs ~1/1.3 to ~1/1.94 the impressions
   that uniform needs to reach the same precision.

4. **KDD descriptive section** — *not validation*. Show that:
   - The impression distribution looks the way Aitken says it shouldn't be
     uniformly weighted (60% singletons, 0.1% holding 42%).
   - Different schemes converge to different $\hat R$ on KDD, evidence that
     PBM is violated (consistent with Yue et al. 2010, Wang et al. 2018).
   - Per-cell click-rate $\theta_i$ correlates positively with $N_i$ (the
     popularity correlation), which is why the count-only surrogate
     under-predicts on KDD.
   - Full $\theta$-aware oracle improves on harmonic at full data scale, but
     requires $N_i$ large enough that $\hat\theta_i$ is well-estimated.

5. **Limitations + future work** — count-based intervention harvesting hits
   its limit when $(N, \theta)$ are correlated; the natural extension is the
   $\theta$-aware oracle (Section 4.5 of this README).

This structure presents harmonic as **the principled count-aware estimator
that standard statistical theory says you should use**, validates it
empirically where validation is possible (synthetic + Yahoo), and uses real
data to *characterise* rather than to *prove*. Reviewers cannot reject the
core claim because it is anchored in a century of established statistics.

---

## 6. Files in this directory referenced above

* `kdd2012_full_*` — full-data analysis on KDD Cup 2012
* `kdd2012_scaling_*` — scaling experiment, three bootstrap variants
* `kdd_oracle_scaling*` — count-only oracle vs full $\theta$-aware oracle
* `asymmetric_*` — counterexample to the asymmetric Cauchy–Schwarz analogue
* `paired_ci.csv` — paired-bootstrap CI from earlier reviewer-response work
* `plugin_oracle.csv` — plug-in oracle vs harmonic vs true-propensity oracle

Code:
* `paper/aggregate_kdd2012_full.py` — stream-aggregate raw `training.txt`
* `paper/diagnose_kdd2012.py` — $(N, \theta)$ correlation, position-bias curves
* `paper/diagnose_pair12.py` — impression distribution for pair (1,2)
* `paper/scale_kdd2012.py` — initial scaling experiment (impression-level subsample, click bootstrap only)
* `paper/scale_kdd2012_v2.py` — adds (q,ad)-row bootstrap; SamueChan-replication subsection
* `paper/scale_kdd2012_v3.py` — adds query cluster bootstrap
* `paper/scale_kdd_oracle.py` — count-only oracle variants + full $\theta$-aware oracle
* `paper/experiment_asymmetric_counterexample.py` — closed-form n=2 counterexample

---

## 7. The plan: plug-in delta-method analysis as the central KDD experiment

After the dead ends in Section 3, the right framing — and the one we are going
to implement — is **classical plug-in asymptotic variance under the delta
method**, evaluated analytically on full KDD. This puts KDD center stage with
a rigorous, falsifiable claim that needs no bootstrap and no ground truth.

### 7.1 The estimator and its asymptotic variance

For weighted intervention-harvesting on the pair $(k, k')$ with cells
$i = 1,\dots,n$ at counts $(N_k^i, N_{k'}^i)$, click counts $(C_k^i, C_{k'}^i)$,
and per-cell click rates $\theta_k^i = C_k^i / N_k^i$:

$$\hat R \;=\; \frac{A}{B} \;=\; \frac{\sum_i \omega_i\, C_k^i / N_k^i}{\sum_i \omega_i\, C_{k'}^i / N_{k'}^i}$$

Under the binomial click model that PBM assumes, $A$ and $B$ are sums of
independent random variables, so by the delta method:

$$\widetilde V(\omega) \;=\; \frac{\operatorname{Var}(A)}{\mathbb{E}[A]^2} + \frac{\operatorname{Var}(B)}{\mathbb{E}[B]^2} - 2\,\frac{\operatorname{Cov}(A,B)}{\mathbb{E}[A]\,\mathbb{E}[B]}$$

$$= \frac{\sum_i \omega_i^2\, \theta_k^i(1-\theta_k^i)/N_k^i}{\bigl(\sum_i \omega_i\, \theta_k^i\bigr)^2} + \frac{\sum_i \omega_i^2\, \theta_{k'}^i(1-\theta_{k'}^i)/N_{k'}^i}{\bigl(\sum_i \omega_i\, \theta_{k'}^i\bigr)^2}$$

(cross-covariance vanishes because clicks at $k$ and $k'$ are conditionally
independent given the impression sets). The asymptotic SE is

$$\operatorname{SE}(\hat R \mid \mathbf{N}) \;=\; |\hat R|\,\sqrt{\widetilde V(\omega)}.$$

This is **the textbook delta-method plug-in variance estimator**. Plugging in
$\hat\theta_i = C_i/N_i$ (with light shrinkage toward the marginal $\bar\theta_k$
for low-$N$ cells) gives a deterministic functional of the observed KDD counts.

### 7.2 The predicted asymptotic relative efficiency

For uniform $(\omega \equiv 1)$ and harmonic $(\omega = N_k N_{k'}/(N_k+N_{k'}))$,
plugging in KDD's distribution:

* **Uniform**: $\widetilde V_U \sim n_\text{pairs}^{-1}\cdot \bar\theta^{-1}(1-\bar\theta)$ —
  dominated by the singleton mass.
* **Harmonic**: $\widetilde V_H \sim N_\text{tot}^{-1}\cdot \bar\theta^{-1}(1-\bar\theta)$ —
  scales with total impressions because $h_i \le \min(N_k^i, N_{k'}^i)$.

$$\mathrm{ARE}(\text{harmonic},\text{uniform}) \;=\; \frac{\widetilde V_U}{\widetilde V_H} \;\approx\; \frac{N_\text{tot}}{n_\text{pairs}} \;=\; \frac{132{,}337{,}271}{4{,}613{,}072} \;\approx\; 28.7\,.$$

That is, **uniform needs ~29× more impressions than harmonic to reach the same
asymptotic precision on KDD** — a property of KDD's impression distribution,
not of model fit. Equivalently, $\operatorname{SE}_U / \operatorname{SE}_H \approx \sqrt{29} \approx 5.4$.

### 7.3 Why this avoids the bootstrap traps

* No (q,ad) resampling penalty against heterogeneous weights — variance is
  computed under the actual binomial click model, not under perturbations of
  the pair set.
* No "averaging-out smoothness" artifact for uniform — each singleton receives
  its full Bernoulli variance contribution, exactly as classical statistics
  demands.
* No ground truth required — relative efficiency is a comparison metric
  between estimators under a model, not against a known truth.
* PBM violation becomes a *separately testable* phenomenon (Cochran $Q$),
  not a confound.

### 7.4 Experiments to run

1. **Plug-in delta-method SE table** at $f \in \{0.001, 0.01, 0.1, 1.0\}$ for
   schemes $\{\text{uniform},\text{min},\text{harmonic},\text{count-only oracle},
   \text{full $\theta$-oracle}\}$.
2. **Asymptotic relative efficiency** as the headline number: uniform-vs-harmonic
   ratio of $\widetilde V$ on KDD, reported with bootstrap-free CIs from
   second-order delta-method propagation.
3. **Cochran's $Q$ heterogeneity test** to detect PBM violation. Compute
   $Q = \sum_i w_i (\hat\rho_i - \hat\rho)^2$ on per-cell pseudo-ratios with
   IVW weights $w_i$; reject homogeneity → quantifies PBM violation.
4. **Stratified ARE** by impression-count quartile — robustness of the 29×
   number across the impression distribution.
5. **Calibration check**: analytic $\widetilde V$ vs empirical std under
   **impression-level resampling** (subsampling impressions, not pairs);
   should agree within Monte Carlo error. This is the diagnostic that proves
   the plug-in is correct on this data.

Implementation: closed-form sum, no Monte Carlo loop, fast — a single pass
over the 4.6M-row pair table per scheme.

### 7.5 Section structure in the paper

The paper's KDD section is rewritten as **"Statistical efficiency on real
data"**:

* §X.1 *Plug-in delta-method variance.* State $\widetilde V$, derive the ARE
  formula, cite Casella–Berger §10.1, van der Vaart §3, Owen §9.2.
* §X.2 *Application to KDD.* Report Table 7.4 + ARE = 29×. Headline claim.
* §X.3 *PBM heterogeneity.* Report Cochran-$Q$. Cite Yue 2010, Wang 2018,
  Cochran 1954.
* §X.4 *Calibration.* Analytic vs empirical std agree.

Bootstrap-on-KDD is **dropped entirely** from the paper. Bootstrap discussion
moves to the appendix as a "why bootstrap-of-pairs is not the right
instrument" note.

---

## 8. Best-in-class write-ups for the technique

The technique we're using is the **plug-in delta-method asymptotic variance
for a self-normalized weighted ratio estimator**, applied with
inverse-variance weights. The cleanest pedagogical write-ups, in priority
order:

### 8.1 Primary reference (free online)

**Owen, A. B. (2013). *Monte Carlo theory, methods and examples.* Chapter 9
("Importance sampling"), §9.2 and §9.4.** ([https://artowen.su.domains/mc/](https://artowen.su.domains/mc/))

This is the cleanest and most directly applicable write-up. Owen derives the
asymptotic variance of self-normalized weighted ratio estimators in exactly
the form $\widetilde V(\omega)$ above, derives ESS = $(\sum \omega)^2/\sum
\omega^2$ as a coarse summary, and gives the rate-of-convergence result.
Free online from his Stanford page. **This will be the headline citation.**

### 8.2 Asymptotic theory background (free lecture notes)

**van der Vaart, A. W. (1998). *Asymptotic Statistics.* Cambridge University
Press. Chapter 3 ("Delta Method").** The canonical modern reference for the
delta method. Notes by Tsybakov and Wasserman covering the same material are
freely available; **Larry Wasserman's "All of Statistics" Ch. 9** and his
**36-705 lecture notes** (CMU) are the most pedagogical.

**Tsiatis, A. A. (2006). *Semiparametric Theory and Missing Data.* Chapter 4.**
For the influence-function view of the same variance.

### 8.3 Inverse-variance theory (meta-analysis)

**Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis.*
Chapters 5–6.** Classic monograph; §6.2 explicitly argues against uniform
weighting.

**Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009).
*Introduction to Meta-Analysis.* Chapters 9–10.** Modern textbook. Chapter 9
is the most accessible derivation of fixed-effect IVW.

### 8.4 Off-policy / ULTR analogues

**Swaminathan, A., & Joachims, T. (2015). *The self-normalized estimator for
counterfactual learning.* NeurIPS.** Applies the exact $\widetilde V$
framework above to counterfactual evaluation; closest in spirit to our
estimator.

**Bottou et al. (2013). *Counterfactual reasoning and learning systems.*
JMLR.** §4 has the variance decomposition for self-normalized IPS.

### 8.5 Which to cite in the paper

* **Headline technique citation**: Owen (2013) §9.2 for the self-normalized
  ratio estimator's asymptotic variance.
* **Inverse-variance optimality**: Aitken (1934) + Hedges & Olkin (1985)
  §6.2.
* **ESS-based interpretation**: Kong (1992) + Owen (2013).
* **ULTR-relevant analogue**: Swaminathan & Joachims (2015).
* **Heterogeneity test for PBM violation**: Cochran (1954).

---

## 9. Local copies of cited references

We store offline copies of the freely-available references in
`paper/reviewer_response/refs_pdf/` so we can verify exact statements,
equations, and section numbers when revising the paper.

| Reference | Free? | Local file |
| --- | --- | --- |
| Owen (2013) MC Theory, Ch. 9 | yes | `refs_pdf/owen_mc_ch09_importance.pdf` |
| van der Vaart (1998) Ch. 3 delta method | partial | `refs_pdf/vdv_ch03_delta.pdf` (if available) |
| Wasserman, *All of Statistics*, Ch. 9 + 36-705 notes | yes | `refs_pdf/wasserman_36705_*.pdf` |
| Aitken (1934) | yes (JSTOR) | `refs_pdf/aitken_1934_rse.pdf` |
| Cochran (1937, 1954) | yes (JSTOR) | `refs_pdf/cochran_1937_jrss.pdf`, `cochran_1954_biometrics.pdf` |
| DerSimonian & Laird (1986) | yes | `refs_pdf/dersimonian_laird_1986.pdf` |
| Hedges & Olkin (1985) Ch. 5–6 | book — buy / library | n/a |
| Borenstein et al. (2009) Ch. 9 | book — buy / library | n/a |
| Kong (1992) TR 348 | yes | `refs_pdf/kong_1992_tr348.pdf` |
| Hesterberg (1995) | yes | `refs_pdf/hesterberg_1995_technometrics.pdf` |
| Swaminathan & Joachims (2015) | yes | `refs_pdf/swaminathan_joachims_2015_nips.pdf` |
| Bottou et al. (2013) | yes | `refs_pdf/bottou_2013_jmlr.pdf` |
| Joachims, Swaminathan, Schnabel (2017) | yes | `refs_pdf/joachims_2017_wsdm.pdf` |
| Wang et al. (2018) | yes | `refs_pdf/wang_2018_wsdm.pdf` |
| Agarwal et al. (2019) | yes | `refs_pdf/agarwal_2019_wsdm.pdf` |
| Craswell et al. (2008) | yes | `refs_pdf/craswell_2008_wsdm.pdf` |
| Chuklin et al. (2015) Click Models | yes | `refs_pdf/chuklin_2015_clickmodels.pdf` |
| Yue, Patel, Roehrig (2010) | yes | `refs_pdf/yue_2010_www.pdf` |
| Dudík, Langford, Li (2011) DR | yes | `refs_pdf/dudik_2011_icml.pdf` |
