# Harmonic Weighting for Variance-Reduced Intervention Harvesting

## Context

This spec describes extensions to the `ultr-bias-toolkit` repo (https://github.com/alemagnani/ultr-bias-toolkit) for a SIGIR 2026 paper revision on variance-reduced position bias estimation via intervention harvesting.

The paper builds on Agarwal et al. (2019) "Estimating Position Bias without Intrusive Interventions" (WSDM). The original intervention harvesting uses unit weights per (q,d) pair. We propose harmonic weighting to reduce variance.

## Core Theory

### Setup
For an interventional set S_{k,k'}, each (q,d) pair i has:
- N_k^i: impressions at position k
- N_{k'}^i: impressions at position k'
- C_k^i, C_{k'}^i: total clicks at each position

The ratio estimator is R = A/B where:
- A = sum_i omega_i * C_k^i / N_k^i  (numerator)
- B = sum_i omega_i * C_{k'}^i / N_{k'}^i  (denominator)

### Weighting schemes to implement

1. **Original** (baseline): omega_i = 1
2. **Min-weighting**: omega_i = min(N_k^i, N_{k'}^i)
3. **Harmonic weighting**: omega_i = N_k^i * N_{k'}^i / (N_k^i + N_{k'}^i)
4. **Clipped weights**: subsample the majority position if N_k^i/N_{k'}^i > tau. Test tau in {2, 5, 10}
5. **Self-normalized**: divide by sum of weights instead of count n
6. **Adaptive (two-stage)**:
   - Stage 0: estimate propensities with harmonic weights -> get p_hat_k, p_hat_{k'}
   - Stage 1: compute alpha = (1-p_hat_k)/p_hat_k, beta = (1-p_hat_{k'})/p_hat_{k'}
   - Set omega_i = N_k^i * N_{k'}^i / (alpha * N_{k'}^i + beta * N_k^i)
   - Re-estimate propensities
   - Use sample splitting (cross-fitting) to preserve unbiasedness

### Key theoretical results

**Ratio-unbiasedness**: For ANY weight omega_i that depends only on observation counts (not clicks), E[A]/E[B] = p_k/p_{k'}. This holds for original, min, harmonic. Clipping breaks this. Adaptive breaks this without sample splitting.

**Delta method variance**: Var(R | N) ≈ R^2 [alpha * V_k(omega) + beta * V_{k'}(omega)]
where:
- R = p_k/p_{k'}
- alpha = (1-p_k)/p_k, beta = (1-p_{k'})/p_{k'}
- V_r(omega) = sum_i omega_i^2 / N_r^i / (sum_i omega_i)^2

**Variance decomposition** (Theorem 6 in paper):
alpha*D_k + beta*D_{k'} = beta*(D_k + D_{k'}) - (beta - alpha)*D_k
where D_r = V_r(omega^h) - V_r(1).
- First term <= 0 always (Cauchy-Schwarz)
- Second term <= 0 when D_k >= 0 (systematic imbalance)

**Optimality**: The variance-optimal weight is omega_i* proportional to N_k^i * N_{k'}^i / (alpha * N_{k'}^i + beta * N_k^i). When alpha = beta, this equals harmonic.

## Experiments to Implement

### Experiment 1: Imbalance Sweep (Synthetic)
- PBM with p_k = (1/k)^eta, eta = 1.0, M = 10 positions
- Sweep traffic split: 50/50, 60/40, 70/30, 80/20, 90/10, 95/5
- Sweep frequency imbalance ratio: 1:1 to 100:1
- For each configuration, run 500 independent trials
- Report: MSE, variance, bias^2 for each weighting scheme
- Plot variance reduction as function of imbalance severity
- Test both AdjacentChain and AllPairs estimators
- Expected result: harmonic >= min > self-norm > clipping (well-tuned) > original; convergence as imbalance -> 0

### Experiment 2: PBM Violations (Synthetic)
- Generate clicks under models that violate PBM:
  (a) Trust bias: users click non-relevant items at top positions. Use model from Vardasbi et al. (CIKM 2020): P(C=1|q,d,k) = p_k * rel(q,d) + epsilon_k * (1 - rel(q,d)), where epsilon_k is trust bias at position k
  (b) Cascade model: user scans top-down, stops after first click
  (c) Position-dependent relevance: P(C=1|q,d,k) = p_k * rel(q,d,k) where relevance depends on position
- Key insight: ALL intervention harvesting methods share the same bias under model misspecification. Our method has lower VARIANCE, hence lower MSE.
- Show MSE breakdown: bias^2 + variance for each method

### Experiment 3: Yahoo LTR (Semi-Synthetic)
- 10 LambdaMART rankers trained on different query subsets
- E-commerce simulation: 1% anchor items (1-50 impressions), rest 1-10 impressions
- Report both AdjacentChain and AllPairs
- Sweep anchor item fraction and impression counts

### Experiment 4: Baidu-ULTR (Null Result)
- Load Baidu-ULTR dataset
- Show that N_k^i = N_{k'}^i = 1 for essentially all pairs
- Therefore all weighting schemes produce identical estimates
- Report this as confirming our method targets the imbalanced regime
- One table or one paragraph is sufficient

### Experiment 5: Downstream LTR Quality
- Use estimated propensities in IPS-weighted LTR objective on Yahoo
- Train ranking model, evaluate NDCG@5, NDCG@10
- Compare: true propensities, original estimated, harmonic estimated
- Hypothesis: lower-variance propensity estimates -> better ranking

### Experiment 6: Adaptive Two-Stage (NEW)
- Compare: harmonic (one stage) vs adaptive (2 iterations) vs adaptive (3 iterations)
- Use cross-fitting for unbiasedness: split data in half, estimate on each half using weights from the other
- Show improvement is largest for AllPairs with non-adjacent positions (where alpha != beta)
- Show marginal improvement for AdjacentChain (where alpha ≈ beta, harmonic already near-optimal)

### Experiment 7: Verify D_k >= 0 Condition
- For each experimental configuration, compute D_k and D_{k'} for harmonic weights
- Show D_k >= 0 holds in all systematic imbalance configurations
- Show it can fail under adversarial mixed imbalance (construct example)
- This validates the sufficient condition in Theorem 6

## Implementation Notes

### Weighting schemes
Each weighting scheme should be implemented as a function that takes (N_k, N_{k'}) arrays and returns weight arrays. They should be plug-and-play into existing AdjacentChain and AllPairs estimators.

```python
def weight_original(N_k, N_kp):
    return np.ones_like(N_k, dtype=float)

def weight_min(N_k, N_kp):
    return np.minimum(N_k, N_kp).astype(float)

def weight_harmonic(N_k, N_kp):
    return (N_k * N_kp) / (N_k + N_kp)

def weight_clipped(N_k, N_kp, tau):
    """Subsample majority position to enforce max ratio tau."""
    ratio = N_k / N_kp
    N_k_clipped = np.where(ratio > tau, tau * N_kp, N_k)
    N_kp_clipped = np.where(1/ratio > tau, tau * N_k, N_kp)
    return N_k_clipped, N_kp_clipped  # Returns modified counts, not weights

def weight_adaptive(N_k, N_kp, alpha, beta):
    """Variance-optimal weight given known alpha, beta."""
    return (N_k * N_kp) / (alpha * N_kp + beta * N_k)
```

### Variance diagnostics
For each run, compute and store:
- V_k(omega), V_{k'}(omega) for each weighting scheme
- D_k = V_k(omega) - V_k(1), D_{k'} = V_{k'}(omega) - V_{k'}(1)
- The decomposition: beta*(D_k + D_{k'}) and -(beta-alpha)*D_k
- Verify that both terms <= 0 in systematic imbalance

### Cross-fitting for adaptive weights
```python
def adaptive_cross_fit(data, estimator):
    # Split data into two halves
    half1, half2 = split_data(data)
    
    # Stage 0: estimate on each half with harmonic weights
    p_hat_1 = estimator(half1, weight_fn=weight_harmonic)
    p_hat_2 = estimator(half2, weight_fn=weight_harmonic)
    
    # Stage 1: compute adaptive weights from opposite half's estimates
    alpha_1, beta_1 = compute_alpha_beta(p_hat_1)
    alpha_2, beta_2 = compute_alpha_beta(p_hat_2)
    
    # Re-estimate using adaptive weights on opposite half
    p_final_1 = estimator(half2, weight_fn=lambda Nk, Nkp: weight_adaptive(Nk, Nkp, alpha_1, beta_1))
    p_final_2 = estimator(half1, weight_fn=lambda Nk, Nkp: weight_adaptive(Nk, Nkp, alpha_2, beta_2))
    
    # Average
    return (p_final_1 + p_final_2) / 2
```

## Reviewer Concerns to Address

1. **Reviewer VdLt**: Theorem 2 proof was flawed (comparing second moments without equal first moments). FIXED in revised paper with delta method on ratio estimator.

2. **Reviewer PLK6**: Wants Baidu-ULTR (Exp 4 — null result), REM baseline, AdjacentChain on Yahoo, PBM violations (Exp 2), writing fixes.

3. **Reviewer U93F**: Wants clipping/self-normalized baselines (Exp 1), moderate imbalance regimes (Exp 1 sweep), connection to importance weight clipping discussed.

4. **All reviewers**: Only synthetic/semi-synthetic experiments. Baidu null result explains why. Sweep experiments show robustness across conditions.

## Paper Structure (for reference)

The LaTeX source is in paper.tex. Key sections:
- Theorem 1: Ratio-unbiasedness (any count-dependent weight)
- Proposition 3: Delta method conditional variance
- Theorem 4: Cauchy-Schwarz (V_k + V_{k'} always decreases) — UNCONDITIONAL
- Theorem 5: Harmonic is optimal when alpha = beta
- Theorem 6: Variance decomposition with D_k >= 0 condition — the main result
- Remarks 7-9: Interpretation of D_k, systematic imbalance, mixed imbalance

