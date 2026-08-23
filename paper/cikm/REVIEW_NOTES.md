# CIKM'26 Short Paper — Review Self-Assessment & Camera-Ready Notes

**Submission 3456**: *Surrogate Inverse-Variance Weighting for Intervention
Harvesting on Heavy-Tailed Click Logs* (short track).
Notifications: **Aug 7, 2026**. Camera-ready: **Aug 20**. Conf: Rome, Nov 9–11.

**NOTE: CIKM has NO author-rebuttal / response phase.** The Aug 7 decision is
final; there is no window to answer reviewers. So this is NOT rebuttal prep.
Its value is: (1) a realistic self-assessment of accept/reject odds *now*;
(2) a checklist of what to fix for camera-ready *if accepted*; (3) what to
strengthen *before resubmitting elsewhere if rejected* — the objections below are
exactly what the next venue's reviewers will raise too.

Key asset: the **full-length draft** (`cikm_full.tex`) already ran several of
the analyses reviewers care about. For each point below: **[SHORT]** = already
in the submitted short paper, **[FULL]** = exists in the full draft and can be
folded into the camera-ready or a resubmission, **[GAP]** = genuinely not done.

---

## Tier 1 — the objections most likely to decide the paper

### R1. "No downstream LTR result — does better propensity precision improve nDCG?" [GAP]
The single most predictable ULTR-reviewer ask. Neither version demonstrates that
harmonic propensities yield better unbiased-LTR ranking.
- **Response framing:** the paper's scope is propensity *estimation efficiency*,
  not end-to-end LTR; the estimand is the position-bias ratio, and we are explicit
  (Sec. estimand) that we are not claiming a "more correct" ratio.
- **Best rebuttal move:** if bandwidth allows before the response deadline, run one
  small semi-synthetic unbiased-LTR experiment (Yahoo, IPS-LTR with propensities
  from uniform vs harmonic) and report ΔnDCG. Even a single cell neutralizes this.
- **Fallback:** full-draft open-direction (a) already frames AdjacentChain →
  absolute-propensity propagation as the honest next step; cite it as scoped.
- **Do NOT** over-claim a downstream benefit we haven't measured.

### R2. "The bootstrap is discarded exactly where it makes harmonic look worse — motivated reasoning?" [SHORT arg / FULL evidence]
This is the sharpest possible attack and only a sophisticated reviewer will find it.
The short paper argues pair-bootstrap is "misaligned" and adopts click-resample.
The **full draft (Table `tab:kdd2012`) openly shows harmonic/min have 3× LARGER
bootstrap std than uniform on (1,2).** So the "bootstrap is the wrong tool"
argument conveniently sidelines the one model-free measure where harmonic loses.
- **Response (defensible, deliver carefully):** the two quantities are different by
  construction — $V_k(\omega)$ is click-noise variance *conditional on counts*
  (what Thm 3.x bounds and what matters for a fixed interventional set), whereas
  pair-bootstrap resamples the *count population* and is dominated by the
  heavy-tailed count distribution, penalizing any concentrated-weight estimator
  (the IS-ESS penalty). This is a property of the resampling design, not the
  estimator.
- **Corroboration we DO have:** plug-in $\widetilde V$ agrees with the
  *click-resample* (parametric Bernoulli, conditional-on-counts) SE to **1–2%**
  (Table `tab:scaling`). That is the apples-to-apples empirical check.
- **Honesty hedge:** concede the framing openly — "we agree bootstrap-std is worse
  for harmonic and explain precisely why; we did not choose click-resample to hide
  it, we chose it because it isolates the conditional efficiency the theorem is
  about." Pre-empting beats getting caught.

### R3. "Harmonic ≈ min empirically (4.53 vs 4.41). Why not just use min?" [FULL]
The theory's payoff over a trivial heuristic is <3% on KDD and statistically
indistinguishable in the high-imbalance regimes.
- **Evidence ready [FULL]:** paired bootstrap CIs (Table `tab:paired_ci`) —
  harmonic−min CI *crosses zero at r=50,100*. The short paper already concedes the
  tie honestly; point reviewers to the full-draft CI table as proof we quantified it.
- **Response:** the contribution is not the margin over min — it is (i) the
  count-only **surrogate-IVW characterization** (harmonic is the inverse-variance
  optimum of the relevance-agnostic surrogate; min is not optimal for anything),
  (ii) the **unconditional** Cauchy–Schwarz guarantee that min lacks, and (iii) the
  **θ-oracle-collapse finding** (below). Min is a heuristic that happens to work;
  harmonic is the principled object that explains *why* count-only weighting is the
  right family.

### R4. "Inverse-variance weighting is textbook — where's the novelty?" [SHORT]
- **Response:** novelty is not IVW per se. It is: (a) the **count-only surrogate** —
  dropping the relevance-dependent variance term is what makes the optimum
  *deployable* (the exact IVW oracle needs per-cell relevance); (b) the empirical
  **discovery that the exact plug-in IVW oracle collapses** (ARE 0.002) on
  heavy-tailed logs while count-only weighting is robust — a genuinely
  counter-intuitive result; (c) the **methodological point that pair-bootstrap is
  the wrong efficiency diagnostic** on singleton-dominated data. Frame the paper as
  "the right diagnostic + robustness finding for a real, ignored problem," not "a
  new weight."

---

## Tier 2 — likely but answerable

### R5. "Estimand drift: PBM is mis-specified, so you precisely estimate a weight-specific target, not the truth." [SHORT — already addressed]
- Fully pre-empted: Cochran Q/df=3.83 quantifies the mis-specification; the estimand
  paragraph states plainly that choosing ω is a precision/estimand trade-off and
  that $\widetilde V$ is a precision statement at fixed ω, not a correctness claim.
  This is a **strength** — lead with the honesty. If a reviewer raises it as a flaw,
  quote our own text back.

### R6. "ESS = 1,849 out of 4.6M pairs — the estimate rests on ~1,800 cells. Fragile." [SHORT — addressed]
- Weight-cap robustness: ARE stays 4.12 (99.99th pct), 2.83 (99th), 1.86 (90th).
  Harmonic beats uniform across all caps. Response ready.
- Caveat to own: capping shifts the estimand ($\hat R$ 1.71→1.61). Frame as
  consistent with R5 (precision/estimand trade-off), not a contradiction.

### R7. "Single real dataset; only 3 positions; sponsored search, not organic web." [SHORT + FULL]
- **Response:** KDD Cup 2012 is, to our knowledge, the only *public* log preserving
  per-impression counts at scale; the widely-used Baidu-ULTR release is deduplicated
  (94% of cells N=1), which *erases* the very imbalance the method targets — this is
  itself a documented finding [FULL Sec `sec:baidu`], not an excuse. Open-direction
  (d) explicitly calls for an organic-web log with M≥5. Turn the limitation into a
  call-to-action for better dataset releases.

### R8. "Why no comparison to feature-based EM (Wang et al. 2018)?" [SHORT — addressed]
- Vanilla no-feature PBM-MLE collapses to the uniform estimand (1.682) on
  singleton-dominated data; feature-based EM needs features absent from the
  anonymized aggregation. On synthetic PBM, MLE recovers R=2.0 (SE 0.035) confirming
  implementation. The paradigms solve different problems (parametric joint inference
  vs model-free propensity-only) — comparison is methodologically ill-posed, not
  omitted.

### R9. "Asymmetric surrogate: your guarantee is only symmetric; does harmonic ever lose?" [FULL — strong material]
- **Do not hide this — the full draft turns it into a credibility win.** We have an
  explicit closed-form n=2 counterexample and a *structured* two-archetype search
  where harmonic loses to uniform in 100% of a 15×15 (α,β) grid (worst ratio 7.78).
  We then show real distributions (KDD, Yahoo) never exhibit the required
  anti-correlated degenerate pattern. Pasting this into a rebuttal signals rigor and
  pre-empts the "you only proved the easy case" critique.

---

## Tier 3 — minor / camera-ready fixes (do regardless of outcome)

- **F1. Stale template metadata.** `cikm_short.tex` line 19:
  `\acmConference[CIKM '26]{ACM CIKM}{October 2026}{Toronto, Canada}` — CIKM'26 is
  **Rome, Nov 9–11**. (The full draft is worse: SIGIR template, Washington DC.) Fix
  before camera-ready.
- **F2. Cross-paper R̂ inconsistency.** Short paper reports uniform $\hat R_{1,2}=1.683$
  (Table `tab:are_full`); full draft reports uniform $\hat R_{1,2}=1.542$
  (Table `tab:kdd2012`) on ostensibly the same data/pair. Different venues so
  reviewers won't see both, but reconcile the pipeline (likely a click-threshold or
  subsampling difference) so the numbers agree in your own records — a reviewer who
  finds an arXiv full version could flag it.
- **F3. GenAI disclosure** present and reasonable — keep.
- **F4. Abstract density.** If accepted, consider leading the abstract with the
  θ-oracle-collapse robustness finding rather than the raw 4.5× — it is the more
  novel and less attackable claim.

---

## Framing paragraph (reuse for camera-ready intro tweak-spots, or a resubmission cover letter — NOT a rebuttal, CIKM has none)
"We would emphasize that the paper's load-bearing claim is
a *deterministic* algebraic reduction of the click-noise variance surrogate on real
counts (Thm), not a task-performance claim; we are explicit that under PBM
mis-specification the choice of weight is a precision/estimand trade-off (Cochran
Q/df=3.83). The most novel finding is that the exact plug-in inverse-variance oracle
*collapses* on heavy-tailed logs (ARE 0.002) where count-only harmonic remains
robust — a result with direct implications for anyone applying IVW to production
click data. We have additionally run [paired CIs / adversarial (α,β) search /
plug-in oracle], which we are happy to include."

---

## Pre-notification TODO (before Aug 7, optional but high-leverage)
1. **[R1]** Stand up one Yahoo IPS-LTR ΔnDCG cell (uniform vs harmonic propensities).
   This is the single highest-value insurance against the most predictable rejection.
2. **[F2]** Reconcile the 1.683 vs 1.542 uniform-R̂ discrepancy; document the cause.
3. Keep `cikm_full.tex`'s reviewer-response tables handy — they are rebuttal-ready.
