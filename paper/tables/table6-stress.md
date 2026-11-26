# Table 6 — I5 stress variant (unfreeze jumps)

| Environment | Tier | open-only cov. (SE) | open width | cross-event cov. (SE) | cross width | width ratio | $\\hat\\gamma$ [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\\gamma=0$ | oracle | 0.906 (0.007) | 0.1635 | 0.901 (0.008) | 0.1606 | 0.98× | 0.00 [0.00, 2.14] |
| $\\gamma=0$ | estimated | 0.895 (0.008) | 0.1576 | 0.900 (0.008) | 0.1605 | 1.02× | 1.72 [0.00, 2.85] |
| $\\gamma=4$ | oracle | 0.864 (0.009) | 0.1635 | 0.902 (0.008) | 0.1797 | 1.10× | 3.97 [3.15, 4.70] |
| $\\gamma=4$ | estimated | 0.843 (0.009) | 0.1573 | 0.902 (0.008) | 0.1820 | 1.16× | 4.31 [3.58, 4.98] |

Notes. 90% prediction intervals, 5-fold out-of-fold (fold = seed mod 5), 128
seeds × 4 windows = 512 paths, 1,536 double-endpoint events (τ=2/65/81, 512 each). Nominal
0.90 ($\alpha=0.1$). Coverage SEs are binomial over the 1,536 event predictions
and ignore within-path and across-fold dependence, so they are lower bounds.
`open-only` uses ordinary OPEN→OPEN one-step transitions;
`cross-event` uses $\operatorname{Var}(r_e)=v_{0e}+\theta w_e$ with
$w_e=\sigma^2(1-e^{-\tau_e/\tau_0})$, $\theta=\gamma^2$, and a profile
likelihood CI. The γ̂ CI is the full-data MLE profile interval, not a coverage
interval. γ=4 CIs are
[3.15, 4.70]
(oracle) and
[3.58, 4.98]
(estimated); γ=0 CIs are
[0.00, 2.14]
and
[0.00, 2.85]:
the intervals do not overlap.

Identification diagnostics: the ordinary OPEN→OPEN one-step kernel is
γ-free by construction; the twin open-increment KS statistic is
0.00047
(p=1.000, descriptive and
dependent samples). The de-freeze $z^2$ variance ratio (γ=4/γ=0) is
1.233,
paired path-permutation p=0.0005.
