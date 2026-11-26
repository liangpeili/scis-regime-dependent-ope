# Table 9 — Non-Markov Brownian bridge stress

## Panel A. Coverage and width

| Environment | Tier | open-only cov. (SE) | open width | cross-event cov. (SE) | cross width | width ratio | $\tau$-stratified cov. (SE) | $\tau$-stratified width ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\nu_{\rm mult}=0$ placebo | oracle | 0.897 (0.008) | 0.1618 | 0.897 (0.008) | 0.1637 | 1.01× | 0.901 (0.008) | 1.01× |
| $\nu_{\rm mult}=0$ placebo | estimated | 0.882 (0.008) | 0.1576 | 0.900 (0.008) | 0.1662 | 1.05× | 0.900 (0.008) | 1.05× |
| $\nu_{\rm mult}=0.6$ bridge | oracle | 0.818 (0.010) | 0.1618 | 0.900 (0.008) | 0.2047 | 1.27× | 0.902 (0.008) | 1.27× |
| $\nu_{\rm mult}=0.6$ bridge | estimated | 0.806 (0.010) | 0.1571 | 0.900 (0.008) | 0.2072 | 1.32× | 0.901 (0.008) | 1.31× |

Notes. 90% prediction intervals, 5-fold out-of-fold (fold = seed mod 5), 128
seeds × 4 windows = 512 paths, 1,536 double-endpoint events ($\tau=2/65/81$, 512 each).
Nominal 0.90 ($\alpha=0.1$). Coverage SEs are binomial over the 1,536 event
predictions and ignore within-path and across-fold dependence, so they are
lower bounds. `open-only` uses ordinary OPEN→OPEN one-step transitions and the
Markov $\tau$-step variance $v_{0e}$; `cross-event` uses
$\operatorname{Var}(r_e)=v_{0e}+\theta\sigma_e^2\,\mathrm{gap}_e$ with
$\theta$ fitted by Gaussian MLE; `$\tau$-stratified` uses a separate conformal
quantile per freeze length and no functional form in $\tau$.

## Panel B. Bridge variance component

| Environment | Tier | $\hat\theta$ | 95% CI | $\hat\nu/\sigma$ | 95% CI |
| --- | --- | --- | --- | --- | --- |
| $\nu_{\rm mult}=0$ placebo | oracle | 0.0282 | [0.000, 0.097] | 0.168 | [0.000, 0.311] |
| $\nu_{\rm mult}=0$ placebo | estimated | 0.0703 | [0.008, 0.138] | 0.265 | [0.092, 0.372] |
| $\nu_{\rm mult}=0.6$ bridge | oracle | 0.4912 | [0.397, 0.595] | 0.701 | [0.630, 0.771] |
| $\nu_{\rm mult}=0.6$ bridge | estimated | 0.5360 | [0.443, 0.639] | 0.732 | [0.665, 0.799] |

Notes. $\theta=\nu_{\rm mult}^2\,\mathbb E[e^{2\zeta}]$; the design value
is $\theta_{\rm true}=0.457$ from the realized
event-level draws. The $\nu=0$ placebo has $\theta=0$ by construction. The
oracle-tier placebo interval contains zero; the estimated-tier placebo has a
small positive bias from fitting $T$ and $\sigma$ on open transitions, as in
the $\gamma=0$ placebo of Table 6.

## Panel C. Coverage by freeze length

| Environment | Tier | $\tau$ | $n$ | open cov. | open width | cross cov. | cross width | $\tau$-strat cov. | $\tau$-strat width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\nu_{\rm mult}=0$ placebo | oracle | 2 | 512 | 0.891 | 0.0488 | 0.893 | 0.0492 | 0.900 | 0.0497 |
| $\nu_{\rm mult}=0$ placebo | oracle | 65 | 512 | 0.891 | 0.2089 | 0.891 | 0.2113 | 0.902 | 0.2188 |
| $\nu_{\rm mult}=0$ placebo | oracle | 81 | 512 | 0.910 | 0.2277 | 0.908 | 0.2305 | 0.900 | 0.2239 |
| $\nu_{\rm mult}=0$ placebo | estimated | 2 | 512 | 0.887 | 0.0479 | 0.902 | 0.0502 | 0.898 | 0.0499 |
| $\nu_{\rm mult}=0$ placebo | estimated | 65 | 512 | 0.865 | 0.2034 | 0.893 | 0.2145 | 0.902 | 0.2213 |
| $\nu_{\rm mult}=0$ placebo | estimated | 81 | 512 | 0.895 | 0.2213 | 0.906 | 0.2339 | 0.900 | 0.2273 |
| $\nu_{\rm mult}=0.6$ bridge | oracle | 2 | 512 | 0.820 | 0.0488 | 0.895 | 0.0597 | 0.904 | 0.0612 |
| $\nu_{\rm mult}=0.6$ bridge | oracle | 65 | 512 | 0.797 | 0.2089 | 0.889 | 0.2641 | 0.900 | 0.2800 |
| $\nu_{\rm mult}=0.6$ bridge | oracle | 81 | 512 | 0.838 | 0.2277 | 0.916 | 0.2903 | 0.900 | 0.2744 |
| $\nu_{\rm mult}=0.6$ bridge | estimated | 2 | 512 | 0.812 | 0.0479 | 0.902 | 0.0607 | 0.900 | 0.0610 |
| $\nu_{\rm mult}=0.6$ bridge | estimated | 65 | 512 | 0.791 | 0.2029 | 0.879 | 0.2672 | 0.902 | 0.2879 |
| $\nu_{\rm mult}=0.6$ bridge | estimated | 81 | 512 | 0.814 | 0.2206 | 0.920 | 0.2935 | 0.900 | 0.2708 |

Notes. The open-only deficit is present at every $\tau$; it is not a single
long-freeze cell. The $\tau$-stratified calibrator recovers within-$\tau$
marginal coverage without assuming the bridge variance shape.

## Panel D. Same endpoints, different bridge volatility

| Quantity | Value |
| --- | --- |
| Same-endpoint events | 64 |
| Max abs pool-price difference | 0.0 |
| Observed paths bit-identical | True |
| Mean latent QV at $\nu_e$ | 1.6023e-03 |
| Mean latent QV at $2\nu_e$ | 6.1677e-03 |
| Empirical QV ratio | 3.85 |
| Theoretical QV ratio | 4.00 |

Notes. Each selected event is re-simulated twice with the same realized
endpoints, the same standard Brownian bridge shape, and volatilities $\nu_e$
and $2\nu_e$. Freeze interiors are masked by the observation map, so the
observed pool-price paths are bit-identical; only the latent quadratic
variation changes. The empirical QV ratio is close to the theoretical
$2^2=4$. This is the empirical counterpart of the Theorem 4 construction, not
a coverage experiment: with endpoints pinned, no observable-data estimator can
distinguish the two bridge laws.

## Panel E. Coverage against the realized event-level volatility

| $\nu_e$ quartile | $n$ | $\nu$ min | $\nu$ max | open cov. | cross cov. | open width | cross width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 384 | 0.0010 | 0.0034 | 0.859 | 0.919 | 0.1210 | 0.1534 |
| 1 | 384 | 0.0034 | 0.0048 | 0.844 | 0.917 | 0.1451 | 0.1839 |
| 2 | 384 | 0.0048 | 0.0064 | 0.797 | 0.904 | 0.1654 | 0.2093 |
| 3 | 384 | 0.0064 | 0.0293 | 0.773 | 0.859 | 0.2157 | 0.2724 |

Notes. The groups use the simulation-only $\nu_e$, which is not observed by
either method; they are a dose-response diagnostic, not a deployable
stratification. Open-only coverage falls monotonically across all four
quartiles; cross-event calibration recovers marginal coverage but not
coverage conditional on the latent $\nu_e$ (the highest quartile remains below
nominal). The latter is expected: $\nu_e$ is not in the observed information
set.

## Bridge law

Brownian bridge pinned at the latent pre/post anchors. The post endpoint is
drawn from $N(m_+, v_{\rm ar}+\nu_e^2\,\mathrm{gap})$, with
$\nu_e=\sigma\nu_{\rm mult}e^{\zeta_e}$,
$\zeta_e\sim N(0,0.35^2)$ clipped to
$[-1.0,1.0]$, and
$\nu_{\rm mult}=0.60$. The interior is unobserved, so the
bridge volatility's observable footprint is the endpoint law; the same-endpoint
panel isolates its path-functional consequence.
