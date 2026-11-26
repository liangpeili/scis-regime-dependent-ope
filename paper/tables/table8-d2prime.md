# Table 8 — Three-term decomposition, reported separately

The decomposition is not a theorem. The terms are along observed histories, not a uniform sup over beliefs; the third term is a diagnostic refinement for observation-type policies, not an independent bound term. The phases are anchor = last informative step before the freeze, freeze = pre+1..post−1, and post = first OPEN after the freeze.

## Panel A. $W_1$ estimation term (identified posterior vs naive delta; oracle uses latent $S_t$)

| $\tau$ (h) | Phase | $n$ | $W_1$ mean | $W_1$ SD | $W_1$ p90 | Oracle $W_1$ mean | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\tau=2$ | anchor | 128 | 3.96e-04 | 1.63e-19 | 3.96e-04 | 4.04e-04 | 0.980 |
| $\tau=2$ | freeze | 128 | 0.0076 | 0.0028 | 0.0105 | 0.0077 | 0.987 |
| $\tau=2$ | post | 128 | 3.96e-04 | 5.44e-07 | 3.97e-04 | 4.10e-04 | 0.966 |
| $\tau=81$ | anchor | 128 | 0.0027 | 7.56e-05 | 0.0028 | 0.0028 | 0.953 |
| $\tau=81$ | freeze | 128 | 0.0407 | 0.0160 | 0.0567 | 0.0406 | 1.003 |
| $\tau=81$ | post | 128 | 3.96e-04 | 2.70e-08 | 3.96e-04 | 4.09e-04 | 0.968 |

Notes. $W_1 = \mathbb{E}|\log S_t - \log P_t|$ under the identified posterior, i.e., $W_1$ between the identified posterior and the delta at $\log P_t$ (the naive treats $P_t$ as $S_t$); the oracle replaces $P_t$ by the simulator's latent $S_t$ for validation only. Ratio = identified mean / oracle mean. $n$ is the number of episodes (128 per cell); the $\tau=2$ anchor cell is numerically exact (SD $<10^{-15}$); the $\tau=81$ anchor SD is $7.6\times10^{-5}$. The freeze term grows with the freeze duration: 0.0076 at $\tau=2$ and 0.0407 at $\tau=81$.

## Panel B. Predictive-distribution TV information term

| $\tau$ (h) | Phase | $n$ | TV mean | TV SD | TV p90 | TV max | Step frac TV$\ge$0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\tau=2$ | anchor | 128 | 1.000 | 0.000 | 1.000 | 1.000 | 1.00 |
| $\tau=2$ | freeze | 128 | 0.646 | 0.039 | 0.686 | 0.721 | 0.50 |
| $\tau=81$ | anchor | 128 | 0.739 | 0.261 | 1.000 | 1.000 | 0.27 |
| $\tau=81$ | freeze | 128 | 0.766 | 0.066 | 0.838 | 0.911 | 0.00 |

Notes. TV between the identified and naive one-step predictives of $\log P_{t+1}$, evaluated at the observed history. `Step frac` is the fraction of pooled steps with TV $\ge 0.99$; episode-level maxima can be below 1 even when some steps are at 1. The post phase is not reported in the current JSON. At the $\tau=2$ anchor the two predictives are maximally different (TV = 1); during the freeze TV is 0.65 ($\tau=2$) and 0.77 ($\tau=81$).

## Panel C. Policy boundary/action-flip diagnostic

## Panel C.1 Flip probability

| Policy | $\tau$ (h) | $n$ | Flip mean | Flip SD | Flip p90 | Flip max | Mean $p_{\rm id}$ | Mean $p_{\rm naive}$ | Frac flip > 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide 2% | $\tau=2$ | 128 | 0.131 | 0.081 | 0.248 | 0.329 | 0.201 | 0.073 | 0.000 |
| Wide 2% | $\tau=81$ | 128 | 0.456 | 0.239 | 0.737 | 0.814 | 0.737 | 0.361 | 0.531 |
| Narrow 1% | $\tau=2$ | 128 | 0.214 | 0.092 | 0.334 | 0.433 | 0.460 | 0.247 | 0.000 |
| Narrow 1% | $\tau=81$ | 128 | 0.318 | 0.254 | 0.700 | 0.793 | 0.865 | 0.644 | 0.328 |

## Panel C.2 Boundary proximity (context)

| Policy | $\tau$ (h) | $n$ | Mean margin | Frac rebalance | Frac $\vert m\vert\le0.0025$ | Frac $\vert m\vert\le0.005$ | Frac $\vert m\vert\le0.01$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wide 2% | $\tau=2$ | 128 | -0.0126 | 0.062 | 0.031 | 0.109 | 0.281 |
| Wide 2% | $\tau=81$ | 128 | -0.0093 | 0.133 | 0.117 | 0.219 | 0.469 |
| Narrow 1% | $\tau=2$ | 128 | -0.0034 | 0.234 | 0.219 | 0.453 | 0.969 |
| Narrow 1% | $\tau=81$ | 128 | -0.0021 | 0.281 | 0.250 | 0.469 | 0.953 |

Notes. Flip = minimal-coupling probability that the rebalance decision at $t+1$ differs, $|p_{\rm id}-p_{\rm naive}|$, where $p_{\rm model} = P(|P_{t+1}/{\rm anchor}-1| > \theta)$ under that model's predictive; positions are from the observed trajectory (first-order). $p_{\rm id}$ and $p_{\rm naive}$ are the model-implied rebalance probabilities. Margin = $|P_t/{\rm anchor}-1| - \theta$ at decision steps; $m>0$ means the observed path rebalances. The boundary-proximity panel is context for the flip diagnostic, not a fourth term. The flip diagnostic is not a bound: it describes how close the policy is to its decision boundary under the two models.
