# Table 13 — Second synthetic sensor-POMDP testbed

## Panel A. Return prediction error and interval coverage

| Model | Mean absolute return error (G) | Plug-in coverage | Raw full-window coverage | Raw LORO outage coverage |
| --- | --- | --- | --- | --- |
| identified transition | 4.805 | 0.906 | 0.883 | 0.758 |
| fitted transition | 5.850 | 0.727 | 0.914 | 0.812 |

Notes. Mean absolute realized-return prediction error is the mean over evaluation
paths of the absolute difference between the 64-rollout predictive mean and the
realized target return $G$; it is not a value-bias estimate. These are descriptive
point estimates without paired uncertainty intervals. Calibration and evaluation
use 128 independent paths each. Truth and both predictive models use the same
target-action rule and identified-belief filter, with action realizations following
their simulated observation histories. Both use the known control coefficient
$B=0.35$. The identified-transition predictor also knows $A$, $\sigma$, and the
observation model; the fitted-transition comparator replaces only the autonomous
rollout AR(1) coefficient and residual scale, estimated from burn-in data. The
JSON retains `naive` as its internal key.
This is not an observation-filter comparison or evidence of uniform model superiority.

## Panel B. Raw-score coverage by observation regime

| Regime | Known-trans. within | Known-trans. LORO | Fitted-trans. within | Fitted-trans. LORO |
| --- | --- | --- | --- | --- |
| normal | 0.938 | 0.953 | 0.906 | 0.992 |
| degraded | 0.859 | 0.805 | 0.898 | 0.836 |
| outage | 0.859 | 0.758 | 0.906 | 0.812 |

Notes. LORO calibrates on the other two regimes. The outage boundary is retained
rather than pooled away.

## Panel C. Post-decision intervention-branch coverage (identified, raw)

| Rule | Overall coverage | Branch 0 coverage | Branch 1 coverage | Mean width (return units) |
| --- | --- | --- | --- | --- |
| marginal | 0.883 | 0.966 | 0.692 | 14.851 |
| predecision | 0.891 | 0.978 | 0.692 | 16.288 |
| branch_union | 0.953 | 0.978 | 0.897 | 28.057 |

Notes. Calibration has 43 intervention events among 128 paths;
evaluation has 39 among 128 (the other 89 are branch 0).
Marginal intervention-branch SE is 0.074;
branch-union SE is 0.049.
The pre-decision rule uses three groups from the sign and magnitude of the last
burn-in observation. Branch-union uses the larger calibration-side branch threshold,
so the test-time branch label is not needed to form the interval. Mean width is the
interval width in target-return (reward) units, not a coverage probability; the
1.89$\times$ figure in the main text is the branch-union/marginal width ratio.
