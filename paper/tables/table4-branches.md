# Table 4 — Branch-conditional coverage (L2)

## Panel A. Wide τ=2, post-decision branches (main seed, 128 test events)

| Score | hold n | hold cov. | hold union | rebal n | rebal cov. | rebal union | rebal gain (SE) | overall width ratio | trivial bound | taxonomy hold / rebal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 104 | 0.9712 | 1.0000 | 24 | 0.6667 | 0.9167 | +0.250 (0.088) | 3.96× | 0.542 | cost_only / clear_gain |
| studentized | 104 | 0.9712 | 1.0000 | 24 | 0.6667 | 0.9583 | +0.292 (0.093) | 3.97× | 0.542 | cost_only / clear_gain |
| cqr | 104 | 0.9423 | 1.0000 | 24 | 0.7500 | 0.9167 | +0.167 (0.076) | 3.37× | 0.500 | clear_gain / clear_gain |

Notes. Nominal 0.90 ($\alpha=0.1$). The paired gain's standard error (one SE,
not a 95% interval) is in parentheses; a single-cell coverage SE is about
0.05–0.10 for the 24-event branch and 0.02–0.05 for the 104-event branch.
`Trivial bound` is
$1-(1-\hat c)/\hat q_b$ with $\hat c$ the measured overall marginal coverage
and $\hat q_b=24/128$; it uses the observed $\hat c$, not the nominal level, so
it is an empirical identity, not a guarantee. The branch-union threshold is
$\max(\hat q_{\rm hold},\hat q_{\rm rebalance})$ on the calibration side;
`post_rebalance` is a post-decision label and is never used at prediction time.
`width ratio` is overall union width / overall marginal width.

## Panel B. Branch-union aggregate across 56 method×direction blocks

| Statistic | Value |
| --- | --- |
| Coverage dominance (union ≥ marginal) | 56/56 blocks, 112/112 branch cells |
| Nonempty calibration branches | 112/112 branch cells; smallest $n_b$ = 16 |
| Infinite branch thresholds ($\hat q_b=\infty$) | 0 cells |
| Taxonomy | cost_only 44, clear_gain 25, gain_but_fails 10, fails_no_clear_gain 4, ambiguous 29 |
| Wide 81→2 rebalance, identified studentized | 0.667 → 0.750 (width 1.27×, ambiguous) |
| Wide 81→2 rebalance, identified CQR | 0.750 → 0.875 (width 1.67×, ambiguous) |

Notes. Taxonomy uses normal-approximation 95% intervals per branch; the cells are
descriptive, not simultaneous. `cost_only` = marginal coverage already clearly
at or above nominal; `clear_gain` = a gain whose 95% interval excludes zero,
with union still possibly below nominal; `gain_but_fails` = a gain whose 95%
interval excludes zero, but union still clearly below nominal;
`fails_no_clear_gain` = below nominal with no gain whose 95% interval excludes
zero; `ambiguous` otherwise.

## Panel C. Pre-decision Mondrian cannot isolate the post branch (text summary)

Wide τ=2, identified raw/studentized, three seeds, `onset_action`/`onset_side`:
post-rebalance coverage ranges 0.583–0.708 across
12 cells, versus 0.971
on hold and 0.667 on
rebalance under the unstratified marginal pipeline. Pre-decision stratification
cannot recover the post-decision branch because the branch label is not in the
pre-decision $\sigma$-field.

## Panel D. Branch-level Wilson intervals (Wide τ=2)

| Score | Branch | $n$ | Marginal [95% CI] | Union [95% CI] |
| --- | --- | --- | --- | --- |
| raw | hold | 104 | 0.971 [0.919, 0.990] | 1.000 [0.964, 1.000] |
| raw | rebalance | 24 | 0.667 [0.467, 0.820] | 0.917 [0.742, 0.977] |
| studentized | hold | 104 | 0.971 [0.919, 0.990] | 1.000 [0.964, 1.000] |
| studentized | rebalance | 24 | 0.667 [0.467, 0.820] | 0.958 [0.798, 0.993] |
| cqr | hold | 104 | 0.942 [0.880, 0.973] | 1.000 [0.964, 1.000] |
| cqr | rebalance | 24 | 0.750 [0.551, 0.880] | 0.917 [0.742, 0.977] |

Notes. Wilson score 95% intervals for branch coverage of the fitted score,
conditional on the fitted quantile; they do not include calibration
uncertainty. One branch cell at a time; they are not simultaneous. The
rebalance cells have 24 test events, so their marginal intervals are about
0.35 wide. All three rebalance marginal intervals lie entirely below 0.90:
the deficit is directional even at this sample size, but its magnitude is not
pinned down. Every rebalance union interval contains 0.90: union coverage is
compatible with nominal but does not establish nominal coverage. Panel B's
aggregate taxonomy retains normal-approximation intervals for comparability
across all 56 blocks; Panel D uses Wilson intervals because the normal
approximation is unreliable at $n=24$.
