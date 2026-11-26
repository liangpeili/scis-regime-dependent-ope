# Table 3 — Row-level and trajectory-level coverage

## Panel A. L0: held-out-regime row-level coverage (90% split conformal)

| Regime | $n_{\rm test}$ | Naive global 90% cov. [95% CI] | width | Mondrian cov. | width ratio | Similarity cov. | width ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPEN | 46,848 | 0.9239 [0.9215, 0.9263] | 0.00614 | 0.9349 | 1.12× | 0.9033 | 0.83× |
| EXTENDED | 97,536 | 0.9301 [0.9285, 0.9317] | 0.00644 | 0.9378 | 1.08× | 0.9055 | 0.79× |
| WEEKEND | 92,928 | 0.8626 [0.8604, 0.8648] | 0.00517 | 0.9020 | 1.31× | 0.9020 | 1.31× |
| HOLIDAY | 18,432 | 0.8990 [0.8947, 0.9034] | 0.00597 | 0.9156 | 1.15× | 0.9156 | 1.15× |
| HALT | 1,536 | 0.8906 [0.8750, 0.9062] | 0.00598 | 0.9017 | 1.15× | 0.9017 | 1.15× |

Notes. Nominal coverage is 0.90 ($\alpha=0.1$); the 95% CI is a binomial normal
approximation on $n_{\rm test}$ rows and ignores within-trajectory correlation,
so it is a lower bound on the true uncertainty. Widths are mean interval widths.
Naive global = one pooled calibration radius (baseline). Mondrian = one radius
per calibration regime, unseen regime falls back to the largest calibration
radius. Similarity = Mondrian with the two-class fallback
{OPEN, EXTENDED} / {WEEKEND, HOLIDAY, HALT}.
Baseline aggregate `coverage_gap` = 0.0136; similarity fallback
= 0.0000; strategy-conditional = 0.00076.
Strategy-weighted gap: naive 0.0429, similarity
0.0392, strategy-conditional
0.0024. WEEKEND coverage by window (W1–W4):
W1 0.8550, W2 0.8782, W3 0.8706, W4 0.8467.
HALT calibration radius: 384 rows per fold, row-bootstrap SE
0.000296–
0.000315
(8.8–
9.3% of the radius).

## Panel B. L1: trajectory-level 90% conformal coverage and width

| Policy | Model | Score | $G$ within | $G$ LORO | $G_r$ within | $G_r$ LORO | width $G$ | width $G_r$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide | identified | raw | 0.9297 | 0.9238 | 0.9062 | 0.8719 | 0.04752 | 0.01585 |
| Wide | identified | studentized | 0.9180 | 0.9121 | 0.9012 | 0.8965 | 0.04158 | 0.01373 |
| Wide | identified | cqr | 0.9102 | 0.9082 | 0.9035 | 0.9004 | 0.03867 | 0.01229 |
| Wide | naive | raw | 0.9199 | 0.9102 | 0.9000 | 0.8738 | 0.04452 | 0.01823 |
| Wide | naive | studentized | 0.9121 | 0.9102 | 0.8949 | 0.8812 | 0.04890 | 0.01411 |
| Wide | naive | cqr | 0.9141 | 0.9004 | 0.8930 | 0.8836 | 0.04138 | 0.01342 |
| Wide | identified | plug-in normal | 0.8438 | -- | 0.8887 | -- | 0.03586 | 0.01283 |
| Wide | naive | plug-in normal | 0.6660 | -- | 0.7336 | -- | 0.02669 | 0.00895 |
| Narrow | identified | raw | 0.9199 | 0.9121 | 0.9059 | 0.8668 | 0.10866 | 0.04785 |
| Narrow | identified | studentized | 0.8984 | 0.8926 | 0.9066 | 0.8875 | 0.10100 | 0.03806 |
| Narrow | identified | cqr | 0.8945 | 0.8906 | 0.9082 | 0.9008 | 0.09946 | 0.03647 |
| Narrow | naive | raw | 0.9082 | 0.8906 | 0.8980 | 0.8801 | 0.12040 | 0.05784 |
| Narrow | naive | studentized | 0.9277 | 0.9258 | 0.8918 | 0.8801 | 0.13524 | 0.04994 |
| Narrow | naive | cqr | 0.9141 | 0.9062 | 0.8965 | 0.8699 | 0.11310 | 0.04547 |
| Narrow | identified | plug-in normal | 0.8359 | -- | 0.8820 | -- | 0.08656 | 0.03498 |
| Narrow | naive | plug-in normal | 0.6699 | -- | 0.7113 | -- | 0.07261 | 0.02774 |

Notes. $G$ is the full-window cumulative reward and $G_r$ the regime-segment
reward; within = calibrate and test on the same regime/window population; LORO =
leave-one-regime-out (for $G_r$) or leave-one-window-out (for $G$). Widths are
mean 90% interval widths: for conformal rows they are the LORO widths, whereas
for plug-in normal rows they are the full-window ($G$) and within-regime
($G_r$) widths, since the plug-in has no LORO variant. Plug-in normal =
$\pm z_{0.95}\,\hat{\rm sd}$, not a conformal interval. Identified uses
the regime-dependent observation model with G0 true-$S$ diagnostic parameters;
naive treats $P$ as $S$ with per-path $\kappa_P/\mu_P/\sigma_{\rm all}$.

## Panel C. L1: path-level paired bootstrap, identified $-$ naive

| Policy | Level | Score | identified $-$ naive | 95% CI |
| --- | --- | --- | --- | --- |
| Wide | full-window | studentized | +0.0020 | [-0.0312, +0.0352] |
| Wide | full-window | cqr | +0.0078 | [-0.0254, +0.0430] |
| Wide | full-window | plugin_normal | +0.1777 | [+0.1270, +0.2285] |
| Wide | regime | studentized | +0.0152 | [+0.0012, +0.0289] |
| Wide | regime | cqr | +0.0168 | [+0.0031, +0.0312] |
| Wide | regime | plugin_normal | +0.1551 | [+0.1379, +0.1730] |
| Narrow | full-window | studentized | -0.0332 | [-0.0684, +0.0020] |
| Narrow | full-window | cqr | -0.0156 | [-0.0508, +0.0215] |
| Narrow | full-window | plugin_normal | +0.1660 | [+0.1152, +0.2168] |
| Narrow | regime | studentized | +0.0074 | [-0.0074, +0.0223] |
| Narrow | regime | cqr | +0.0309 | [+0.0164, +0.0461] |
| Narrow | regime | plugin_normal | +0.1707 | [+0.1520, +0.1891] |

Notes. 2,000 path-level resamples, conditioning on the fitted calibration
quantiles; the interval quantifies test-path sampling only.
