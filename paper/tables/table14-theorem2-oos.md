# Table 14 — Three-sample out-of-sample check of Theorem 2

| Direction | Score | Mean $\hat\kappa$ | Mean 2B | Mean Cor 2.2 | Final coverage [mean CI95] | Mean width | Below 2B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2$\to$81 | raw | 0.8453 | 0.0000 | 0.0000 | 0.1046 [0.0977, 0.1116] | 0.00126 | 0/100 |
| 2$\to$81 | studentized | 0.1397 | 0.1989 | 0.3980 | 0.9099 [0.9029, 0.9168] | 0.02231 | 0/100 |
| 2$\to$81 | CQR | 0.6112 | 0.0000 | 0.0040 | 0.8886 [0.8831, 0.8940] | 0.01910 | 0/100 |
| 81$\to$2 | raw | 0.8448 | 0.0000 | 0.0000 | 1.0000 [1.0000, 1.0000] | 0.02369 | 0/100 |
| 81$\to$2 | studentized | 0.1484 | 0.1902 | 0.3892 | 0.9059 [0.8972, 0.9146] | 0.00115 | 0/100 |
| 81$\to$2 | CQR | 0.6052 | 0.0000 | 0.0046 | 0.9081 [0.8840, 0.9322] | 0.00199 | 0/100 |

Notes. Wide identified pipeline, 100 independent outer replicates. Each replicate uses
64 independently seeded calibration paths, the same number of
independent target-drift paths, and 256 final target-test paths. One path is drawn
per seed and window types cycle by seed. The radius uses calibration only; $\hat\kappa$ uses calibration and
the drift sample only; the final sample evaluates coverage and width without re-entering either calculation.
The bracket is a 95% interval for the mean final coverage across independent outer replicates, not a binomial
interval over pooled events. `Below 2B` counts replicates whose noisy final-sample coverage falls below the
2B bound; it is an empirical tightness diagnostic, not an exact PAC-violation frequency. Bounds
can be empty (zero) for any score. Width is in target-return units. Both $\tau=2\to81$ and $\tau=81\to2$ are shown.
