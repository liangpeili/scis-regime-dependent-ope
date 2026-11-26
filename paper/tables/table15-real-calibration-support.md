# Table 15 — Leave-one-episode-out calibration support

| Episode | Type | Test n | Test symbols | Test h | Cal n (episodes) | Cal SPY/NVDA/TSLA | Cal h min/med/max | Wide B0/B1 | Narrow B0/B1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-02 | holiday | 2 | S+T | 138 | 31 (15) | 14/3/14 | 114/114/138 | 25/6 | 18/13 |
| 2026-07-10 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 25/6 | 18/13 |
| 2026-07-17 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 17/14 |
| 2026-07-24 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 19/12 |
| 2026-07-31 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 25/6 | 18/13 |
| 2026-08-07 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 17/14 |
| 2026-08-14 | weekend | 1 | S | 114 | 32 (15) | 14/3/15 | 114/114/138 | 25/7 | 18/14 |
| 2026-08-21 | weekend | 1 | T | 114 | 32 (15) | 15/3/14 | 114/114/138 | 25/7 | 19/13 |
| 2026-08-28 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 25/6 | 18/13 |
| 2026-09-04 | holiday | 2 | S+T | 138 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 18/13 |
| 2026-09-11 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 17/14 |
| 2026-09-18 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 18/13 |
| 2026-09-25 | weekend | 2 | S+T | 114 | 31 (15) | 14/3/14 | 114/114/138 | 24/7 | 18/13 |
| 2026-10-02 | weekend | 3 | N+S+T | 114 | 30 (15) | 14/2/14 | 114/114/138 | 24/6 | 17/13 |
| 2026-10-09 | weekend | 3 | N+S+T | 114 | 30 (15) | 14/2/14 | 114/114/138 | 24/6 | 17/13 |
| 2026-10-16 | weekend | 3 | N+S+T | 114 | 30 (15) | 14/2/14 | 114/114/138 | 24/6 | 18/12 |

Counts are observed symbol-events, not an effective-sample-size estimate; events within an episode remain clustered by date. S/N/T = SPYon/NVDAon/TSLAon. B0/B1 gives calibration counts by post-open branch. A finite 90% branch radius requires at least nine events in that branch. Wide is underpowered in 16/16 folds (B1 6–7); Narrow is underpowered in 0/16 folds. NVDA contributes only 2–3 calibration events per fold. The 114/138 h freeze durations are represented in every calibration fold, but duration support alone does not identify a cross-duration radius.
