# Table 12 — Real-path usage check in the liquid-regime case

Sample: 33 A-layer symbol-events over 16 independent freeze episodes (14 weekend + 2 holiday), 2026-07-01T00:00Z/2026-10-20T00:00Z. Post-hoc regime-conditional case, not a pre-registered sample.

## A. Observation-map parameters

| Symbol | Events | sigma_open | sigma_closed | open_sd | stale_phi |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPYon | 15 | 0.00205 | 0.00174 | 0.00405 | 0.908 |
| NVDAon | 3 | 0.00477 | 0.00152 | 0.00320 | 0.920 |
| TSLAon | 15 | 0.00721 | 0.00337 | 0.00588 | 0.878 |

## B. Event-level conformal coverage (alpha = 0.1, leave-one-episode-out, raw score)

| Target | Identified cov. | Identified width | Identified MAE | Naive cov. | Naive width | Naive MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Freeze | 0.909 | 0.00033 | 0.00006 | 0.909 | 0.00191 | 0.00045 |
| Post-open OPEN | 0.909 | 0.00430 | 0.00074 | 0.909 | 0.00744 | 0.00105 |
| Post-open 2 h | 0.909 | 0.00236 | 0.00032 | 0.909 | 0.00252 | 0.00032 |
| Event window | 0.909 | 0.00402 | 0.00105 | 0.909 | 0.00414 | 0.00095 |
| Total main | 0.909 | 0.00535 | 0.00122 | 0.909 | 0.00559 | 0.00104 |

Coverage is 30/33 in every cell; the cluster-bootstrap 95% CI is [0.81, 1.00] for the total and event windows. Widths and MAE are in units of $V_0$.

## C. Modeled replay reward decomposition (Wide policy, mean truth, fraction of V0)

| Window | Fee | IL | Cost | Net |
| --- | ---: | ---: | ---: | ---: |
| Freeze | +0.000056 | -0.000019 | +0.000000 | +0.000038 |
| Post-open 2 h | +0.000003 | -0.000107 | +0.000220 | -0.000323 |
| Event window | +0.000067 | -0.000216 | +0.000251 | -0.000399 |
| Total main | +0.000116 | -0.000653 | +0.001398 | -0.001934 |

P-marked model decomposition; not feeGrowthGlobal or deployed-LP validation.

## D. Score sensitivity (coverage / width)

| Target | Score | Identified | Naive |
| --- | --- | ---: | ---: |
| freeze | raw | 0.909 / 0.00033 | 0.909 / 0.00191 |
| freeze | studentized | 0.909 / 0.00041 | 0.909 / 0.00123 |
| freeze | cqr | 0.909 / 0.00038 | 0.909 / 0.00200 |
| total_main | raw | 0.909 / 0.00535 | 0.909 / 0.00559 |
| total_main | studentized | 0.909 / 0.00407 | 0.909 / 0.00388 |
| total_main | cqr | 0.909 / 0.00447 | 0.909 / 0.00399 |

## E. Branch and target-shift diagnostics

| Diagnostic | Identified | Naive |
| --- | ---: | ---: |
| Wide branch | B=1 n=7; union n.i. | B=1 n=7; union n.i. |
| Narrow branch | B=1 n=14; union 0.970; width x1.28 | B=1 n=14; union 0.970; width x1.08 |
| Leave-one-symbol-out RAW (TSLA) | 0.267 | 0.267 |
| Leave-one-symbol-out CQR (TSLA) | 0.933 | 0.933 |
| Within-symbol (SPY/TSLA macro) | 0.933 | 0.933 |
| In-sample 2h->24h target shift | 0.879 | 0.848 |

n.i. = not identifiable at 90% (fewer than nine branch calibration events). The symbol-conditional row is a macro average over SPY and TSLA; NVDA has 3 events and is excluded from a 90% within-symbol radius. Duration rows are in-sample target shifts over the same events, not out-of-sample transfer; reverse 1.000 reflects the wider 24h radius, not a symmetric guarantee.

## Notes

- Post-hoc liquid-regime window (2026-07-01T00:00Z/2026-10-20T00:00Z); not a pre-registered sample. Results are regime-conditional and descriptive.
- Truth is deterministic replay of the target policy on the observed pool-price path with a P-marked reward; it is not a deployed-policy counterfactual and not a causal logged OPE estimate.
- Identified/naive are model-based OPE pipelines; the model is fit to observable P/oracle-basis data, not to latent S.
- open_sd is estimated from event-level boundary basis values, not a full Chainlink round path; treat it as a first-pass estimate.
- CLOSED_LIVE is a separate observation branch with its own sigma_closed; its noise scale is derived from open_sd through the closed/open volatility ratio, not estimated independently.
- NVDAon uses the NVDA-USD (24/5) proxy feed and contributes only 3 of the 33 A-layer events.
- Bootstrap CIs resample the 16 calendar dates, not symbol-events; they account for cross-symbol dependence but are wide.
- Observation-map parameters are estimated once on all events; leave-one-episode-out folds recalibrate only the conformal radius, so this is not nested cross-validation.
- Duration target-shift rows calibrate and evaluate on the same event set; they are not out-of-sample transfer results.
- Thin events are retained for coverage, width, and transfer analyses; the sensitivity check splits coverage over all 33 events (16 thin/17 non-thin). No economic-value estimate is reported.
