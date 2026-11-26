# P1 tables — provenance

Generated 2026-10-21 for M3. Do not hand-edit these files; regenerate them from
the JSON inputs.

## Generator

```
../../.venv-c1/bin/python code/build_tables.py
```

The generator is paper tooling, not part of the C1 experiment loop; it is
versioned in `code/` and the command is collected in Appendix C at M7.

## Inputs

- `results/coverage_gap_naive_global.json` — naive global baseline, reproduced
  from commit `c610494`.
- `results/coverage_gap_exp1_mondrian.json`
- `results/coverage_gap.json` (similarity fallback, kept default)
- `results/coverage_gap_strategy.json`
- `results/trajectory_value_check.json`
- `results/trajectory_value_check_obs_h.json`
- `results/trajectory_value_check_obs_h_z25.json`
- `results/trajectory_value_check_obs_h_z40.json`
- `results/trajectory_value_check_obs_h_sigma_open.json`
- `results/policy_value_check.json`
- `results/observable_h_check.json`
- `results/stress_variant.json`
- `results/stress/misspecification_suite.json`
- `results/semi_synthetic_calibration.json`
- `results/real_data_replay.json` (main 33-event case)
- `results/real_data_replay_53.json` (nine-month 53-event sensitivity)
- `results/real_slice_9m_summary.json` (window audit)
- `results/general_pomdp_check.json` (general sensor-POMDP validation)
- `results/theorem2_bounds.json` (per-fold bounds and 100-replicate three-sample check)

## Outputs

| File | Content |
| --- | --- |
| `table3-coverage.md` | L0 held-out-regime coverage; L1 trajectory coverage/width; paired bootstrap |
| `table4-branches.md` | L2 post-decision branches, Wilson branch CIs, union dominance, pre-decision Mondrian |
| `table5-h.md` | L3 observable `h`, uncertainty, coverage sensitivity |
| `table6-stress.md` | I5 stress variant and identification diagnostics |
| `table7-q1q3.md` | Q1–Q3 summary |
| `table8-d2prime.md` | Three-term decomposition: $W_1$ term, predictive TV term, action-flip diagnostic, boundary proximity |
| `table9-bridge.md` | Non-Markov Brownian bridge stress: coverage, variance component, same-endpoint QV diagnostic |
| `table10-misspec.md` | Misspecification suite: bias and coverage under AR(2) noise, heavy-tailed jumps, regime-label noise, hidden stale state |
| `table11-semi-synthetic.md` | Semi-synthetic calibration from the real slice |
| `table12-real-data.md` | Real-path usage check: observation-map parameters, event coverage/width/MAE, reward decomposition, score and branch diagnostics |
| `table13-general-pomdp.md` | Second synthetic sensor-POMDP return prediction error, regime transfer, and post-decision branch recovery |
| `table14-theorem2-oos.md` | Three independent samples for Theorem 2 radius/drift/final evaluation, coverage, bounds, width, and tightness |
| `table15-real-calibration-support.md` | Leave-one-episode-out event, symbol, duration, and branch calibration counts |
| `m3-derived.json` | every derived value plus a slug-to-source map |
| `number-sources.md` | readable slug-to-input-path table (data or code) with input/script hashes |

`m3-derived.json` is the audit object for M5: prose `[AUDIT: slug]` markers
resolve to `values[slug]`, and `sources[slug]` gives the JSON path or the
computation.

## Figures

```
../../.venv-c1/bin/python code/make_figures.py
```

Writes `paper/figures/fig3-l0-coverage.{pdf,png}`, `fig4-l1-boundary`,
`fig5-l2-branches`, `fig6-l3-h`, and `fig7-general-pomdp` at 300 dpi. Every plotted series is read
from the same JSON inputs; no value is hand-entered.

## Placement in the 2026-11-09 TNNLS remediation candidate

The 15-page page budget (after restoring IEEEtran defaults) required moving two wide
result floats to the supplementary PDF. Only their **placement** changed — all numbers,
panels, captions, notes, negative results, and caveats are unchanged, and the evidence
bindings in `number-sources.md` are untouched (same result files, same script hashes).

| Generated LaTeX | Source Markdown | 2026-11-09 placement |
| --- | --- | --- |
| `tables/table4.tex` | `table4-branches.md` (L2) | Supplement Appendix E, **Appendix Table B12** |
| `tables/table5.tex` | `table5-h.md` (L3) | Supplement Appendix E, **Appendix Table B13** |
| `tables/table6.tex` | `table6-stress.md` (I5 stress) | **Kept in main text** as TABLE I (main p11) |

The generated `table4.tex`/`table5.tex` still exist and are `\input` by
`paper/latex/appendix-e.tex` (`\setcounter{table}{11}` → B12/B13); they keep
`\label{tab:4}`/`\label{tab:5}` so Appendix Table B8 (`table7.tex`, Q1→B13 / Q2→B12)
resolves locally in the supplement. The main text uses 3× "Appendix Table B12"
+ 1× "Appendix Table B13" hardcoded pointers (the paper's existing convention;
the main text has no `xr`). See `drafts/preflight-2026-11-09-ieee-author-center-remediation.md`.
