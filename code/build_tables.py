#!/usr/bin/env python3
"""Build the M3 tables (§6–§7) from results/*.json.

Every printed number is read or computed from a JSON file under results/ and is
also written to paper/tables/m3-derived.json with a source path.  This is paper
tooling, not part of the C1 experiment loop; the command is recorded in
paper/tables/README.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'paper' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

REG = {'0': 'OPEN', '1': 'EXTENDED', '2': 'WEEKEND', '3': 'HOLIDAY', '4': 'HALT'}
REG_ORDER = ['0', '1', '2', '3', '4']
METH = ['identified_raw', 'identified_studentized', 'identified_cqr',
        'naive_raw', 'naive_studentized', 'naive_cqr']

def load(name):
    return json.loads((RES / name).read_text())

base = load('coverage_gap_naive_global.json')
mon = load('coverage_gap_exp1_mondrian.json')
sim = load('coverage_gap.json')
strat = load('coverage_gap_strategy.json')
traj = load('trajectory_value_check.json')
traj_specs = {
    'truth (G0)': traj,
    'observable $h$ (z=3)': load('trajectory_value_check_obs_h.json'),
    'observable $h$, z=2.5': load('trajectory_value_check_obs_h_z25.json'),
    'observable $h$, z=4': load('trajectory_value_check_obs_h_z40.json'),
    'observable $h$, $\\sigma$ from OPEN': load('trajectory_value_check_obs_h_sigma_open.json'),
}
pol = load('policy_value_check.json')
obs = load('observable_h_check.json')
stress = load('stress_variant.json')
miss_path = RES / 'stress' / 'misspecification_suite.json'
miss = json.loads(miss_path.read_text()) if miss_path.exists() else None
general = load('general_pomdp_check.json')
theorem2 = load('theorem2_bounds.json')

V = {}
S = {}

def put(slug, value, source, fmt=None):
    if not source.startswith(('computed:', 'results/', 'code/')):
        source = 'results/' + source
    V[slug] = value
    S[slug] = {'json': source, 'format': fmt}
    return value


for policy, methods in theorem2['policies'].items():
    for method, rec in methods.items():
        put(f'theorem2.{policy}.{method}.summary', rec['summary'],
            f'theorem2_bounds.json:policies.{policy}.{method}.summary; code/theorem2_bounds.py', 'f4')


def f4(x):
    return f'{x:.4f}'

def f5(x):
    return f'{x:.5f}'

def f2(x):
    return f'{x:.2f}'

def pct(x, nd=1):
    return f'{100*x:.{nd}f}'

def ci_binom(p, n, z=1.96):
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval; used for the 24-event branch cells."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]

def md_table(headers, rows, align=None):
    align = align or ['---'] * len(headers)
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join(align) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)

# ---------------------------------------------------------------------------
# L0
# ---------------------------------------------------------------------------
l0_rows = []
for r in REG_ORDER:
    n = base['n_test_per_regime'][r]
    b = base['heldout_regime_coverage'][r]
    bw = base['heldout_regime_width'][r]
    lo, hi = ci_binom(b, n)
    put(f'l0.{r}.naive_cov', b, f'coverage_gap_naive_global.json:heldout_regime_coverage.{r}', 'f4')
    put(f'l0.{r}.naive_width', bw, f'coverage_gap_naive_global.json:heldout_regime_width.{r}', 'f5')
    put(f'l0.{r}.n_test', n, f'coverage_gap_naive_global.json:n_test_per_regime.{r}')
    put(f'l0.{r}.ci95', [lo, hi], 'computed: binomial normal approximation from coverage_gap_naive_global.json naive_cov and n_test')
    m = mon['heldout_regime_coverage'][r]
    mw = mon['heldout_regime_width'][r]
    s = sim['heldout_regime_coverage'][r]
    sw = sim['heldout_regime_width'][r]
    put(f'l0.{r}.mondrian_cov', m, f'coverage_gap_exp1_mondrian.json:heldout_regime_coverage.{r}', 'f4')
    put(f'l0.{r}.mondrian_width_ratio', mw / bw, f'computed: coverage_gap_exp1_mondrian.json widths / coverage_gap_naive_global.json widths', 'f3')
    put(f'l0.{r}.similarity_cov', s, f'coverage_gap.json:heldout_regime_coverage.{r}', 'f4')
    put(f'l0.{r}.similarity_width_ratio', sw / bw, 'computed: coverage_gap.json widths / naive baseline widths', 'f3')
    l0_rows.append([
        REG[r], f'{n:,}', f'{b:.4f} [{lo:.4f}, {hi:.4f}]', f5(bw),
        f'{m:.4f}', f'{mw/bw:.2f}×', f'{s:.4f}', f'{sw/bw:.2f}×',
    ])

put('l0.gap.naive', base['coverage_gap'], 'coverage_gap_naive_global.json:coverage_gap', 'f4')
put('l0.gap.similarity', sim['coverage_gap'], 'coverage_gap.json:coverage_gap', 'f4')
put('l0.gap.strategy_conditional', strat['coverage_gap'], 'coverage_gap_strategy.json:coverage_gap', 'f5')
put('l0.strategy_gap.naive', base['strategy_coverage_gap'], 'coverage_gap_naive_global.json:strategy_coverage_gap', 'f4')
put('l0.strategy_gap.similarity', sim['strategy_coverage_gap'], 'coverage_gap.json:strategy_coverage_gap', 'f4')
put('l0.strategy_gap.strategy_conditional', strat['strategy_coverage_gap'], 'coverage_gap_strategy.json:strategy_coverage_gap', 'f4')
put('l0.random_cov', base['random_split_coverage'], 'coverage_gap_naive_global.json:random_split_coverage', 'f4')
put('l0.random_width', base['random_split_width'], 'coverage_gap_naive_global.json:random_split_width', 'f5')
put('l0.weekend.windows', {w: d['coverage'] for w, d in base['heldout_regime_by_window']['2'].items()},
    'coverage_gap_naive_global.json:heldout_regime_by_window.2.*.coverage', 'f4')
put('l0.halt.boot_se_range', [min(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values()),
                              max(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values())],
    'coverage_gap_naive_global.json:halt_calibration_radius_bootstrap.*.boot_se', 'e3')
put('l0.halt.rel_se_range', [min(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values()),
                             max(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values())],
    'coverage_gap_naive_global.json:halt_calibration_radius_bootstrap.*.rel_se_point', 'pct')
put('l0.halt.n_cal', 384, 'coverage_gap_naive_global.json:halt_calibration_radius_bootstrap.*.n_cal_halt')
weekend_strat = base['heldout_strategy_coverage']['2']
put('l0.weekend.strategy_range', [min(weekend_strat.values()), max(weekend_strat.values())],
    'coverage_gap_naive_global.json:heldout_strategy_coverage.2.*', 'f4')

table3a = md_table(
    ['Regime', '$n_{\\rm test}$', 'Naive global 90% cov. [95% CI]', 'width',
     'Mondrian cov.', 'width ratio', 'Similarity cov.', 'width ratio'],
    l0_rows)

# ---------------------------------------------------------------------------
# L1 coverage and widths
# ---------------------------------------------------------------------------
L1_ROWS = []
for p in ['Wide_Rebalance_2pct', 'Narrow_Rebalance_1pct']:
    sm = traj['summary'][p]
    pname = 'Wide' if p.startswith('Wide') else 'Narrow'
    for m in METH:
        s = sm[m]
        vals = [s['full_window_within_coverage'], s['full_window_loro_coverage'],
                s['regime_within_coverage'], s['regime_loro_coverage']]
        put(f'l1.{pname}.{m}.cov', vals,
            f'trajectory_value_check.json:summary.{p}.{m}.{{full_window_within_coverage,full_window_loro_coverage,regime_within_coverage,regime_loro_coverage}}', 'f4')
        put(f'l1.{pname}.{m}.width', [s['full_window_loro_width'], s['regime_loro_width']],
            f'trajectory_value_check.json:summary.{p}.{m}.{{full_window_loro_width,regime_loro_width}}', 'f5')
        model = 'identified' if m.startswith('identified') else 'naive'
        meth = m.split('_', 1)[1]
        L1_ROWS.append([pname, model, meth, *[f'{v:.4f}' for v in vals],
                        f5(s['full_window_loro_width']), f5(s['regime_loro_width'])])
    for m in ['identified_plugin_normal', 'naive_plugin_normal']:
        s = sm[m]
        model = 'identified' if m.startswith('identified') else 'naive'
        put(f'l1.{pname}.{m}.cov', [s['full_window_coverage'], s['regime_coverage']],
            f'trajectory_value_check.json:summary.{p}.{m}.{{full_window_coverage,regime_coverage}}', 'f4')
        put(f'l1.{pname}.{m}.width', [s['full_window_width'], s['regime_width']],
            f'trajectory_value_check.json:summary.{p}.{m}.{{full_window_width,regime_width}}', 'f5')
        L1_ROWS.append([pname, model, 'plug-in normal', f'{s["full_window_coverage"]:.4f}', '--',
                        f'{s["regime_coverage"]:.4f}', '--',
                        f5(s['full_window_width']), f5(s['regime_width'])])

table3b = md_table(
    ['Policy', 'Model', 'Score', '$G$ within', '$G$ LORO', '$G_r$ within', '$G_r$ LORO',
     'width $G$', 'width $G_r$'],
    L1_ROWS)

# ranges
within = [traj['summary'][p][m]['regime_within_coverage'] for p in traj['summary'] for m in METH]
full_within = [traj['summary'][p][m]['full_window_within_coverage'] for p in traj['summary'] for m in METH]
raw_loro = [traj['summary'][p][m]['regime_loro_coverage'] for p in traj['summary'] for m in ['identified_raw', 'naive_raw']]
ext_stud = [traj['summary'][p][m]['regime_loro_per_group_coverage']['EXTENDED']
            for p in traj['summary'] for m in ['identified_studentized', 'naive_studentized']]
ext_cqr = [traj['summary'][p][m]['regime_loro_per_group_coverage']['EXTENDED']
           for p in traj['summary'] for m in ['identified_cqr', 'naive_cqr']]
put('l1.range.regime_within', [min(within), max(within)], 'computed: min/max over trajectory_value_check.json:summary.*.*.regime_within_coverage', 'f4')
within_pg = []
for p in traj['policies']:
    for model in ['ident', 'naive']:
        for m in ['raw', 'studentized', 'cqr']:
            pg = traj['policies'][p]['segment_wise'][model]['regime_level'][m]['within_regime']['per_group']
            within_pg += [g['coverage'] for g in pg.values()]
put('l1.range.regime_within_per_group', [min(within_pg), max(within_pg)],
    'computed: min/max over trajectory_value_check.json:policies.*.segment_wise.{ident,naive}.regime_level.{raw,studentized,cqr}.within_regime.per_group.*.coverage', 'f4')
put('l1.range.full_within', [min(full_within), max(full_within)], 'computed: min/max over trajectory_value_check.json:summary.*.*.full_window_within_coverage', 'f4')
put('l1.range.raw_loro', [min(raw_loro), max(raw_loro)], 'computed: min/max over trajectory_value_check.json:summary.*.*_raw.regime_loro_coverage', 'f4')
put('l1.range.ext_studentized_loro', [min(ext_stud), max(ext_stud)], 'computed: min/max over trajectory_value_check.json:summary.*.*_studentized.regime_loro_per_group_coverage.EXTENDED', 'f4')
put('l1.range.ext_cqr_loro', [min(ext_cqr), max(ext_cqr)], 'computed: min/max over trajectory_value_check.json:summary.*.*_cqr.regime_loro_per_group_coverage.EXTENDED', 'f4')

# LORO/within width ratios for identified studentized and CQR
ratio = {}
for p in traj['policies']:
    pname = 'Wide' if p.startswith('Wide') else 'Narrow'
    for m in ['identified_studentized', 'identified_cqr']:
        rec = traj['policies'][p]['segment_wise']['ident']['regime_level'][m.split('_', 1)[1]]
        ratio[(pname, m)] = rec['leave_one_regime_out']['mean_width'] / rec['within_regime']['mean_width']
put('l1.width_ratio.loro_within', {f'{k[0]}.{k[1]}': v for k, v in ratio.items()},
    'computed: leave_one_regime_out.mean_width / within_regime.mean_width from trajectory_value_check.json:policies.*.segment_wise.ident.regime_level.*', 'f3')

# paired bootstrap
pair_rows = []
for p in ['Wide_Rebalance_2pct', 'Narrow_Rebalance_1pct']:
    pname = 'Wide' if p.startswith('Wide') else 'Narrow'
    pr = traj['policies'][p]['paired']
    for level in ['full_window', 'regime']:
        for m in ['studentized', 'cqr', 'plugin_normal']:
            d = pr[level][m]
            put(f'l1.paired.{pname}.{level}.{m}', d, f'trajectory_value_check.json:policies.{p}.paired.{level}.{m}', 'f4')
            pair_rows.append([pname, level.replace('_', '-'), m,
                              f'{d["diff"]:+.4f}', f'[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]'])

table3c = md_table(['Policy', 'Level', 'Score', 'identified $-$ naive', '95% CI'], pair_rows)

table3 = f"""# Table 3 — Row-level and trajectory-level coverage

## Panel A. L0: held-out-regime row-level coverage (90% split conformal)

{table3a}

Notes. Nominal coverage is 0.90 ($\\alpha=0.1$); the 95% CI is a binomial normal
approximation on $n_{{\\rm test}}$ rows and ignores within-trajectory correlation,
so it is a lower bound on the true uncertainty. Widths are mean interval widths.
Naive global = one pooled calibration radius (baseline). Mondrian = one radius
per calibration regime, unseen regime falls back to the largest calibration
radius. Similarity = Mondrian with the two-class fallback
{{OPEN, EXTENDED}} / {{WEEKEND, HOLIDAY, HALT}}.
Baseline aggregate `coverage_gap` = {base['coverage_gap']:.4f}; similarity fallback
= {sim['coverage_gap']:.4f}; strategy-conditional = {strat['coverage_gap']:.5f}.
Strategy-weighted gap: naive {base['strategy_coverage_gap']:.4f}, similarity
{sim['strategy_coverage_gap']:.4f}, strategy-conditional
{strat['strategy_coverage_gap']:.4f}. WEEKEND coverage by window (W1–W4):
{', '.join(f'{w} {v["coverage"]:.4f}' for w, v in base['heldout_regime_by_window']['2'].items())}.
HALT calibration radius: 384 rows per fold, row-bootstrap SE
{min(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values()):.6f}–
{max(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values()):.6f}
({100*min(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values()):.1f}–
{100*max(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values()):.1f}% of the radius).

## Panel B. L1: trajectory-level 90% conformal coverage and width

{table3b}

Notes. $G$ is the full-window cumulative reward and $G_r$ the regime-segment
reward; within = calibrate and test on the same regime/window population; LORO =
leave-one-regime-out (for $G_r$) or leave-one-window-out (for $G$). Widths are
mean 90% interval widths: for conformal rows they are the LORO widths, whereas
for plug-in normal rows they are the full-window ($G$) and within-regime
($G_r$) widths, since the plug-in has no LORO variant. Plug-in normal =
$\\pm z_{{0.95}}\\,\\hat{{\\rm sd}}$, not a conformal interval. Identified uses
the regime-dependent observation model with G0 true-$S$ diagnostic parameters;
naive treats $P$ as $S$ with per-path $\\kappa_P/\\mu_P/\\sigma_{{\\rm all}}$.

## Panel C. L1: path-level paired bootstrap, identified $-$ naive

{table3c}

Notes. 2,000 path-level resamples, conditioning on the fitted calibration
quantiles; the interval quantifies test-path sampling only.
"""
(OUT / 'table3-coverage.md').write_text(table3)

# ---------------------------------------------------------------------------
# L2 branches
# ---------------------------------------------------------------------------
branch_rows = []
branch_ci_rows = []
bj_stud = pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods']['identified_studentized']['within_tau2']['branch_union_joint']
for meth in ['identified_raw', 'identified_studentized', 'identified_cqr']:
    md = pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods'][meth]['within_tau2']
    bj = md['branch_union_joint']
    hold, reb = bj['branches']['hold'], bj['branches']['rebalance']
    triv = max(0.0, 1 - (1 - bj['overall_marginal_coverage']) / reb['cal_fraction'])
    ci = {}
    for bname, brec in [('hold', hold), ('rebalance', reb)]:
        n = brec['n_test']
        km = round(brec['marginal_coverage'] * n)
        ku = round(brec['union_coverage'] * n)
        ci[bname] = {
            'n': n,
            'marginal': wilson_ci(km, n),
            'union': wilson_ci(ku, n),
        }
        branch_ci_rows.append([
            meth.replace('identified_', ''), bname, f'{n}',
            f'{brec["marginal_coverage"]:.3f} [{ci[bname]["marginal"][0]:.3f}, {ci[bname]["marginal"][1]:.3f}]',
            f'{brec["union_coverage"]:.3f} [{ci[bname]["union"][0]:.3f}, {ci[bname]["union"][1]:.3f}]',
        ])
    put(f'l2.{meth}.branch_ci', ci,
        f'computed: Wilson 95% intervals from policy_value_check.json:event_conformal.policies.Wide_Rebalance_2pct.methods.{meth}.within_tau2.branch_union_joint.branches.*.{{n_test,marginal_coverage,union_coverage}}', 'f4')
    put(f'l2.{meth}.branch', {
        'overall_marginal': bj['overall_marginal_coverage'],
        'overall_union': bj['overall_union_coverage'],
        'overall_gain': bj['overall_coverage_gain'],
        'width_ratio': bj['width_ratio_vs_marginal'],
        'hold': {k: hold[k] for k in ('n_test', 'marginal_coverage', 'union_coverage')},
        'rebalance': {k: reb[k] for k in ('n_test', 'marginal_coverage', 'union_coverage', 'coverage_gain', 'coverage_gain_se', 'width_ratio_vs_marginal', 'taxonomy')},
        'trivial_bound': triv,
    }, f'policy_value_check.json:event_conformal.policies.Wide_Rebalance_2pct.methods.{meth}.within_tau2.branch_union_joint', 'f4')
    branch_rows.append([
        meth.replace('identified_', ''),
        f'{hold["n_test"]}', f'{hold["marginal_coverage"]:.4f}', f'{hold["union_coverage"]:.4f}',
        f'{reb["n_test"]}', f'{reb["marginal_coverage"]:.4f}', f'{reb["union_coverage"]:.4f}',
        f'{reb["coverage_gain"]:+.3f} ({reb["coverage_gain_se"]:.3f})',
        f'{bj["width_ratio_vs_marginal"]:.2f}×', f'{triv:.3f}',
        f'{hold["taxonomy"]} / {reb["taxonomy"]}',
    ])

put('l2.stud.overall', {k: bj_stud[k] for k in ('overall_marginal_coverage', 'overall_union_coverage',
    'overall_coverage_gain', 'width_ratio_vs_marginal', 'worst_branch_marginal_coverage',
    'worst_branch_union_coverage', 'worst_gap_before', 'worst_gap_after')},
    'policy_value_check.json:event_conformal.policies.Wide_Rebalance_2pct.methods.identified_studentized.within_tau2.branch_union_joint', 'f4')

# aggregate taxonomy
import collections
cnt = collections.Counter()
blocks = cells = dom = 0
min_cal_n = None
inf_cells = 0
for p in pol['event_conformal']['policies']:
    md = pol['event_conformal']['policies'][p]['methods']
    for meth, rec in md.items():
        for direction in ['within_tau2', 'within_tau81', 'cal_tau2_test_tau81', 'cal_tau81_test_tau2']:
            bj = rec.get(direction, {}).get('branch_union_joint')
            if not bj:
                continue
            blocks += 1
            dom += int(bj['coverage_dominance'])
            bu = rec.get(direction, {}).get('branch_union', {})
            inf_cells += int(bu.get('n_inf_union', 0))
            for b, br in bj['branches'].items():
                cells += 1
                cnt[br['taxonomy']] += 1
            for b, br in bu.get('branches', {}).items():
                if min_cal_n is None or br['cal_n'] < min_cal_n:
                    min_cal_n = br['cal_n']
put('l2.aggregate', {'blocks': blocks, 'cells': cells, 'dominance_blocks': dom,
                     'taxonomy': dict(cnt), 'min_branch_cal_n': min_cal_n,
                     'inf_union_cells': inf_cells},
    'computed: policy_value_check.json:event_conformal.policies.*.methods.*.{within_tau2,within_tau81,cal_tau2_test_tau81,cal_tau81_test_tau2}.{branch_union_joint,branch_union}')

# cross-regime 81->2
cross = {}
for meth in ['identified_raw', 'identified_studentized', 'identified_cqr']:
    bj = pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods'][meth]['cal_tau81_test_tau2']['branch_union_joint']
    cross[meth] = {'marginal': bj['branches']['rebalance']['marginal_coverage'],
                   'union': bj['branches']['rebalance']['union_coverage'],
                   'width_ratio': bj['branches']['rebalance']['width_ratio_vs_marginal'],
                   'taxonomy': bj['branches']['rebalance']['taxonomy']}
put('l2.cross81to2', cross,
    'computed: selected fields of branches.rebalance from policy_value_check.json:event_conformal.policies.Wide_Rebalance_2pct.methods.*.cal_tau81_test_tau2.branch_union_joint', 'f4')

# Mondrian pre-decision range
mons = []
for seed, sd in pol['event_conformal_seed_robustness'].items():
    for pname, pd in sd.items():
        if pname != 'Wide_Rebalance_2pct':
            continue
        for meth in ['identified_raw', 'identified_studentized']:
            m = pd[meth]['mondrian']['within_tau2']
            for g in ['onset_action', 'onset_side']:
                mons.append((m[g]['post_rebalance_coverage'], seed, meth, g))
mons.sort()
put('l2.mondrian.range', [mons[0], mons[-1]],
    'policy_value_check.json:event_conformal_seed_robustness.*.Wide_Rebalance_2pct.*_raw|studentized.mondrian.within_tau2.{onset_action,onset_side}.post_rebalance_coverage', 'f4')

table4 = f"""# Table 4 — Branch-conditional coverage (L2)

## Panel A. Wide τ=2, post-decision branches (main seed, 128 test events)

{md_table(['Score', 'hold n', 'hold cov.', 'hold union', 'rebal n', 'rebal cov.', 'rebal union',
           'rebal gain (SE)', 'overall width ratio', 'trivial bound', 'taxonomy hold / rebal'], branch_rows)}

Notes. Nominal 0.90 ($\\alpha=0.1$). The paired gain's standard error (one SE,
not a 95% interval) is in parentheses; a single-cell coverage SE is about
0.05–0.10 for the 24-event branch and 0.02–0.05 for the 104-event branch.
`Trivial bound` is
$1-(1-\\hat c)/\\hat q_b$ with $\\hat c$ the measured overall marginal coverage
and $\\hat q_b=24/128$; it uses the observed $\\hat c$, not the nominal level, so
it is an empirical identity, not a guarantee. The branch-union threshold is
$\\max(\\hat q_{{\\rm hold}},\\hat q_{{\\rm rebalance}})$ on the calibration side;
`post_rebalance` is a post-decision label and is never used at prediction time.
`width ratio` is overall union width / overall marginal width.

## Panel B. Branch-union aggregate across 56 method×direction blocks

| Statistic | Value |
| --- | --- |
| Coverage dominance (union ≥ marginal) | {dom}/{blocks} blocks, {cells}/{cells} branch cells |
| Nonempty calibration branches | 112/112 branch cells; smallest $n_b$ = {min_cal_n} |
| Infinite branch thresholds ($\\hat q_b=\\infty$) | {inf_cells} cells |
| Taxonomy | cost_only {cnt['cost_only']}, clear_gain {cnt['clear_gain']}, gain_but_fails {cnt['gain_but_fails']}, fails_no_clear_gain {cnt['fails_no_clear_gain']}, ambiguous {cnt['ambiguous']} |
| Wide 81→2 rebalance, identified studentized | {cross['identified_studentized']['marginal']:.3f} → {cross['identified_studentized']['union']:.3f} (width {cross['identified_studentized']['width_ratio']:.2f}×, {cross['identified_studentized']['taxonomy']}) |
| Wide 81→2 rebalance, identified CQR | {cross['identified_cqr']['marginal']:.3f} → {cross['identified_cqr']['union']:.3f} (width {cross['identified_cqr']['width_ratio']:.2f}×, {cross['identified_cqr']['taxonomy']}) |

Notes. Taxonomy uses normal-approximation 95% intervals per branch; the cells are
descriptive, not simultaneous. `cost_only` = marginal coverage already clearly
at or above nominal; `clear_gain` = a gain whose 95% interval excludes zero,
with union still possibly below nominal; `gain_but_fails` = a gain whose 95%
interval excludes zero, but union still clearly below nominal;
`fails_no_clear_gain` = below nominal with no gain whose 95% interval excludes
zero; `ambiguous` otherwise.

## Panel C. Pre-decision Mondrian cannot isolate the post branch (text summary)

Wide τ=2, identified raw/studentized, three seeds, `onset_action`/`onset_side`:
post-rebalance coverage ranges {mons[0][0]:.3f}–{mons[-1][0]:.3f} across
{len(mons)} cells, versus {bj_stud['branches']['hold']['marginal_coverage']:.3f}
on hold and {bj_stud['branches']['rebalance']['marginal_coverage']:.3f} on
rebalance under the unstratified marginal pipeline. Pre-decision stratification
cannot recover the post-decision branch because the branch label is not in the
pre-decision $\\sigma$-field.

## Panel D. Branch-level Wilson intervals (Wide τ=2)

{md_table(['Score', 'Branch', '$n$', 'Marginal [95% CI]', 'Union [95% CI]'], branch_ci_rows)}

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
"""
(OUT / 'table4-branches.md').write_text(table4)

# ---------------------------------------------------------------------------
# L3 h table
# ---------------------------------------------------------------------------
h_truth = obs['h_truth']
h_g0 = obs['h_g0_true_s_diagnostic']
h_main = obs['h_estimate']
h_unmasked = obs['h_unmasked_mle']['h']
specs_h = {
    'truth': h_truth,
    'G0 (true $S$ diagnostic)': h_g0,
    'observable, z=3 (main)': h_main,
    'observable, z=2.5': traj_specs['observable $h$, z=2.5']['inputs']['h'],
    'observable, z=4': traj_specs['observable $h$, z=4']['inputs']['h'],
    'observable, $\\sigma$ from OPEN': traj_specs['observable $h$, $\\sigma$ from OPEN']['inputs']['h'],
    'unmasked MLE': h_unmasked,
}
h_rows = []
for name, h in specs_h.items():
    h_rows.append([name, f'{h["open_sd"]:.6f}', f'{h["ext_phi"]:.4f}', f'{h["ext_sd"]:.6f}',
                   f'{h["stale_phi"]:.4f}', f'{h["stale_sd"]:.6f}'])
put('l3.h_estimates', {k: v for k, v in specs_h.items()},
    'computed: curated fields from observable_h_check.json:{h_truth,h_g0_true_s_diagnostic,h_estimate,h_unmasked_mle.h} and trajectory_value_check_obs_h*.json:inputs.h', 'f6')

bs = obs['uncertainty']['bootstrap']
hs = obs['uncertainty']['hessian_se']
put('l3.ext_phi.se', {'bootstrap': bs['ext_phi']['se'], 'hessian': hs['ext_phi'],
                      'q05': bs['ext_phi']['q05'], 'q95': bs['ext_phi']['q95']},
    'observable_h_check.json:uncertainty.{bootstrap,hessian_se}.ext_phi', 'f4')
put('l3.open_sd.se', {'bootstrap': bs['open_sd']['se'], 'hessian': hs['open_sd']},
    'observable_h_check.json:uncertainty.{bootstrap,hessian_se}.open_sd', 'e3')

sens_rows = []
coverage_spread = {}
for pname in ['Wide_Rebalance_2pct', 'Narrow_Rebalance_1pct']:
    short = 'Wide' if pname.startswith('Wide') else 'Narrow'
    for key, d in traj_specs.items():
        s = d['summary'][pname]['identified_raw']
        sens_rows.append([short, key, f'{s["full_window_within_coverage"]:.4f}', f'{s["regime_within_coverage"]:.4f}'])
        coverage_spread.setdefault((short, 'full'), []).append(s['full_window_within_coverage'])
        coverage_spread.setdefault((short, 'regime'), []).append(s['regime_within_coverage'])
put('l3.coverage_sensitivity', {f'{k[0]}.{k[1]}': {'min': min(v), 'max': max(v), 'spread_pp': 100*(max(v)-min(v))}
                                for k, v in coverage_spread.items()},
    'computed: summary stats over trajectory_value_check*.json:summary.*.identified_raw.{full_window_within_coverage,regime_within_coverage}', 'f4')

# per-regime five-specification spread of within-regime coverage
spec_items = list(traj_specs.items())
regimes = list(spec_items[0][1]['policies']['Wide_Rebalance_2pct']['segment_wise']['ident']
               ['regime_level']['raw']['within_regime']['per_group'].keys())
pg_spread_raw = {}
pg_spread_all = {}
for pname, short in [('Wide_Rebalance_2pct', 'Wide'), ('Narrow_Rebalance_1pct', 'Narrow')]:
    for reg in regimes:
        vals = [d['policies'][pname]['segment_wise']['ident']['regime_level']['raw']
                ['within_regime']['per_group'][reg]['coverage'] for _, d in spec_items]
        pg_spread_raw[(short, reg)] = 100 * (max(vals) - min(vals))
    for model in ['ident', 'naive']:
        for meth in ['raw', 'studentized', 'cqr']:
            for reg in regimes:
                vals = [d['policies'][pname]['segment_wise'][model]['regime_level'][meth]
                        ['within_regime']['per_group'][reg]['coverage'] for _, d in spec_items]
                pg_spread_all[(short, model, meth, reg)] = 100 * (max(vals) - min(vals))
put('l3.coverage_sensitivity.per_regime_raw', {f'{k[0]}.{k[1]}': v for k, v in pg_spread_raw.items()},
    'computed: min/max over trajectory_value_check*.json:policies.*.segment_wise.ident.regime_level.raw.within_regime.per_group.*.coverage', 'f2')
put('l3.coverage_sensitivity.per_regime_all_methods',
    {'max_pp': max(pg_spread_all.values()),
     'argmax': list(max(pg_spread_all, key=pg_spread_all.get)),
     'all': {'.'.join(k): v for k, v in pg_spread_all.items()}},
    'computed: min/max over trajectory_value_check*.json:policies.*.segment_wise.{ident,naive}.regime_level.{raw,studentized,cqr}.within_regime.per_group.*.coverage', 'f2')
max_pg_raw = max(pg_spread_raw, key=pg_spread_raw.get)
max_pg_all = max(pg_spread_all, key=pg_spread_all.get)

table5a = md_table(['Specification', '$\\\\hat h_{\\\\rm open,sd}$', '$\\\\hat h_{\\\\rm ext,\\\\phi}$',
           '$\\\\hat h_{\\\\rm ext,sd}$', '$\\\\hat h_{\\\\rm stale,\\\\phi}$', '$\\\\hat h_{\\\\rm stale,sd}$'], h_rows)
table5c = md_table(['Policy', 'Specification', '$G$ within', '$G_r$ within'], sens_rows)
wide_full = 100*(max(coverage_spread[('Wide','full')])-min(coverage_spread[('Wide','full')]))
wide_reg = 100*(max(coverage_spread[('Wide','regime')])-min(coverage_spread[('Wide','regime')]))
nar_full = 100*(max(coverage_spread[('Narrow','full')])-min(coverage_spread[('Narrow','full')]))
nar_reg = 100*(max(coverage_spread[('Narrow','regime')])-min(coverage_spread[('Narrow','regime')]))
table5 = f"""# Table 5 — Observable-only observation map $h$ (L3)

## Panel A. Point estimates

{table5a}

Notes. Truth: $h_{{\\rm open,sd}}=0.0005$, $h_{{\\rm ext,\\phi}}=0.8$,
$h_{{\\rm ext,sd}}=0.002$, $h_{{\\rm stale,\\phi}}=0.98$, $h_{{\\rm stale,sd}}=0.0008$.
The G0 row uses true $S$ as a diagnostic and is not an observable estimator.
`open_sd` and `ext_phi` are weakly identified from $P$ alone; the trimming
threshold z moves `ext_phi` from {specs_h['observable, z=4']['ext_phi']:.4f} (z=4)
through {h_main['ext_phi']:.4f} (z=3) to {specs_h['observable, z=2.5']['ext_phi']:.4f} (z=2.5).

## Panel B. Uncertainty for the weakly identified parameters

| Parameter | Bootstrap SE | Hessian SE | 90% bootstrap interval |
| --- | ---: | ---: | --- |
| $h_{{\\rm ext,\\phi}}$ | {bs['ext_phi']['se']:.4f} | {hs['ext_phi']:.4f} | [{bs['ext_phi']['q05']:.4f}, {bs['ext_phi']['q95']:.4f}] |
| $h_{{\\rm open,sd}}$ | {bs['open_sd']['se']:.6f} | {hs['open_sd']:.6f} | [{bs['open_sd']['q05']:.6f}, {bs['open_sd']['q95']:.6f}] |

The 90% bootstrap interval for $h_{{\\rm ext,\\phi}}$ is
[{bs['ext_phi']['q05']:.4f}, {bs['ext_phi']['q95']:.4f}], whose upper end lies
below the simulator truth 0.8. Because the trimming threshold is a researcher
choice, this interval is specification-specific, not a uniform confidence
statement.

## Panel C. Coverage sensitivity to the $h$ specification

{table5c}

Identified raw: the pooled $G$ spread is {wide_full:.1f} pp (Wide) and
{nar_full:.1f} pp (Narrow); the $G_r$ spread is {wide_reg:.1f} pp and
{nar_reg:.1f} pp. The largest per-regime five-specification spread is
{pg_spread_raw[max_pg_raw]:.1f} pp (identified raw, {max_pg_raw[0]},
{max_pg_raw[1]}). Across all conformal score/pipeline combinations the largest
per-regime spread is {pg_spread_all[max_pg_all]:.1f} pp (identified
{max_pg_all[2]}, {max_pg_all[0]}, {max_pg_all[3]}); per-regime cells are noisy,
and the pooled estimands are the stable objects.

Notes. Point estimates only; nominal coverage is 0.90 ($\\alpha=0.1$). No 95%
interval is attached to the individual coverage cells; the uncertainty object
is the across-specification spread. The five specifications differ only in the
$h$ input of the same trajectory pipeline, so they are a specification curve,
not independent replications. The `truth (G0)` rows in Panel C use the
true-$S$ diagnostic $h$, not an observable estimator; the point estimate of $h$
is not used as a certificate.
"""
(OUT / 'table5-h.md').write_text(table5)

# ---------------------------------------------------------------------------
# Stress table
# ---------------------------------------------------------------------------
stress_rows = []
for env, gtrue in [('gamma0', 0.0), ('gamma_star', 4.0)]:
    for tier in ['oracle', 'estimated']:
        rec = stress['results'][env][tier]
        oo, ce_ = rec['open_only'], rec['cross_event']
        gfit = ce_['gamma2_full_fit']
        stress_rows.append([
            f'$\\\\gamma={gtrue:.0f}$', tier,
            f'{oo["coverage"]:.3f} ({oo["coverage_se"]:.3f})', f'{oo["mean_width"]:.4f}',
            f'{ce_["coverage"]:.3f} ({ce_["coverage_se"]:.3f})', f'{ce_["mean_width"]:.4f}',
            f'{ce_["width_ratio_vs_open"]:.2f}×',
            f'{gfit["gamma_hat"]:.2f} [{gfit["ci95_gamma"][0]:.2f}, {gfit["ci95_gamma"][1]:.2f}]',
        ])
        put(f'l6.{env}.{tier}', {'open_cov': oo['coverage'], 'open_width': oo['mean_width'],
                                 'cross_cov': ce_['coverage'], 'cross_width': ce_['mean_width'],
                                 'width_ratio': ce_['width_ratio_vs_open'],
                                 'gamma_hat': gfit['gamma_hat'], 'gamma_ci': gfit['ci95_gamma']},
            f'stress_variant.json:results.{env}.{tier}', 'f4')

ident = stress['identification']
put('l6.ks_open_increment', ident['open_increment_ks_between_twins'],
    'stress_variant.json:identification.open_increment_ks_between_twins', 'f4')
put('l6.var_ratio_perm', ident['defreeze_variance_ratio_permutation'],
    'stress_variant.json:identification.defreeze_variance_ratio_permutation', 'f4')

table6a = md_table(['Environment', 'Tier', 'open-only cov. (SE)', 'open width',
           'cross-event cov. (SE)', 'cross width', 'width ratio', '$\\\\hat\\\\gamma$ [95% CI]'],
          stress_rows)
table6 = f"""# Table 6 — I5 stress variant (unfreeze jumps)

{table6a}

Notes. 90% prediction intervals, 5-fold out-of-fold (fold = seed mod 5), 128
seeds × 4 windows = 512 paths, 1,536 double-endpoint events (τ=2/65/81, 512 each). Nominal
0.90 ($\\alpha=0.1$). Coverage SEs are binomial over the 1,536 event predictions
and ignore within-path and across-fold dependence, so they are lower bounds.
`open-only` uses ordinary OPEN→OPEN one-step transitions;
`cross-event` uses $\\operatorname{{Var}}(r_e)=v_{{0e}}+\\theta w_e$ with
$w_e=\\sigma^2(1-e^{{-\\tau_e/\\tau_0}})$, $\\theta=\\gamma^2$, and a profile
likelihood CI. The γ̂ CI is the full-data MLE profile interval, not a coverage
interval. γ=4 CIs are
[{stress['results']['gamma_star']['oracle']['cross_event']['gamma2_full_fit']['ci95_gamma'][0]:.2f}, {stress['results']['gamma_star']['oracle']['cross_event']['gamma2_full_fit']['ci95_gamma'][1]:.2f}]
(oracle) and
[{stress['results']['gamma_star']['estimated']['cross_event']['gamma2_full_fit']['ci95_gamma'][0]:.2f}, {stress['results']['gamma_star']['estimated']['cross_event']['gamma2_full_fit']['ci95_gamma'][1]:.2f}]
(estimated); γ=0 CIs are
[{stress['results']['gamma0']['oracle']['cross_event']['gamma2_full_fit']['ci95_gamma'][0]:.2f}, {stress['results']['gamma0']['oracle']['cross_event']['gamma2_full_fit']['ci95_gamma'][1]:.2f}]
and
[{stress['results']['gamma0']['estimated']['cross_event']['gamma2_full_fit']['ci95_gamma'][0]:.2f}, {stress['results']['gamma0']['estimated']['cross_event']['gamma2_full_fit']['ci95_gamma'][1]:.2f}]:
the intervals do not overlap.

Identification diagnostics: the ordinary OPEN→OPEN one-step kernel is
γ-free by construction; the twin open-increment KS statistic is
{ident['open_increment_ks_between_twins']['statistic']:.5f}
(p={ident['open_increment_ks_between_twins']['pvalue']:.3f}, descriptive and
dependent samples). The de-freeze $z^2$ variance ratio (γ=4/γ=0) is
{ident['defreeze_variance_ratio_permutation']['ratio_gamma_star_over_gamma0']:.3f},
paired path-permutation p={ident['defreeze_variance_ratio_permutation']['perm_p_two_sided']:.4f}.
"""
(OUT / 'table6-stress.md').write_text(table6)

# ---------------------------------------------------------------------------
# Q1-Q3 table
# ---------------------------------------------------------------------------
table7 = """# Table 7 — Answers to Q1–Q3

| Question | Answer | Evidence |
| --- | --- | --- |
| Q1. What is identifiable from observable data? | Under I1′/I2/I3 (with A1, A2, A4–A7, T1, T4, and A3′ where applicable), the short-range transition, the stale/extended observation scales, and the value functional are identified. The Markov bridge needs no I5/I5′; the non-Markov bridge additionally needs I5/T2/T3 for the endpoint law and I5′ for the path law, so its value conclusion carries those extra assumptions. $h_{\\rm open,sd}/h_{\\rm ext,\\phi}$ are only weakly identified. | §3, Theorem 1; §7.4, Table 5 |
| Q2. What is recoverable by calibration? | Marginal row-level coverage under a held-out regime is restored by a max/similarity fallback radius (within-regime calibration is the unexercised positive control in the held-out-regime L0 design); trajectory-level $G$ or regime-segment $G_r$ coverage is recoverable only with evaluation-regime calibration data; branch-conditional coverage is recoverable with calibration-side branch labels via branch-union at a width cost. | §5.2; §§7.1–7.3, §7.7; Table 4 and Appendix Tables B5, B7 |
| Q3. What is impossible? | For branch-blind, nondegenerate score rules, no distribution-free branch-conditional guarantee from pre-decision information; within the non-Markov freeze class without I5/I5′, there exist observationally equivalent models with different values when the bridge reward is not determined by endpoints (Theorem 4 is an existence result inside that class); no coverage-gain-per-unit-width lower bound. | §5, Theorems 3–4, Propositions 2A–2C; §§7.3–7.5, §7.7 |
"""
(OUT / 'table7-q1q3.md').write_text(table7)

# ---------------------------------------------------------------------------
# D2' decomposition table
# ---------------------------------------------------------------------------
def fmt_w1(x):
    if x == 0:
        return '0'
    if abs(x) < 1e-3:
        return f'{x:.2e}'
    return f'{x:.4f}'

d2 = pol['d2prime']
w1_rows = []
for tau in ['2', '81']:
    tb = d2['belief']['by_tau'][tau]
    for ph in ['anchor', 'freeze', 'post']:
        rec = tb['w1_episode_mean'][ph]
        orc = tb['oracle_w1_episode_mean'][ph]
        ratio = rec['mean'] / orc['mean'] if orc['mean'] else float('nan')
        w1_rows.append([f'$\\tau={tau}$', ph, f'{rec["n"]:,}', fmt_w1(rec['mean']),
                        fmt_w1(rec['sd']), fmt_w1(rec['p90']), fmt_w1(orc['mean']),
                        f'{ratio:.3f}'])
        put(f'd2.w1.tau{tau}.{ph}', {
            'n': rec['n'], 'identified_mean': rec['mean'], 'identified_sd': rec['sd'],
            'identified_p90': rec['p90'], 'oracle_mean': orc['mean'], 'ratio': ratio,
        }, f'policy_value_check.json:d2prime.belief.by_tau.{tau}.w1_episode_mean.{ph} + oracle_w1_episode_mean.{ph}', 'e3')

tv_rows = []
for tau in ['2', '81']:
    tb = d2['belief']['by_tau'][tau]
    for ph in ['anchor', 'freeze']:
        rec = tb['tv_episode_mean'][ph]
        step = tb['tv_step_pooled'][ph]
        tv_rows.append([f'$\\tau={tau}$', ph, f'{rec["n"]:,}', f'{rec["mean"]:.3f}',
                        f'{rec["sd"]:.3f}', f'{rec["p90"]:.3f}', f'{rec["max"]:.3f}',
                        f'{step["frac_ge_0.99"]:.2f}'])
        put(f'd2.tv.tau{tau}.{ph}', {
            'n': rec['n'], 'mean': rec['mean'], 'sd': rec['sd'], 'p90': rec['p90'],
            'max': rec['max'], 'step_frac_ge_099': step['frac_ge_0.99'],
        }, f'policy_value_check.json:d2prime.belief.by_tau.{tau}.tv_episode_mean.{ph} + tv_step_pooled.{ph}.frac_ge_0.99', 'f4')

flip_rows = []
margin_rows = []
for pname, plabel in [('Wide_Rebalance_2pct', 'Wide 2%'), ('Narrow_Rebalance_1pct', 'Narrow 1%')]:
    for tau in ['2', '81']:
        rec = d2['policy']['by_policy'][pname]['by_tau'][tau]
        fl, mg = rec['flip'], rec['margin']
        flip_rows.append([plabel, f'$\\tau={tau}$', f'{fl["n"]:,}', f'{fl["mean"]:.3f}',
                          f'{fl["sd"]:.3f}', f'{fl["p90"]:.3f}', f'{fl["max"]:.3f}',
                          f'{fl["mean_p_id"]:.3f}', f'{fl["mean_p_naive"]:.3f}',
                          f'{fl["frac_gt_0.5"]:.3f}'])
        margin_rows.append([plabel, f'$\\tau={tau}$', f'{mg["n"]:,}', f'{mg["mean"]:.4f}',
                            f'{mg["frac_rebalance"]:.3f}', f'{mg["frac_abs_le_0.0025"]:.3f}',
                            f'{mg["frac_abs_le_0.005"]:.3f}', f'{mg["frac_abs_le_0.01"]:.3f}'])
        put(f'd2.flip.{pname}.tau{tau}', {
            'n': fl['n'], 'mean': fl['mean'], 'sd': fl['sd'], 'p90': fl['p90'],
            'max': fl['max'], 'mean_p_id': fl['mean_p_id'], 'mean_p_naive': fl['mean_p_naive'],
            'frac_gt_0.5': fl['frac_gt_0.5'],
        }, f'policy_value_check.json:d2prime.policy.by_policy.{pname}.by_tau.{tau}.flip', 'f4')
        put(f'd2.margin.{pname}.tau{tau}', {
            'n': mg['n'], 'mean': mg['mean'], 'frac_rebalance': mg['frac_rebalance'],
            'frac_abs_le_0.0025': mg['frac_abs_le_0.0025'],
            'frac_abs_le_0.005': mg['frac_abs_le_0.005'],
            'frac_abs_le_0.01': mg['frac_abs_le_0.01'],
        }, f'policy_value_check.json:d2prime.policy.by_policy.{pname}.by_tau.{tau}.margin', 'f4')

table8a = md_table(['$\\tau$ (h)', 'Phase', '$n$', '$W_1$ mean', '$W_1$ SD',
                    '$W_1$ p90', 'Oracle $W_1$ mean', 'Ratio'], w1_rows)
table8b = md_table(['$\\tau$ (h)', 'Phase', '$n$', 'TV mean', 'TV SD', 'TV p90',
                    'TV max', 'Step frac TV$\\ge$0.99'], tv_rows)
table8c1 = md_table(['Policy', '$\\tau$ (h)', '$n$', 'Flip mean', 'Flip SD', 'Flip p90',
                     'Flip max', 'Mean $p_{\\rm id}$', 'Mean $p_{\\rm naive}$',
                     'Frac flip > 0.5'], flip_rows)
table8c2 = md_table(['Policy', '$\\tau$ (h)', '$n$', 'Mean margin', 'Frac rebalance',
                     'Frac $\\vert m\\vert\\le0.0025$', 'Frac $\\vert m\\vert\\le0.005$',
                     'Frac $\\vert m\\vert\\le0.01$'], margin_rows)

table8 = (
    "# Table 8 — Three-term decomposition, reported separately\n\n"
    "The decomposition is not a theorem. The terms are along observed histories, "
    "not a uniform sup over beliefs; the third term is a diagnostic refinement for "
    "observation-type policies, not an independent bound term. The phases are "
    "anchor = last informative step before the freeze, freeze = pre+1..post−1, and "
    "post = first OPEN after the freeze.\n\n"
    "## Panel A. $W_1$ estimation term (identified posterior vs naive delta; oracle uses latent $S_t$)\n\n"
    + table8a + "\n\n"
    "Notes. $W_1 = \\mathbb{E}|\\log S_t - \\log P_t|$ under the identified posterior, "
    "i.e., $W_1$ between the identified posterior and the delta at $\\log P_t$ (the "
    "naive treats $P_t$ as $S_t$); the oracle replaces $P_t$ by the simulator's latent "
    "$S_t$ for validation only. Ratio = "
    "identified mean / oracle mean. $n$ is the number of episodes (128 per cell); "
    "the $\\tau=2$ anchor cell is numerically exact (SD $<10^{-15}$); the "
    "$\\tau=81$ anchor SD is $7.6\\times10^{-5}$. The freeze term grows "
    "with the freeze duration: 0.0076 at $\\tau=2$ and 0.0407 at $\\tau=81$.\n\n"
    "## Panel B. Predictive-distribution TV information term\n\n"
    + table8b + "\n\n"
    "Notes. TV between the identified and naive one-step predictives of "
    "$\\log P_{t+1}$, evaluated at the observed history. `Step frac` is the fraction "
    "of pooled steps with TV $\\ge 0.99$; episode-level maxima can be below 1 even "
    "when some steps are at 1. The post phase is not reported in the current JSON. "
    "At the $\\tau=2$ anchor the two predictives are maximally different (TV = 1); "
    "during the freeze TV is 0.65 ($\\tau=2$) and 0.77 ($\\tau=81$).\n\n"
    "## Panel C. Policy boundary/action-flip diagnostic\n\n"
    "## Panel C.1 Flip probability\n\n"
    + table8c1 + "\n\n"
    "## Panel C.2 Boundary proximity (context)\n\n"
    + table8c2 + "\n\n"
    "Notes. Flip = minimal-coupling probability that the rebalance decision at "
    "$t+1$ differs, $|p_{\\rm id}-p_{\\rm naive}|$, where $p_{\\rm model} = "
    "P(|P_{t+1}/{\\rm anchor}-1| > \\theta)$ under that model's predictive; positions "
    "are from the observed trajectory (first-order). $p_{\\rm id}$ and "
    "$p_{\\rm naive}$ are the model-implied rebalance probabilities. Margin = "
    "$|P_t/{\\rm anchor}-1| - \\theta$ at decision steps; $m>0$ means the observed "
    "path rebalances. The boundary-proximity panel is context for the flip diagnostic, not a fourth term. The flip diagnostic "
    "is not a bound: it describes how close the policy is to its decision boundary "
    "under the two models.\n"
)
(OUT / 'table8-d2prime.md').write_text(table8)

# ---------------------------------------------------------------------------
# Non-Markov Brownian bridge table
# ---------------------------------------------------------------------------
bridge = stress['nonmarkov_bridge']
if bridge.get('available'):
    env_labels = [
        ('bridge0', '$\\nu_{\\rm mult}=0$ placebo'),
        ('bridge_star', '$\\nu_{\\rm mult}=0.6$ bridge'),
    ]
    bridge_rows = []
    for env, env_label in env_labels:
        for tier in ['oracle', 'estimated']:
            rec = bridge['results'][env][tier]
            oo, ce_, ct = rec['open_only'], rec['cross_event'], rec['cross_event_tau']
            fit = ce_['component_full_fit']
            bridge_rows.append([
                env_label, tier,
                f'{oo["coverage"]:.3f} ({oo["coverage_se"]:.3f})',
                f'{oo["mean_width"]:.4f}',
                f'{ce_["coverage"]:.3f} ({ce_["coverage_se"]:.3f})',
                f'{ce_["mean_width"]:.4f}',
                f'{ce_["width_ratio_vs_open"]:.2f}×',
                f'{ct["coverage"]:.3f} ({ct["coverage_se"]:.3f})',
                f'{ct["width_ratio_vs_open"]:.2f}×',
            ])
            put(f'l9.{env}.{tier}', {
                'open_cov': oo['coverage'], 'open_width': oo['mean_width'],
                'cross_cov': ce_['coverage'], 'cross_width': ce_['mean_width'],
                'cross_width_ratio': ce_['width_ratio_vs_open'],
                'tau_cov': ct['coverage'], 'tau_width': ct['mean_width'],
                'tau_width_ratio': ct['width_ratio_vs_open'],
                'theta_hat': fit['theta_hat'], 'theta_ci': fit['ci95_theta'],
                'nu_over_sigma_hat': fit['nu_over_sigma_hat'],
                'nu_over_sigma_ci': fit['ci95_nu_over_sigma'],
            }, f'stress_variant.json:nonmarkov_bridge.results.{env}.{tier}', 'f4')

    table9a = md_table(
        ['Environment', 'Tier', 'open-only cov. (SE)', 'open width',
         'cross-event cov. (SE)', 'cross width', 'width ratio',
         '$\\tau$-stratified cov. (SE)', '$\\tau$-stratified width ratio'],
        bridge_rows,
    )

    bridge_fit_rows = []
    for env, env_label in env_labels:
        for tier in ['oracle', 'estimated']:
            fit = bridge['results'][env][tier]['cross_event']['component_full_fit']
            bridge_fit_rows.append([
                env_label, tier,
                f'{fit["theta_hat"]:.4f}',
                f'[{fit["ci95_theta"][0]:.3f}, {fit["ci95_theta"][1]:.3f}]',
                f'{fit["nu_over_sigma_hat"]:.3f}',
                f'[{fit["ci95_nu_over_sigma"][0]:.3f}, {fit["ci95_nu_over_sigma"][1]:.3f}]',
            ])
    table9b = md_table(
        ['Environment', 'Tier', '$\\hat\\theta$', '95% CI', '$\\hat\\nu/\\sigma$', '95% CI'],
        bridge_fit_rows,
    )

    tau_rows = []
    for env, env_label in env_labels:
        for tier in ['oracle', 'estimated']:
            rec = bridge['results'][env][tier]
            for tau in ['2', '65', '81']:
                oo = rec['open_only']['by_tau'][tau]
                ce_ = rec['cross_event']['by_tau'][tau]
                ct = rec['cross_event_tau']['by_tau'][tau]
                tau_rows.append([
                    env_label, tier, tau, str(oo['n']),
                    f'{oo["coverage"]:.3f}', f'{oo["mean_width"]:.4f}',
                    f'{ce_["coverage"]:.3f}', f'{ce_["mean_width"]:.4f}',
                    f'{ct["coverage"]:.3f}', f'{ct["mean_width"]:.4f}',
                ])
                put(f'l9.{env}.{tier}.tau{tau}', {
                    'n': oo['n'],
                    'open_cov': oo['coverage'], 'open_width': oo['mean_width'],
                    'cross_cov': ce_['coverage'], 'cross_width': ce_['mean_width'],
                    'tau_cov': ct['coverage'], 'tau_width': ct['mean_width'],
                }, f'stress_variant.json:nonmarkov_bridge.results.{env}.{tier}.{{open_only,cross_event,cross_event_tau}}.by_tau.{tau}', 'f4')
    table9c = md_table(
        ['Environment', 'Tier', '$\\tau$', '$n$', 'open cov.', 'open width',
         'cross cov.', 'cross width', '$\\tau$-strat cov.', '$\\tau$-strat width'],
        tau_rows,
    )

    same = bridge['diagnostics']['bridge_star']['same_endpoint']
    put('l9.same_endpoint', {
        'n_events': same['n_events'],
        'max_abs_pool_price_diff': same['max_abs_pool_price_diff'],
        'observed_paths_identical': same['observed_paths_identical'],
        'mean_qv_nu': same['mean_qv_nu'],
        'mean_qv_factor_nu': same['mean_qv_factor_nu'],
        'mean_qv_ratio': same['mean_qv_ratio'],
        'mean_theory_qv_nu': same['mean_theory_qv_nu'],
        'mean_theory_qv_factor_nu': same['mean_theory_qv_factor_nu'],
        'mean_theory_ratio': same['mean_theory_ratio'],
    }, 'stress_variant.json:nonmarkov_bridge.diagnostics.bridge_star.same_endpoint', 'f4')
    table9d = md_table(
        ['Quantity', 'Value'],
        [
            ['Same-endpoint events', str(same['n_events'])],
            ['Max abs pool-price difference', f'{same["max_abs_pool_price_diff"]:.1f}'],
            ['Observed paths bit-identical', str(same['observed_paths_identical'])],
            ['Mean latent QV at $\\nu_e$', f'{same["mean_qv_nu"]:.4e}'],
            ['Mean latent QV at $2\\nu_e$', f'{same["mean_qv_factor_nu"]:.4e}'],
            ['Empirical QV ratio', f'{same["mean_qv_ratio"]:.2f}'],
            ['Theoretical QV ratio', f'{same["mean_theory_ratio"]:.2f}'],
        ],
    )

    nq = bridge['results']['bridge_star']['oracle']['nu_quartile_coverage']
    nq_rows = []
    if nq.get('available'):
        for group in nq['groups']:
            nq_rows.append([
                str(group['group']), str(group['n']),
                f'{group["nu_min"]:.4f}', f'{group["nu_max"]:.4f}',
                f'{group["open_coverage"]:.3f}', f'{group["cross_coverage"]:.3f}',
                f'{group["open_width"]:.4f}', f'{group["cross_width"]:.4f}',
            ])
            put(f'l9.nu_quartile.{group["group"]}', group,
                f'stress_variant.json:nonmarkov_bridge.results.bridge_star.oracle.nu_quartile_coverage.groups.{group["group"]}', 'f4')
    table9e = md_table(
        ['$\\nu_e$ quartile', '$n$', '$\\nu$ min', '$\\nu$ max',
         'open cov.', 'cross cov.', 'open width', 'cross width'],
        nq_rows,
    )

    design = bridge['design']
    put('l9.design', {
        'nu_mult': design['nu_mult'],
        'zeta_sd': design['zeta_sd'],
        'nu_clip': design['nu_clip'],
        'theta_true_empirical': design['theta_true_empirical'],
    }, 'stress_variant.json:nonmarkov_bridge.design', 'f4')
    table9 = f"""# Table 9 — Non-Markov Brownian bridge stress

## Panel A. Coverage and width

{table9a}

Notes. 90% prediction intervals, 5-fold out-of-fold (fold = seed mod 5), 128
seeds × 4 windows = 512 paths, 1,536 double-endpoint events ($\\tau=2/65/81$, 512 each).
Nominal 0.90 ($\\alpha=0.1$). Coverage SEs are binomial over the 1,536 event
predictions and ignore within-path and across-fold dependence, so they are
lower bounds. `open-only` uses ordinary OPEN→OPEN one-step transitions and the
Markov $\\tau$-step variance $v_{{0e}}$; `cross-event` uses
$\\operatorname{{Var}}(r_e)=v_{{0e}}+\\theta\\sigma_e^2\\,\\mathrm{{gap}}_e$ with
$\\theta$ fitted by Gaussian MLE; `$\\tau$-stratified` uses a separate conformal
quantile per freeze length and no functional form in $\\tau$.

## Panel B. Bridge variance component

{table9b}

Notes. $\\theta=\\nu_{{\\rm mult}}^2\\,\\mathbb E[e^{{2\\zeta}}]$; the design value
is $\\theta_{{\\rm true}}={design['theta_true_empirical']:.3f}$ from the realized
event-level draws. The $\\nu=0$ placebo has $\\theta=0$ by construction. The
oracle-tier placebo interval contains zero; the estimated-tier placebo has a
small positive bias from fitting $T$ and $\\sigma$ on open transitions, as in
the $\\gamma=0$ placebo of Table 6.

## Panel C. Coverage by freeze length

{table9c}

Notes. The open-only deficit is present at every $\\tau$; it is not a single
long-freeze cell. The $\\tau$-stratified calibrator recovers within-$\\tau$
marginal coverage without assuming the bridge variance shape.

## Panel D. Same endpoints, different bridge volatility

{table9d}

Notes. Each selected event is re-simulated twice with the same realized
endpoints, the same standard Brownian bridge shape, and volatilities $\\nu_e$
and $2\\nu_e$. Freeze interiors are masked by the observation map, so the
observed pool-price paths are bit-identical; only the latent quadratic
variation changes. The empirical QV ratio is close to the theoretical
$2^2=4$. This is the empirical counterpart of the Theorem 4 construction, not
a coverage experiment: with endpoints pinned, no observable-data estimator can
distinguish the two bridge laws.

## Panel E. Coverage against the realized event-level volatility

{table9e}

Notes. The groups use the simulation-only $\\nu_e$, which is not observed by
either method; they are a dose-response diagnostic, not a deployable
stratification. Open-only coverage falls monotonically across all four
quartiles; cross-event calibration recovers marginal coverage but not
coverage conditional on the latent $\\nu_e$ (the highest quartile remains below
nominal). The latter is expected: $\\nu_e$ is not in the observed information
set.

## Bridge law

Brownian bridge pinned at the latent pre/post anchors. The post endpoint is
drawn from $N(m_+, v_{{\\rm ar}}+\\nu_e^2\\,\\mathrm{{gap}})$, with
$\\nu_e=\\sigma\\nu_{{\\rm mult}}e^{{\\zeta_e}}$,
$\\zeta_e\\sim N(0,{design['zeta_sd']:.2f}^2)$ clipped to
$[-{design['nu_clip']:.1f},{design['nu_clip']:.1f}]$, and
$\\nu_{{\\rm mult}}={design['nu_mult']:.2f}$. The interior is unobserved, so the
bridge volatility's observable footprint is the endpoint law; the same-endpoint
panel isolates its path-functional consequence.
"""
    (OUT / 'table9-bridge.md').write_text(table9)
else:
    (OUT / 'table9-bridge.md').write_text(
        '# Table 9 — Non-Markov Brownian bridge stress\n\n'
        'Not available: `stress_variant.json` has no `nonmarkov_bridge` block.\n'
    )

# ---------------------------------------------------------------------------
# Misspecification suite (L10)
# ---------------------------------------------------------------------------
MISS_VARIANTS = [
    ('correct', 'Correctly specified'),
    ('ar2_obs', 'AR(2) observation noise'),
    ('t_jumps', 'Student-$t(3)$ jumps'),
    ('regime_noise', 'Regime-label noise'),
    ('latent_regime', 'Hidden stale state'),
]
MISS_POLICIES = [
    ('Wide_Rebalance_2pct', 'Wide'),
    ('Narrow_Rebalance_1pct', 'Narrow'),
]
if miss is not None:
    def _miss_cell(variant, policy, model):
        m = miss['results'][variant][policy]['models'][model]
        fw = m['full_window']
        rl = m['regime_level']
        return {
            'bias': m['bias']['full'],
            'bias_extended': m['bias']['by_regime']['EXTENDED'],
            'g_within': fw['raw']['within_window']['coverage'],
            'g_loro': fw['raw']['leave_one_window_out']['coverage'],
            'gr_within': rl['raw']['within_regime']['coverage'],
            'gr_loro': rl['raw']['leave_one_regime_out']['coverage'],
            'plugin_g': fw['plugin_normal']['coverage'],
            'g_width_loro': fw['raw']['leave_one_window_out']['mean_width'],
            'gr_width_loro': rl['raw']['leave_one_regime_out']['mean_width'],
            'g_se': fw['raw']['within_window']['coverage_se'],
            'gr_se': rl['raw']['within_regime']['coverage_se'],
        }

    def _miss_paired(variant, policy):
        p = miss['results'][variant][policy]['paired']
        return {
            'g_studentized': p['full_window']['studentized'],
            'gr_studentized': p['regime']['studentized'],
            'plugin_g': p['full_window']['plugin_normal'],
        }

    panel_a_rows = []
    panel_b_rows = []
    for variant, vlabel in MISS_VARIANTS:
        for policy, ptag in MISS_POLICIES:
            block = panel_a_rows if ptag == 'Wide' else panel_b_rows
            for model in ['identified', 'naive']:
                c = _miss_cell(variant, policy, model)
                put(f'l10.{variant}.{ptag}.{model}', c,
                    f'stress/misspecification_suite.json:results.{variant}.{policy}.models.{model}', 'f4')
                block.append([
                    vlabel, model, f'{c["bias"]:+.4f}',
                    f'{c["g_within"]:.3f}', f'{c["g_loro"]:.3f}',
                    f'{c["gr_within"]:.3f}', f'{c["gr_loro"]:.3f}',
                    f'{c["plugin_g"]:.3f}', f'{c["g_width_loro"]:.4f}',
                ])

    panel_c_rows = []
    for variant, vlabel in MISS_VARIANTS:
        for policy, ptag in MISS_POLICIES:
            p = _miss_paired(variant, policy)
            put(f'l10.paired.{variant}.{ptag}', {
                'g_studentized': {k: p['g_studentized'][k] for k in ('diff', 'se', 'ci95')},
                'gr_studentized': {k: p['gr_studentized'][k] for k in ('diff', 'se', 'ci95')},
                'plugin_g': {k: p['plugin_g'][k] for k in ('diff', 'se', 'ci95')},
            }, f'stress/misspecification_suite.json:results.{variant}.{policy}.paired', 'f4')
            cell = []
            for key in ['g_studentized', 'gr_studentized', 'plugin_g']:
                rec = p[key]
                cell.append(
                    f'{rec["diff"]:+.3f} '
                    f'[{rec["ci95"][0]:+.3f}, {rec["ci95"][1]:+.3f}]'
                )
            panel_c_rows.append([vlabel, ptag] + cell)

    all_cells = {
        model: [
            _miss_cell(v, p, model)
            for v, _ in MISS_VARIANTS
            for p, _ in MISS_POLICIES
        ]
        for model in ['identified', 'naive']
    }
    ident = all_cells['identified']
    naive = all_cells['naive']
    pair_plugin = [
        _miss_paired(v, p)['plugin_g']['diff']
        for v, _ in MISS_VARIANTS
        for p, _ in MISS_POLICIES
    ]
    pair_plugin_lower = [
        _miss_paired(v, p)['plugin_g']['ci95'][0]
        for v, _ in MISS_VARIANTS
        for p, _ in MISS_POLICIES
    ]
    put('l10.range.ident_bias_abs',
        max(abs(c['bias']) for c in ident),
        'computed: max |bias| over stress/misspecification_suite.json identified cells', 'f4')
    put('l10.range.naive_bias',
        [min(c['bias'] for c in naive), max(c['bias'] for c in naive)],
        'computed: min/max bias over stress/misspecification_suite.json naive cells', 'f4')
    put('l10.range.plugin_identified',
        [min(c['plugin_g'] for c in ident), max(c['plugin_g'] for c in ident)],
        'computed: min/max plug-in G coverage over stress/misspecification_suite.json identified cells', 'f4')
    put('l10.range.plugin_naive',
        [min(c['plugin_g'] for c in naive), max(c['plugin_g'] for c in naive)],
        'computed: min/max plug-in G coverage over stress/misspecification_suite.json naive cells', 'f4')
    put('l10.range.paired_plugin',
        [min(pair_plugin), max(pair_plugin)],
        'computed: min/max paired identified - naive plug-in G coverage over stress/misspecification_suite.json', 'f4')
    put('l10.range.paired_plugin_ci_lower',
        min(pair_plugin_lower),
        'computed: min lower 95% bound over paired plug-in G gaps in stress/misspecification_suite.json', 'f4')
    put('l10.range.gr_loro_identified',
        [min(c['gr_loro'] for c in ident), max(c['gr_loro'] for c in ident)],
        'computed: min/max G_r LORO raw coverage over stress/misspecification_suite.json identified cells', 'f4')
    put('l10.range.gr_loro_naive',
        [min(c['gr_loro'] for c in naive), max(c['gr_loro'] for c in naive)],
        'computed: min/max G_r LORO raw coverage over stress/misspecification_suite.json naive cells', 'f4')

    cfg = miss['config']
    put('l10.design', {
        'n_cal_seeds': cfg['n_cal_seeds'],
        'n_eval_seeds': cfg['n_eval_seeds'],
        'reps': cfg['reps'],
        'alpha': cfg['alpha'],
    }, 'stress/misspecification_suite.json:config', 'f4')

    table10a = md_table(
        ['Variant', 'Model', 'bias $G$', '$G$ within', '$G$ LORO',
         '$G_r$ within', '$G_r$ LORO', 'plug-in $G$', '$G$ LORO width'],
        panel_a_rows,
    )
    table10b = md_table(
        ['Variant', 'Model', 'bias $G$', '$G$ within', '$G$ LORO',
         '$G_r$ within', '$G_r$ LORO', 'plug-in $G$', '$G$ LORO width'],
        panel_b_rows,
    )
    table10c = md_table(
        ['Variant', 'Policy', '$G$ studentized', '$G_r$ studentized', 'plug-in $G$'],
        panel_c_rows,
    )
    design = miss['design']
    hidden_occ = (
        design['latent_regime']['enter']
        / (design['latent_regime']['enter'] + design['latent_regime']['exit'])
    )
    hidden_enter = design['latent_regime']['enter']
    hidden_exit = design['latent_regime']['exit']
    hidden_occ_pct = 100.0 * hidden_occ
    table10d = md_table(
        ['Ingredient', 'Setting'],
        [
            ['AR(2) EXTENDED',
             f'$\\phi=({design["ar2_obs"]["extended"]["phi1"]:.2f}, '
             f'{design["ar2_obs"]["extended"]["phi2"]:.2f})$, '
             f'stationary sd {design["ar2_obs"]["extended"]["stationary_sd"]:.4f}'],
            ['AR(2) stale',
             f'$\\phi=({design["ar2_obs"]["stale"]["phi1"]:.2f}, '
             f'{design["ar2_obs"]["stale"]["phi2"]:.2f})$, '
             f'stationary sd {design["ar2_obs"]["stale"]["stationary_sd"]:.4f}'],
            ['Student-$t(3)$ jumps', 'jump probability 0.005; sd $3\\sigma$'],
            ['Regime-label noise', 'flip probability 0.15 on OPEN/EXTENDED'],
            ['Hidden stale state',
             f'enter {design["latent_regime"]["enter"]:.3f}, '
             f'exit {design["latent_regime"]["exit"]:.3f}, stationary '
             f'occupancy {100*hidden_occ:.1f}%, '
             f'burn-in {design["latent_regime"]["burn_in_hours"]} h'],
        ],
    )
    table10 = f"""# Table 10 — Misspecification suite

## Panel A. Wide policy: coverage under one perturbation magnitude per class

{table10a}

## Panel B. Narrow policy

{table10b}

## Panel C. Paired identified $-$ naive coverage

{table10c}

Notes. {cfg['n_cal_seeds']} calibration and {cfg['n_eval_seeds']} evaluation seeds
per window, four windows ({cfg['n_cal_seeds']*4} calibration and
{cfg['n_eval_seeds']*4} evaluation paths), {cfg['reps']} model draws per path;
nominal 90% ($\\alpha=0.1$). $G$ = full
window, $G_r$ = regime segment; within = same-population calibration, LORO =
leave-one-window-out ($G$) or leave-one-regime-out ($G_r$). Bias is the mean
model-mean minus truth over evaluation paths, in units of the initial position
value $V_0=10^5$ USD ($0.001=\\$100$). Panel C
reports identified minus naive with a 95% path bootstrap interval (2,000
resamples); $G$ and $G_r$ use the studentized score. The model side is fixed at
the G0 $\\kappa,\\mu$, jump probability, and $h$ for every variant; only the
data-generating process changes. `AR(2) observation noise` replaces the
EXTENDED AR(1) with $(\\phi_1,\\phi_2)=(0.70,0.20)$ and the stale AR(1) with
$(0.90,0.08)$, matching the stationary sd. `Student-$t(3)$ jumps` keeps the
0.005 jump probability and scales the jump to sd $3\\sigma$. `Regime-label
noise` flips 15% of OPEN and EXTENDED labels; the latent path and $P$ are
unchanged. `Hidden stale state` adds a simulation-only two-state chain during
dense hours (enter {hidden_enter:.2f}, exit {hidden_exit:.3f} per hour;
stationary occupancy {hidden_occ_pct:.1f}%) that
switches the observation law to the stale map; the model and policy still see
the calendar.

## Panel D. Design

{table10d}
"""
    (OUT / 'table10-misspec.md').write_text(table10)
else:
    (OUT / 'table10-misspec.md').write_text(
        '# Table 10 — Misspecification suite\n\n'
        'Not available: `results/stress/misspecification_suite.json` is missing.\n'
    )

# ---------------------------------------------------------------------------
# L11: semi-synthetic calibration (table generated by
# code/semi_synthetic_calibration.py; this block registers audit values)
# ---------------------------------------------------------------------------
calib_path = RES / 'semi_synthetic_calibration.json'
if calib_path.exists():
    calib = json.loads(calib_path.read_text())
    syms = list(calib['symbols'])
    for sym in syms:
        c = calib['symbols'][sym]
        put(f'l11.{sym}.sigma_h', c['sigma_h_regular'],
            f'semi_synthetic_calibration.json:symbols.{sym}.sigma_h_regular', 'f5')
        put(f'l11.{sym}.sigma_247', c['sigma_247'],
            f'semi_synthetic_calibration.json:symbols.{sym}.sigma_247', 'f5')
        put(f'l11.{sym}.kappa', c['kappa_ar1_regular'],
            f'semi_synthetic_calibration.json:symbols.{sym}.kappa_ar1_regular', 'f5')
        put(f'l11.{sym}.jump_prob', c['jump']['threshold_4sd']['prob'],
            f'semi_synthetic_calibration.json:symbols.{sym}.jump.threshold_4sd.prob', 'f4')
        put(f'l11.{sym}.jump_sd_ratio', c['jump']['threshold_4sd']['cond_sd_ratio'],
            f'semi_synthetic_calibration.json:symbols.{sym}.jump.threshold_4sd.cond_sd_ratio', 'f2')
        put(f'l11.{sym}.weekend_tau', c['freeze']['weekend']['median_h'],
            f'semi_synthetic_calibration.json:symbols.{sym}.freeze.weekend.median_h', 'f1')
        put(f'l11.{sym}.holiday_tau', c['freeze']['holiday']['median_h'],
            f'semi_synthetic_calibration.json:symbols.{sym}.freeze.holiday.median_h', 'f1')
        put(f'l11.{sym}.gap_ratio_weekend', c['benchmarks']['gap_ratio_weekend'],
            f'semi_synthetic_calibration.json:symbols.{sym}.benchmarks.gap_ratio_weekend', 'f2')
        put(f'l11.{sym}.gap_ratio_holiday', c['benchmarks']['gap_ratio_holiday'],
            f'semi_synthetic_calibration.json:symbols.{sym}.benchmarks.gap_ratio_holiday', 'f2')
        put(f'l11.{sym}.gap_over_g4_weekend', c['benchmarks']['gap_over_g4_weekend'],
            f'semi_synthetic_calibration.json:symbols.{sym}.benchmarks.gap_over_g4_weekend', 'f2')
        put(f'l11.{sym}.gap_over_g4_holiday', c['benchmarks']['gap_over_g4_holiday'],
            f'semi_synthetic_calibration.json:symbols.{sym}.benchmarks.gap_over_g4_holiday', 'f2')

    def _rng(key):
        vals = [calib['symbols'][s][key] for s in syms]
        return [min(vals), max(vals)]

    put('l11.range.sigma_h', _rng('sigma_h_regular'),
        'computed: min/max over semi_synthetic_calibration.json symbols.*.sigma_h_regular', 'f5')
    put('l11.range.sigma_247', _rng('sigma_247'),
        'computed: min/max over semi_synthetic_calibration.json symbols.*.sigma_247', 'f5')
    put('l11.range.kappa', _rng('kappa_ar1_regular'),
        'computed: min/max over semi_synthetic_calibration.json symbols.*.kappa_ar1_regular', 'f5')
    put('l11.range.jump_prob', [
        min(calib['symbols'][s]['jump']['threshold_4sd']['prob'] for s in syms),
        max(calib['symbols'][s]['jump']['threshold_4sd']['prob'] for s in syms),
    ], 'computed: min/max over semi_synthetic_calibration.json jump prob', 'f4')
    put('l11.range.jump_sd_ratio', [
        min(calib['symbols'][s]['jump']['threshold_4sd']['cond_sd_ratio'] for s in syms),
        max(calib['symbols'][s]['jump']['threshold_4sd']['cond_sd_ratio'] for s in syms),
    ], 'computed: min/max over semi_synthetic_calibration.json jump sd ratio', 'f2')
    put('l11.range.weekend_tau', [
        min(calib['symbols'][s]['freeze']['weekend']['median_h'] for s in syms),
        max(calib['symbols'][s]['freeze']['weekend']['median_h'] for s in syms),
    ], 'computed: min/max over semi_synthetic_calibration.json weekend freeze', 'f1')
    put('l11.range.holiday_tau', [
        min(calib['symbols'][s]['freeze']['holiday']['median_h'] for s in syms),
        max(calib['symbols'][s]['freeze']['holiday']['median_h'] for s in syms),
    ], 'computed: min/max over semi_synthetic_calibration.json holiday freeze', 'f1')
    put('l11.range.gap_ratio', [
        min(min(calib['symbols'][s]['benchmarks']['gap_ratio_weekend'],
                calib['symbols'][s]['benchmarks']['gap_ratio_holiday']) for s in syms),
        max(max(calib['symbols'][s]['benchmarks']['gap_ratio_weekend'],
                calib['symbols'][s]['benchmarks']['gap_ratio_holiday']) for s in syms),
    ], 'computed: min/max gap/(sigma_h*sqrt(tau)) over semi_synthetic_calibration.json symbols and windows', 'f2')
    put('l11.range.gap_over_g4', [
        min(min(calib['symbols'][s]['benchmarks']['gap_over_g4_weekend'],
                calib['symbols'][s]['benchmarks']['gap_over_g4_holiday']) for s in syms),
        max(max(calib['symbols'][s]['benchmarks']['gap_over_g4_weekend'],
                calib['symbols'][s]['benchmarks']['gap_over_g4_holiday']) for s in syms),
    ], 'computed: min/max gap/g_gamma=4 over semi_synthetic_calibration.json symbols and windows', 'f2')
    put('l11.p1', {
        'sigma_h': 0.008,
        'kappa': 0.003,
        'jump_prob': 0.005,
        'jump_sd_mult': 3.0,
        'weekend_tau': 65,
        'holiday_tau': 81,
    }, 'code/synthetic_env.py:SIGMA_BASE/KAPPA and simulate_market jump/calendar constants', 'f4')

# ---------------------------------------------------------------------------
# L12: real-path replay, 33-event case (§7.9) and 53-event robustness rerun
# (A27–A31).  Registered so the prose numbers are machine-checked instead of
# hand-checked; code/audit_numbers.py recomputes the same block from JSON.
# ---------------------------------------------------------------------------
for _tag, _fn in (('main33', 'real_data_replay.json'),
                  ('rob53', 'real_data_replay_53.json')):
    _p = RES / _fn
    if not _p.exists():
        continue
    _r = json.loads(_p.read_text())
    _s, _e1, _e2, _e3, _e5 = (_r['sample'], _r['e1'], _r['e2'],
                              _r['e3'], _r['e5'])
    put(f'l12.{_tag}.sample', {
        'window': _s['window'],
        'n_events': _s['n_events'], 'n_dates': _s['n_dates'],
        'n_weekend': _s['n_weekend'], 'n_holiday': _s['n_holiday'],
        'by_symbol': _s['by_symbol'], 'thin': _s['thin'],
        'nonthin': _s['nonthin'],
    }, f'{_fn}:sample')
    if 'calibration_support' in _r:
        _cs = _r['calibration_support']
        put(f'l12.{_tag}.calibration_support', {
            k: _cs[k] for k in (
                'min_branch_calibration', 'cal_events_range',
                'wide_branch1_cal_range', 'narrow_branch0_cal_range',
                'narrow_branch1_cal_range', 'nvda_cal_range',
                'wide_underpowered_folds', 'narrow_underpowered_folds')
        }, f'{_fn}:calibration_support')
    put(f'l12.{_tag}.e1', {
        'coverage': _e1['identified.raw.total_main']['coverage'],
        'coverage_ci95': _e1['identified.raw.total_main']['coverage_ci95'],
        'freeze_width_identified': _e1['identified.raw.freeze']['width'],
        'freeze_width_naive': _e1['naive.raw.freeze']['width'],
        'open_width_ratio_naive_over_identified':
            _e1['naive.raw.open']['width'] / _e1['identified.raw.open']['width'],
    }, f'{_fn}:e1', 'f4')
    put(f'l12.{_tag}.e1_total_by_symbol', {
        _m: {k: {'n': v['n'], 'coverage': v['coverage']} for k, v in
             _e1[f'{_m}.raw.total_main']['per_symbol'].items()}
        for _m in ('identified', 'naive')
    }, f'{_fn}:e1.*.total_main.per_symbol')
    _w = _e2['Wide_Rebalance_2pct']['identified']
    put(f'l12.{_tag}.e2_wide', {
        'n_branch1': _w['n_branch1'],
        'n_branch1_first_open': _w['n_branch1_first_open'],
        'underpowered': _w['underpowered'],
        'union_coverage': _w['union_coverage'],
        'width_ratio': _w['width_ratio'],
    }, f'{_fn}:e2.Wide_Rebalance_2pct.identified')
    _w = _e2['Narrow_Rebalance_1pct']['identified']
    put(f'l12.{_tag}.e2_narrow', {
        'n_branch1': _w['n_branch1'],
        'underpowered': _w['underpowered'],
        'union_coverage': _w['union_coverage'],
        'width_ratio': _w['width_ratio'],
    }, f'{_fn}:e2.Narrow_Rebalance_1pct.identified')
    put(f'l12.{_tag}.e3_loso_raw', {
        'identified': {k: v['coverage'] for k, v in
                       _e3['identified']['raw']['loso'].items()},
        'naive': {k: v['coverage'] for k, v in
                  _e3['naive']['raw']['loso'].items()},
        'identified_cqr': {k: v['coverage'] for k, v in
                           _e3['identified']['cqr']['loso'].items()},
    }, f'{_fn}:e3.*.loso')
    put(f'l12.{_tag}.e3_loro_raw', {
        'identified': {k: v['coverage'] for k, v in
                       _e3['identified']['raw']['loro'].items()},
        'naive': {k: v['coverage'] for k, v in
                  _e3['naive']['raw']['loro'].items()},
    }, f'{_fn}:e3.*.loro')
    put(f'l12.{_tag}.e3_duration_raw', {
        'identified': {k: v['coverage'] for k, v in
                       _e3['identified']['raw']['duration_transfer'].items()},
        'naive': {k: v['coverage'] for k, v in
                  _e3['naive']['raw']['duration_transfer'].items()},
    }, f'{_fn}:e3.*.duration_transfer')
    put(f'l12.{_tag}.e3_duration_by_symbol', {
        _m: _e3[_m]['raw']['duration_transfer_per_symbol'][
            'post2h_to_post24h'] for _m in ('identified', 'naive')
    }, f'{_fn}:e3.*.duration_transfer_per_symbol')
    put(f'l12.{_tag}.e3_symbol_conditional', {
        _m: _e3[_m]['raw']['symbol_conditional']
        for _m in ('identified', 'naive')
    }, f'{_fn}:e3.*.symbol_conditional')
    put(f'l12.{_tag}.e5_thin', {
        'identified': _e5['thin_nonthin']['identified.raw.total_main'],
        'naive': _e5['thin_nonthin']['naive.raw.total_main'],
    }, f'{_fn}:e5.thin_nonthin')
    put(f'l12.{_tag}.e5_fee', {
        _m: {_t: {'coverage': _e5['fee'][_t][_m]['total_main']['coverage'],
                  'width': _e5['fee'][_t][_m]['total_main']['width']}
             for _t in sorted(_e5['fee'])}
        for _m in ('identified', 'naive')
    }, f'{_fn}:e5.fee', 'f4')

# Nine-month supply audit behind the window choice (A27): 109 events, the
# 2h/24h A-layer counts, and the G1-R gate that fails.
_audit_p = RES / 'real_slice_9m_summary.json'
if _audit_p.exists():
    _a = json.loads(_audit_p.read_text())
    put('l12.nine_month', {
        'n_events': _a['n_events'],
        'n_by_symbol': _a['n_by_symbol'],
        'A_usable_2h_total': _a['A_usable_2h']['total'],
        'A_usable_2h_rate_pct': _a['A_usable_2h']['aggregate_rate_pct'],
        'A_usable_2h_dates': _a['A_usable_2h']['independent_calendar_events'],
        'A_usable_24h_total': _a['A_usable_24h']['total'],
        'A_usable_24h_rate_pct': _a['A_usable_24h']['aggregate_rate_pct'],
        'gate_usable_ge_90': _a['A_usable_2h']['gate']['usable_ge_90'],
    }, 'real_slice_9m_summary.json', None)

# ---------------------------------------------------------------------------
# General sensor-POMDP validation
# ---------------------------------------------------------------------------
gm = general['models']
gb = general['branch']
general_rows = []
for model in ('identified', 'naive'):
    rec = gm[model]
    loro = rec['scores']['raw']['leave_one_regime_out']
    model_label = 'identified transition' if model == 'identified' else 'fitted transition'
    general_rows.append([
        model_label, f"{rec['mean_absolute_return_error']:.3f}", f"{rec['plugin_normal']['coverage']:.3f}",
        f"{rec['scores']['raw']['full_window']['coverage']:.3f}",
        f"{loro['outage']['coverage']:.3f}",
    ])
    put(f'general.{model}', {
        'mean_absolute_return_error': rec['mean_absolute_return_error'],
        'plugin_coverage': rec['plugin_normal']['coverage'],
        'raw_full_coverage': rec['scores']['raw']['full_window']['coverage'],
        'raw_loro_outage_coverage': loro['outage']['coverage'],
    }, f'general_pomdp_check.json:models.{model}', 'f4')
    put(f'general.{model}.raw_regimes', {
        regime: {
            'within_coverage': rec['scores']['raw']['within_regime'][regime]['coverage'],
            'within_se': rec['scores']['raw']['within_regime'][regime]['coverage_se'],
            'loro_coverage': rec['scores']['raw']['leave_one_regime_out'][regime]['coverage'],
            'loro_se': rec['scores']['raw']['leave_one_regime_out'][regime]['coverage_se'],
        } for regime in ('normal', 'degraded', 'outage')
    }, f'general_pomdp_check.json:models.{model}.scores.raw', 'f4')

regime_rows = []
for regime in ('normal', 'degraded', 'outage'):
    row = [regime]
    for model in ('identified', 'naive'):
        rec = gm[model]['scores']['raw']
        row += [f"{rec['within_regime'][regime]['coverage']:.3f}",
                f"{rec['leave_one_regime_out'][regime]['coverage']:.3f}"]
    regime_rows.append(row)

branch_rows = []
for rule in ('marginal', 'predecision', 'branch_union'):
    rec = gb['rules'][rule]
    branch_rows.append([
        rule, f"{rec['coverage']:.3f}", f"{rec['per_branch']['0']['coverage']:.3f}",
        f"{rec['per_branch']['1']['coverage']:.3f}", f"{rec['mean_width']:.3f}",
    ])
    put(f'general.branch.{rule}', {
        'coverage': rec['coverage'],
        'branch0_coverage': rec['per_branch']['0']['coverage'],
        'branch0_se': rec['per_branch']['0']['coverage_se'],
        'branch1_coverage': rec['per_branch']['1']['coverage'],
        'branch1_se': rec['per_branch']['1']['coverage_se'],
        'mean_width': rec['mean_width'],
        'n_branch1': rec['per_branch']['1']['n_test'],
    }, f'general_pomdp_check.json:branch.rules.{rule}', 'f4')
put('general.branch.width_ratio',
    gb['rules']['branch_union']['mean_width'] / gb['rules']['marginal']['mean_width'],
    'computed: general_pomdp_check.json branch-union width / marginal width', 'f4')
put('general.branch.fractions', {
    'cal': gb['cal_branch_fraction'], 'test': gb['test_branch_fraction'],
    'cal_count': gb['cal_branch_count'], 'test_count': gb['test_branch_count'],
}, 'general_pomdp_check.json:branch', 'f4')

table13a = md_table(['Model', 'Mean absolute return error (G)', 'Plug-in coverage',
                     'Raw full-window coverage', 'Raw LORO outage coverage'], general_rows)
table13b = md_table(['Regime', 'Known-trans. within', 'Known-trans. LORO',
                     'Fitted-trans. within', 'Fitted-trans. LORO'], regime_rows)
table13c = md_table(['Rule', 'Overall coverage', 'Branch 0 coverage',
                     'Branch 1 coverage', 'Mean width (return units)'], branch_rows)
table13 = f"""# Table 13 — Second synthetic sensor-POMDP testbed

## Panel A. Return prediction error and interval coverage

{table13a}

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

{table13b}

Notes. LORO calibrates on the other two regimes. The outage boundary is retained
rather than pooled away.

## Panel C. Post-decision intervention-branch coverage (identified, raw)

{table13c}

Notes. Calibration has {gb['cal_branch_count']} intervention events among 128 paths;
evaluation has {gb['test_branch_count']} among 128 (the other {128-gb['test_branch_count']} are branch 0).
Marginal intervention-branch SE is {gb['rules']['marginal']['per_branch']['1']['coverage_se']:.3f};
branch-union SE is {gb['rules']['branch_union']['per_branch']['1']['coverage_se']:.3f}.
The pre-decision rule uses three groups from the sign and magnitude of the last
burn-in observation. Branch-union uses the larger calibration-side branch threshold,
so the test-time branch label is not needed to form the interval. Mean width is the
interval width in target-return (reward) units, not a coverage probability; the
1.89$\\times$ figure in the main text is the branch-union/marginal width ratio.
"""
(OUT / 'table13-general-pomdp.md').write_text(table13)

if 'out_of_sample' in theorem2:
    oos = theorem2['out_of_sample']
    oos_rows = []
    for direction in ('cal_tau2_test_tau81', 'cal_tau81_test_tau2'):
        for method, label in (('identified_raw', 'raw'),
                              ('identified_studentized', 'studentized'),
                              ('identified_cqr', 'CQR')):
            rec = oos[direction][method]
            summary = rec['summary']
            put(f'theorem2_oos.{direction}.{method}.summary', summary,
                f'theorem2_bounds.json:out_of_sample.{direction}.{method}.summary; code/theorem2_bounds.py', 'f4')
            ci = summary['coverage_mean_ci95']
            oos_rows.append([
                direction.replace('cal_tau', '').replace('_test_tau', '$\\to$'),
                label, f4(summary['mean_kappa_hat']), f4(summary['mean_two_b']),
                f4(summary['mean_cor22']),
                f"{summary['mean_coverage']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]",
                f5(summary['mean_width']),
                f"{summary['below_two_b_count']}/{summary['n_replicates']}",
            ])
    oos_cfg = theorem2['config']['out_of_sample']
    table14 = f"""# Table 14 — Three-sample out-of-sample check of Theorem 2

{md_table(['Direction', 'Score', 'Mean $\\hat\\kappa$', 'Mean 2B', 'Mean Cor 2.2', 'Final coverage [mean CI95]', 'Mean width', 'Below 2B'], oos_rows)}

Notes. Wide identified pipeline, {oos_cfg['replicates']} independent outer replicates. Each replicate uses
{oos_cfg['calibration_and_drift_seeds_each']} independently seeded calibration paths, the same number of
independent target-drift paths, and {oos_cfg['final_test_seeds']} final target-test paths. One path is drawn
per seed and window types cycle by seed. The radius uses calibration only; $\\hat\\kappa$ uses calibration and
the drift sample only; the final sample evaluates coverage and width without re-entering either calculation.
The bracket is a 95% interval for the mean final coverage across independent outer replicates, not a binomial
interval over pooled events. `Below 2B` counts replicates whose noisy final-sample coverage falls below the
2B bound; it is an empirical tightness diagnostic, not an exact PAC-violation frequency. Bounds
can be empty (zero) for any score. Width is in target-return units. Both $\\tau=2\\to81$ and $\\tau=81\\to2$ are shown.
"""
    (OUT / 'table14-theorem2-oos.md').write_text(table14)

# ---------------------------------------------------------------------------
# Derived JSON + number sources
# ---------------------------------------------------------------------------
(OUT / 'm3-derived.json').write_text(json.dumps({'values': V, 'sources': S}, indent=2) + '\n')

import hashlib
import re

SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _input_hashes(source):
    items = []
    seen = set()
    if 'general_pomdp_check.json' in source:
        script = ROOT / 'code' / 'general_pomdp_check.py'
        seen.add('code/general_pomdp_check.py')
        digest = hashlib.sha256(script.read_bytes()).hexdigest()[:16]
        items.append(f'`code/general_pomdp_check.py` `{digest}`')

    for name in re.findall(r'code/[A-Za-z0-9_.-]+\.py', source):
        path = ROOT / name
        if not path.exists():
            return name, 'MISSING'
        if name not in seen:
            seen.add(name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            items.append(f'`{name}` `{digest}`')

    names = re.findall(r'(?:(?:results)/)?((?:stress/)?[A-Za-z0-9_.*-]+\.json)', source)
    for name in names:
        if '*' in name:
            matches = sorted(RES.glob(name))
            if not matches:
                return name, 'MISSING'
            for path in matches:
                rel = f'results/{path.name}'
                if rel in seen:
                    continue
                seen.add(rel)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                items.append(f'`{rel}` `{digest}`')
        else:
            rel = f'results/{name}'
            if rel in seen:
                continue
            path = RES / name
            if not path.exists():
                return name, 'MISSING'
            seen.add(rel)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            items.append(f'`{rel}` `{digest}`')

    return None, '; '.join(items) if items else '—'


lines = ['# Number sources (M3)', '',
         'Generated by `code/build_tables.py` '
         f'(SHA256 `{SCRIPT_SHA[:16]}`). Each slug is a value in',
         '`paper/tables/m3-derived.json`; the `source` column gives a',
         'repository-relative path, and the hash columns give the first 16 hex',
         'characters of each input file (data or code) and of the generating',
         'script.', '',
         '**Audit scope.** The `audit_numbers.py` pass checks internal consistency',
         'of these derived values against `results/*.json` (201/201 at the last',
         'run). Separate guards verify the direct L1 and Theorem 2 bindings and',
         'the TNNLS Figure 7 and all SCIS figure sidecar hashes; these guards are',
         'not part of the 201 derived values. The audit does not read `main-draft.md`, `body.tex`,',
         'or rendered tables; display values and prose numbers are checked',
         'separately in `paper/audit-m5.md`.', '',
         '| Slug | Source | Input SHA256 | Script SHA256 |',
         '| --- | --- | --- | --- |']
for slug in sorted(S):
    missing, hashes = _input_hashes(S[slug]['json'])
    if missing:
        hashes = f'MISSING `{missing}`'
    lines.append(f'| `{slug}` | `{S[slug]["json"]}` | {hashes} | `{SCRIPT_SHA[:16]}` |')
(OUT / 'number-sources.md').write_text('\n'.join(lines) + '\n')

print('wrote', OUT)
for f in sorted(OUT.iterdir()):
    print(' ', f.name, f.stat().st_size)
print('\nkey values:')
for slug in ['l0.gap.naive', 'l0.gap.similarity', 'l0.weekend.windows', 'l1.range.regime_within',
             'l1.range.raw_loro', 'l1.range.ext_studentized_loro', 'l1.range.ext_cqr_loro',
             'l2.aggregate', 'l2.mondrian.range', 'l3.ext_phi.se', 'l6.gamma_star.oracle']:
    print(slug, '=', json.dumps(V[slug])[:300])
