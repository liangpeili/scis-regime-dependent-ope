#!/usr/bin/env python3
"""M5 number audit: recompute the derived values that are not direct JSON reads.

Every check is computed from results/*.json and compared with the corresponding
entry of paper/tables/m3-derived.json. Prints "checked N ok N bad 0" on success.
Run after code/build_tables.py.
"""
import collections, json, math, glob
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
D=json.load(open(ROOT/'paper/tables/m3-derived.json'))['values']
def load(f): return json.load(open(R/f))
base=load('coverage_gap_naive_global.json'); mon=load('coverage_gap_exp1_mondrian.json'); sim=load('coverage_gap.json')
traj=load('trajectory_value_check.json'); pol=load('policy_value_check.json'); obs=load('observable_h_check.json'); st=load('stress_variant.json')

def ci(p,n,z=1.96):
    se=math.sqrt(p*(1-p)/n); return [max(0,p-z*se),min(1,p+z*se)]
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]
checks={}
for r in '01234':
    checks[f'l0.{r}.ci95']=ci(base['heldout_regime_coverage'][r],base['n_test_per_regime'][r])
    checks[f'l0.{r}.mondrian_width_ratio']=mon['heldout_regime_width'][r]/base['heldout_regime_width'][r]
    checks[f'l0.{r}.similarity_width_ratio']=sim['heldout_regime_width'][r]/base['heldout_regime_width'][r]
checks['l0.weekend.strategy_range']=[min(base['heldout_strategy_coverage']['2'].values()),max(base['heldout_strategy_coverage']['2'].values())]
checks['l0.weekend.windows']={w:d['coverage'] for w,d in base['heldout_regime_by_window']['2'].items()}
checks['l0.halt.boot_se_range']=[min(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values()),max(d['boot_se'] for d in base['halt_calibration_radius_bootstrap'].values())]
checks['l0.halt.rel_se_range']=[min(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values()),max(d['rel_se_point'] for d in base['halt_calibration_radius_bootstrap'].values())]
checks['l0.halt.n_cal']=384

# L1
within=[traj['summary'][p][m]['regime_within_coverage'] for p in traj['summary'] for m in ['identified_raw','identified_studentized','identified_cqr','naive_raw','naive_studentized','naive_cqr']]
full=[traj['summary'][p][m]['full_window_within_coverage'] for p in traj['summary'] for m in ['identified_raw','identified_studentized','identified_cqr','naive_raw','naive_studentized','naive_cqr']]
raw=[traj['summary'][p][m]['regime_loro_coverage'] for p in traj['summary'] for m in ['identified_raw','naive_raw']]
est=[traj['summary'][p][m]['regime_loro_per_group_coverage']['EXTENDED'] for p in traj['summary'] for m in ['identified_studentized','naive_studentized']]
ecq=[traj['summary'][p][m]['regime_loro_per_group_coverage']['EXTENDED'] for p in traj['summary'] for m in ['identified_cqr','naive_cqr']]
checks['l1.range.regime_within']=[min(within),max(within)]
checks['l1.range.full_within']=[min(full),max(full)]
checks['l1.range.raw_loro']=[min(raw),max(raw)]
checks['l1.range.ext_studentized_loro']=[min(est),max(est)]
checks['l1.range.ext_cqr_loro']=[min(ecq),max(ecq)]
pg=[]
for p in traj['policies']:
    for model in ['ident','naive']:
        for m in ['raw','studentized','cqr']:
            pg+= [g['coverage'] for g in traj['policies'][p]['segment_wise'][model]['regime_level'][m]['within_regime']['per_group'].values()]
checks['l1.range.regime_within_per_group']=[min(pg),max(pg)]
ratio={}
for p in traj['policies']:
    pn='Wide' if p.startswith('Wide') else 'Narrow'
    for m in ['identified_studentized','identified_cqr']:
        rec=traj['policies'][p]['segment_wise']['ident']['regime_level'][m.split('_',1)[1]]
        ratio[f'{pn}.{m}']=rec['leave_one_regime_out']['mean_width']/rec['within_regime']['mean_width']
checks['l1.width_ratio.loro_within']=ratio

# L2
def branch(tag):
    bj=pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods'][tag]['within_tau2']['branch_union_joint']
    hold,reb=bj['branches']['hold'],bj['branches']['rebalance']
    triv=max(0.0,1-(1-bj['overall_marginal_coverage'])/reb['cal_fraction'])
    return {'overall_marginal':bj['overall_marginal_coverage'],'overall_union':bj['overall_union_coverage'],'overall_gain':bj['overall_coverage_gain'],'width_ratio':bj['width_ratio_vs_marginal'],'hold':{k:hold[k] for k in ('n_test','marginal_coverage','union_coverage')},'rebalance':{k:reb[k] for k in ('n_test','marginal_coverage','union_coverage','coverage_gain','coverage_gain_se','width_ratio_vs_marginal','taxonomy')},'trivial_bound':triv}
for tag in ['identified_raw','identified_studentized','identified_cqr']:
    checks[f'l2.{tag}.branch']=branch(tag)
import collections
cnt=collections.Counter(); blocks=cells=dom=0; min_cal_n=None; inf_cells=0
for p in pol['event_conformal']['policies']:
    for meth,rec in pol['event_conformal']['policies'][p]['methods'].items():
        for dr in ['within_tau2','within_tau81','cal_tau2_test_tau81','cal_tau81_test_tau2']:
            bj=rec.get(dr,{}).get('branch_union_joint')
            if not bj: continue
            blocks+=1; dom+=int(bj['coverage_dominance'])
            bu=rec.get(dr,{}).get('branch_union',{})
            inf_cells+=int(bu.get('n_inf_union',0))
            for b,br in bj['branches'].items(): cells+=1; cnt[br['taxonomy']]+=1
            for b,br in bu.get('branches',{}).items():
                if min_cal_n is None or br['cal_n']<min_cal_n: min_cal_n=br['cal_n']
checks['l2.aggregate']={'blocks':blocks,'cells':cells,'dominance_blocks':dom,'taxonomy':dict(cnt),'min_branch_cal_n':min_cal_n,'inf_union_cells':inf_cells}
cross={}
for meth in ['identified_raw','identified_studentized','identified_cqr']:
    bj=pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods'][meth]['cal_tau81_test_tau2']['branch_union_joint']
    cross[meth]={'marginal':bj['branches']['rebalance']['marginal_coverage'],'union':bj['branches']['rebalance']['union_coverage'],'width_ratio':bj['branches']['rebalance']['width_ratio_vs_marginal'],'taxonomy':bj['branches']['rebalance']['taxonomy']}
checks['l2.cross81to2']=cross
for meth in ['identified_raw','identified_studentized','identified_cqr']:
    bj=pol['event_conformal']['policies']['Wide_Rebalance_2pct']['methods'][meth]['within_tau2']['branch_union_joint']
    bci={}
    for b in ['hold','rebalance']:
        rec=bj['branches'][b]; n=rec['n_test']
        bci[b]={'n':n,
                'marginal':wilson(round(rec['marginal_coverage']*n),n),
                'union':wilson(round(rec['union_coverage']*n),n)}
    checks[f'l2.{meth}.branch_ci']=bci
mons=[]
for seed,sd in pol['event_conformal_seed_robustness'].items():
    for pname,pd in sd.items():
        if pname!='Wide_Rebalance_2pct': continue
        for meth in ['identified_raw','identified_studentized']:
            m=pd[meth]['mondrian']['within_tau2']
            for g in ['onset_action','onset_side']:
                mons.append((m[g]['post_rebalance_coverage'],seed,meth,g))
mons.sort(); checks['l2.mondrian.range']=[list(mons[0]),list(mons[-1])]

# L3
checks['l3.ext_phi.se']={'bootstrap':obs['uncertainty']['bootstrap']['ext_phi']['se'],'hessian':obs['uncertainty']['hessian_se']['ext_phi'],'q05':obs['uncertainty']['bootstrap']['ext_phi']['q05'],'q95':obs['uncertainty']['bootstrap']['ext_phi']['q95']}
checks['l3.open_sd.se']={'bootstrap':obs['uncertainty']['bootstrap']['open_sd']['se'],'hessian':obs['uncertainty']['hessian_se']['open_sd']}
files=['trajectory_value_check.json','trajectory_value_check_obs_h.json','trajectory_value_check_obs_h_z25.json','trajectory_value_check_obs_h_z40.json','trajectory_value_check_obs_h_sigma_open.json']
cs={}
for f in files:
    d=load(f)
    for p in d['summary']:
        pn='Wide' if p.startswith('Wide') else 'Narrow'
        for lvl,key in [('full','full_window_within_coverage'),('regime','regime_within_coverage')]:
            cs.setdefault(f'{pn}.{lvl}',[]).append(d['summary'][p]['identified_raw'][key])
checks['l3.coverage_sensitivity']={k:{'min':min(v),'max':max(v),'spread_pp':100*(max(v)-min(v))} for k,v in cs.items()}
# per-regime raw: spread across 5 specs for each policy/regime/method
pr={}
for f in files:
    d=load(f)
    for p in d['policies']:
        pn='Wide' if p.startswith('Wide') else 'Narrow'
        rec=d['policies'][p]['segment_wise']['ident']['regime_level']['raw']['within_regime']['per_group']
        for gname,g in rec.items(): pr.setdefault(f'{pn}.{gname}',[]).append(g['coverage'])
checks['l3.coverage_sensitivity.per_regime_raw']={k:100*(max(v)-min(v)) for k,v in pr.items()}
# all methods
pa={}
for f in files:
    d=load(f)
    for p in d['policies']:
        pn='Wide' if p.startswith('Wide') else 'Narrow'
        for model in ['ident','naive']:
            for meth in ['raw','studentized','cqr']:
                rec=d['policies'][p]['segment_wise'][model]['regime_level'][meth]['within_regime']['per_group']
                for gname,g in rec.items(): pa.setdefault(f'{pn}.{model}.{meth}.{gname}',[]).append(g['coverage'])
spreads={k:100*(max(v)-min(v)) for k,v in pa.items()}
mx=max(spreads.values()); arg=[k for k,v in spreads.items() if v==mx][0]
checks['l3.coverage_sensitivity.per_regime_all_methods']={'max_pp':mx,'argmax':arg.split('.'),'all':spreads}

# L6
for tier in ['gamma0','gamma_star']:
    for est in ['oracle','estimated']:
        r=st['results'][tier][est]
        checks[f'l6.{tier}.{est}']={'open_cov':r['open_only']['coverage'],'open_width':r['open_only']['mean_width'],'cross_cov':r['cross_event']['coverage'],'cross_width':r['cross_event']['mean_width'],'width_ratio':r['cross_event']['width_ratio_vs_open'],'gamma_hat':r['cross_event']['gamma2_full_fit']['gamma_hat'],'gamma_ci':r['cross_event']['gamma2_full_fit']['ci95_gamma']}

# L9 (non-Markov bridge)
br=st.get('nonmarkov_bridge',{})
if br.get('available'):
    for env in ['bridge0','bridge_star']:
        for est in ['oracle','estimated']:
            r=br['results'][env][est]
            fit=r['cross_event']['component_full_fit']
            checks[f'l9.{env}.{est}']={
                'open_cov':r['open_only']['coverage'],
                'open_width':r['open_only']['mean_width'],
                'cross_cov':r['cross_event']['coverage'],
                'cross_width':r['cross_event']['mean_width'],
                'cross_width_ratio':r['cross_event']['width_ratio_vs_open'],
                'tau_cov':r['cross_event_tau']['coverage'],
                'tau_width':r['cross_event_tau']['mean_width'],
                'tau_width_ratio':r['cross_event_tau']['width_ratio_vs_open'],
                'theta_hat':fit['theta_hat'],
                'theta_ci':fit['ci95_theta'],
                'nu_over_sigma_hat':fit['nu_over_sigma_hat'],
                'nu_over_sigma_ci':fit['ci95_nu_over_sigma'],
            }
            for tau in ['2','65','81']:
                oo=r['open_only']['by_tau'][tau]; ce=r['cross_event']['by_tau'][tau]; ct=r['cross_event_tau']['by_tau'][tau]
                checks[f'l9.{env}.{est}.tau{tau}']={
                    'n':oo['n'],'open_cov':oo['coverage'],'open_width':oo['mean_width'],
                    'cross_cov':ce['coverage'],'cross_width':ce['mean_width'],
                    'tau_cov':ct['coverage'],'tau_width':ct['mean_width'],
                }
    same=br['diagnostics']['bridge_star']['same_endpoint']
    checks['l9.same_endpoint']={
        'n_events':same['n_events'],
        'max_abs_pool_price_diff':same['max_abs_pool_price_diff'],
        'observed_paths_identical':same['observed_paths_identical'],
        'mean_qv_nu':same['mean_qv_nu'],
        'mean_qv_factor_nu':same['mean_qv_factor_nu'],
        'mean_qv_ratio':same['mean_qv_ratio'],
        'mean_theory_qv_nu':same['mean_theory_qv_nu'],
        'mean_theory_qv_factor_nu':same['mean_theory_qv_factor_nu'],
        'mean_theory_ratio':same['mean_theory_ratio'],
    }
    nq=br['results']['bridge_star']['oracle']['nu_quartile_coverage']
    if nq.get('available'):
        for group in nq['groups']:
            checks[f'l9.nu_quartile.{group["group"]}']=group
    dg=br['design']
    checks['l9.design']={k:dg[k] for k in ('nu_mult','zeta_sd','nu_clip','theta_true_empirical')}

# L10 (misspecification suite)
msp_path=R/'stress'/'misspecification_suite.json'
if msp_path.exists():
    msp=json.load(open(msp_path))
    for variant,vrec in msp['results'].items():
        for policy,prec in vrec.items():
            ptag='Wide' if policy.startswith('Wide') else 'Narrow'
            for model in ['identified','naive']:
                m=prec['models'][model]; fw=m['full_window']; rl=m['regime_level']
                checks[f'l10.{variant}.{ptag}.{model}']={
                    'bias':m['bias']['full'],
                    'bias_extended':m['bias']['by_regime']['EXTENDED'],
                    'g_within':fw['raw']['within_window']['coverage'],
                    'g_loro':fw['raw']['leave_one_window_out']['coverage'],
                    'gr_within':rl['raw']['within_regime']['coverage'],
                    'gr_loro':rl['raw']['leave_one_regime_out']['coverage'],
                    'plugin_g':fw['plugin_normal']['coverage'],
                    'g_width_loro':fw['raw']['leave_one_window_out']['mean_width'],
                    'gr_width_loro':rl['raw']['leave_one_regime_out']['mean_width'],
                    'g_se':fw['raw']['within_window']['coverage_se'],
                    'gr_se':rl['raw']['within_regime']['coverage_se'],
                }
            p=prec['paired']
            checks[f'l10.paired.{variant}.{ptag}']={
                'g_studentized':{k:p['full_window']['studentized'][k] for k in ('diff','se','ci95')},
                'gr_studentized':{k:p['regime']['studentized'][k] for k in ('diff','se','ci95')},
                'plugin_g':{k:p['full_window']['plugin_normal'][k] for k in ('diff','se','ci95')},
            }
    cfg=msp['config']
    checks['l10.design']={k:cfg[k] for k in ('n_cal_seeds','n_eval_seeds','reps','alpha')}
    all_cells={model:[msp['results'][v][p]['models'][model] for v in msp['results'] for p in msp['results'][v]] for model in ['identified','naive']}
    def _cell(m):
        fw=m['full_window']; rl=m['regime_level']
        return {'bias':m['bias']['full'],
                'plugin_g':fw['plugin_normal']['coverage'],
                'gr_loro':rl['raw']['leave_one_regime_out']['coverage']}
    ident=[_cell(m) for m in all_cells['identified']]
    naive=[_cell(m) for m in all_cells['naive']]
    pair_plugin=[msp['results'][v][p]['paired']['full_window']['plugin_normal']['diff'] for v in msp['results'] for p in msp['results'][v]]
    pair_plugin_lower=[msp['results'][v][p]['paired']['full_window']['plugin_normal']['ci95'][0] for v in msp['results'] for p in msp['results'][v]]
    checks['l10.range.ident_bias_abs']=max(abs(c['bias']) for c in ident)
    checks['l10.range.naive_bias']=[min(c['bias'] for c in naive),max(c['bias'] for c in naive)]
    checks['l10.range.plugin_identified']=[min(c['plugin_g'] for c in ident),max(c['plugin_g'] for c in ident)]
    checks['l10.range.plugin_naive']=[min(c['plugin_g'] for c in naive),max(c['plugin_g'] for c in naive)]
    checks['l10.range.paired_plugin']=[min(pair_plugin),max(pair_plugin)]
    checks['l10.range.paired_plugin_ci_lower']=min(pair_plugin_lower)
    checks['l10.range.gr_loro_identified']=[min(c['gr_loro'] for c in ident),max(c['gr_loro'] for c in ident)]
    checks['l10.range.gr_loro_naive']=[min(c['gr_loro'] for c in naive),max(c['gr_loro'] for c in naive)]

# L11 (semi-synthetic calibration)
calib_path=R/'semi_synthetic_calibration.json'
if calib_path.exists():
    calib=json.load(open(calib_path))
    syms=list(calib['symbols'])
    for sym in syms:
        c=calib['symbols'][sym]
        checks[f'l11.{sym}.sigma_h']=c['sigma_h_regular']
        checks[f'l11.{sym}.sigma_247']=c['sigma_247']
        checks[f'l11.{sym}.kappa']=c['kappa_ar1_regular']
        checks[f'l11.{sym}.jump_prob']=c['jump']['threshold_4sd']['prob']
        checks[f'l11.{sym}.jump_sd_ratio']=c['jump']['threshold_4sd']['cond_sd_ratio']
        checks[f'l11.{sym}.weekend_tau']=c['freeze']['weekend']['median_h']
        checks[f'l11.{sym}.holiday_tau']=c['freeze']['holiday']['median_h']
        for w in ('weekend','holiday'):
            checks[f'l11.{sym}.gap_ratio_{w}']=c['benchmarks'][f'gap_ratio_{w}']
            checks[f'l11.{sym}.gap_over_g4_{w}']=c['benchmarks'][f'gap_over_g4_{w}']
    def rng(vals): return [min(vals),max(vals)]
    checks['l11.range.sigma_h']=rng([calib['symbols'][s]['sigma_h_regular'] for s in syms])
    checks['l11.range.sigma_247']=rng([calib['symbols'][s]['sigma_247'] for s in syms])
    checks['l11.range.kappa']=rng([calib['symbols'][s]['kappa_ar1_regular'] for s in syms])
    checks['l11.range.jump_prob']=rng([calib['symbols'][s]['jump']['threshold_4sd']['prob'] for s in syms])
    checks['l11.range.jump_sd_ratio']=rng([calib['symbols'][s]['jump']['threshold_4sd']['cond_sd_ratio'] for s in syms])
    checks['l11.range.weekend_tau']=rng([calib['symbols'][s]['freeze']['weekend']['median_h'] for s in syms])
    checks['l11.range.holiday_tau']=rng([calib['symbols'][s]['freeze']['holiday']['median_h'] for s in syms])
    checks['l11.range.gap_ratio']=rng([calib['symbols'][s]['benchmarks'][f'gap_ratio_{w}'] for s in syms for w in ('weekend','holiday')])
    checks['l11.range.gap_over_g4']=rng([calib['symbols'][s]['benchmarks'][f'gap_over_g4_{w}'] for s in syms for w in ('weekend','holiday')])
    # P1 baseline: code constants + calendar-derived freeze lengths
    import sys, re
    sys.path.insert(0,str(ROOT/'code'))
    import synthetic_env as se
    src=(ROOT/'code'/'synthetic_env.py').read_text()
    jm=re.search(r'rng\.normal\(0\.0,\s*([0-9.]+)\s*\*\s*sigma\)\s*if\s*rng\.random\(\)\s*<\s*([0-9.]+)',src)
    frozen_regs={se.REGIME_WEEKEND,se.REGIME_HOLIDAY,se.REGIME_HALT}
    # one day past the window so the final Fri 16:00 -> Mon 09:00 run closes
    regs=[se.calendar_regime(d,h) for d in range(se.WINDOW_DAYS+1) for h in range(24)]
    runs=[]; start=None
    for i,r in enumerate(regs+[None]):
        f=r in frozen_regs
        if f and start is None: start=i
        elif not f and start is not None:
            runs.append((start,i-start)); start=None
    holiday_run=next(n for st,n in runs if st<=se.HOLIDAY_DAY*24<st+n)
    weekend_run=max(n for st,n in runs if not (st<=se.HOLIDAY_DAY*24<st+n))
    checks['l11.p1']={'sigma_h':se.SIGMA_BASE,'kappa':se.KAPPA,
                      'jump_prob':float(jm.group(2)),'jump_sd_mult':float(jm.group(1)),
                      'weekend_tau':weekend_run,'holiday_tau':holiday_run}

# L12 (real-path replay: 33-event case + 53-event robustness rerun)
for _tag,_fn in (('main33','real_data_replay.json'),
                 ('rob53','real_data_replay_53.json')):
    _p=R/_fn
    if not _p.exists(): continue
    _r=json.load(open(_p))
    _s,_e1,_e2,_e3,_e5=(_r['sample'],_r['e1'],_r['e2'],_r['e3'],_r['e5'])
    checks[f'l12.{_tag}.sample']={
        'window':_s['window'],
        'n_events':_s['n_events'],'n_dates':_s['n_dates'],
        'n_weekend':_s['n_weekend'],'n_holiday':_s['n_holiday'],
        'by_symbol':_s['by_symbol'],'thin':_s['thin'],'nonthin':_s['nonthin']}
    if 'calibration_support' in _r:
        _folds=[]
        for _date in sorted({e['date'] for e in _r['events']}):
            _cal=[e for e in _r['events'] if e['date']!=_date]
            _folds.append((_cal,{
                p:collections.Counter(e['branch'][p] for e in _cal)
                for p in ('Wide_Rebalance_2pct','Narrow_Rebalance_1pct')}))
        _wide1=[b['Wide_Rebalance_2pct'][1] for c,b in _folds]
        _narrow0=[b['Narrow_Rebalance_1pct'][0] for c,b in _folds]
        _narrow1=[b['Narrow_Rebalance_1pct'][1] for c,b in _folds]
        _nvda=[sum(e['symbol']=='NVDAon' for e in c) for c,b in _folds]
        checks[f'l12.{_tag}.calibration_support']={
            'min_branch_calibration':9,
            'cal_events_range':[min(len(c) for c,b in _folds),max(len(c) for c,b in _folds)],
            'wide_branch1_cal_range':[min(_wide1),max(_wide1)],
            'narrow_branch0_cal_range':[min(_narrow0),max(_narrow0)],
            'narrow_branch1_cal_range':[min(_narrow1),max(_narrow1)],
            'nvda_cal_range':[min(_nvda),max(_nvda)],
            'wide_underpowered_folds':sum(min(b['Wide_Rebalance_2pct'].values())<9 for c,b in _folds),
            'narrow_underpowered_folds':sum(min(b['Narrow_Rebalance_1pct'].values())<9 for c,b in _folds)}
    checks[f'l12.{_tag}.e1']={
        'coverage':_e1['identified.raw.total_main']['coverage'],
        'coverage_ci95':_e1['identified.raw.total_main']['coverage_ci95'],
        'freeze_width_identified':_e1['identified.raw.freeze']['width'],
        'freeze_width_naive':_e1['naive.raw.freeze']['width'],
        'open_width_ratio_naive_over_identified':
            _e1['naive.raw.open']['width']/_e1['identified.raw.open']['width']}
    checks[f'l12.{_tag}.e1_total_by_symbol']={
        _m:{k:{'n':v['n'],'coverage':v['coverage']} for k,v in
            _e1[f'{_m}.raw.total_main']['per_symbol'].items()}
        for _m in ('identified','naive')}
    _w=_e2['Wide_Rebalance_2pct']['identified']
    checks[f'l12.{_tag}.e2_wide']={
        'n_branch1':_w['n_branch1'],
        'n_branch1_first_open':_w['n_branch1_first_open'],
        'underpowered':_w['underpowered'],
        'union_coverage':_w['union_coverage'],
        'width_ratio':_w['width_ratio']}
    _w=_e2['Narrow_Rebalance_1pct']['identified']
    checks[f'l12.{_tag}.e2_narrow']={
        'n_branch1':_w['n_branch1'],
        'underpowered':_w['underpowered'],
        'union_coverage':_w['union_coverage'],
        'width_ratio':_w['width_ratio']}
    checks[f'l12.{_tag}.e3_loso_raw']={
        'identified':{k:v['coverage'] for k,v in _e3['identified']['raw']['loso'].items()},
        'naive':{k:v['coverage'] for k,v in _e3['naive']['raw']['loso'].items()},
        'identified_cqr':{k:v['coverage'] for k,v in _e3['identified']['cqr']['loso'].items()}}
    checks[f'l12.{_tag}.e3_loro_raw']={
        'identified':{k:v['coverage'] for k,v in _e3['identified']['raw']['loro'].items()},
        'naive':{k:v['coverage'] for k,v in _e3['naive']['raw']['loro'].items()}}
    checks[f'l12.{_tag}.e3_duration_raw']={
        'identified':{k:v['coverage'] for k,v in _e3['identified']['raw']['duration_transfer'].items()},
        'naive':{k:v['coverage'] for k,v in _e3['naive']['raw']['duration_transfer'].items()}}
    checks[f'l12.{_tag}.e3_duration_by_symbol']={
        _m:_e3[_m]['raw']['duration_transfer_per_symbol']['post2h_to_post24h']
        for _m in ('identified','naive')}
    checks[f'l12.{_tag}.e3_symbol_conditional']={
        _m:_e3[_m]['raw']['symbol_conditional'] for _m in ('identified','naive')}
    checks[f'l12.{_tag}.e5_thin']={
        'identified':_e5['thin_nonthin']['identified.raw.total_main'],
        'naive':_e5['thin_nonthin']['naive.raw.total_main']}
    checks[f'l12.{_tag}.e5_fee']={
        _m:{_t:{'coverage':_e5['fee'][_t][_m]['total_main']['coverage'],
                'width':_e5['fee'][_t][_m]['total_main']['width']}
            for _t in sorted(_e5['fee'])}
        for _m in ('identified','naive')}

_audit_p=R/'real_slice_9m_summary.json'
if _audit_p.exists():
    _a=json.load(open(_audit_p))
    checks['l12.nine_month']={
        'n_events':_a['n_events'],
        'n_by_symbol':_a['n_by_symbol'],
        'A_usable_2h_total':_a['A_usable_2h']['total'],
        'A_usable_2h_rate_pct':_a['A_usable_2h']['aggregate_rate_pct'],
        'A_usable_2h_dates':_a['A_usable_2h']['independent_calendar_events'],
        'A_usable_24h_total':_a['A_usable_24h']['total'],
        'A_usable_24h_rate_pct':_a['A_usable_24h']['aggregate_rate_pct'],
        'gate_usable_ge_90':_a['A_usable_2h']['gate']['usable_ge_90']}

# General sensor-POMDP validation
_general=load('general_pomdp_check.json')
_gm=_general['models']; _gb=_general['branch']
for _model in ('identified','naive'):
    _rec=_gm[_model]; _loro=_rec['scores']['raw']['leave_one_regime_out']
    checks[f'general.{_model}']={
        'mean_absolute_return_error':_rec['mean_absolute_return_error'],
        'plugin_coverage':_rec['plugin_normal']['coverage'],
        'raw_full_coverage':_rec['scores']['raw']['full_window']['coverage'],
        'raw_loro_outage_coverage':_loro['outage']['coverage']}
    checks[f'general.{_model}.raw_regimes']={
        regime:{
            'within_coverage':_rec['scores']['raw']['within_regime'][regime]['coverage'],
            'within_se':_rec['scores']['raw']['within_regime'][regime]['coverage_se'],
            'loro_coverage':_rec['scores']['raw']['leave_one_regime_out'][regime]['coverage'],
            'loro_se':_rec['scores']['raw']['leave_one_regime_out'][regime]['coverage_se']}
        for regime in ('normal','degraded','outage')}
for _rule in ('marginal','predecision','branch_union'):
    _rec=_gb['rules'][_rule]
    checks[f'general.branch.{_rule}']={
        'coverage':_rec['coverage'],
        'branch0_coverage':_rec['per_branch']['0']['coverage'],
        'branch0_se':_rec['per_branch']['0']['coverage_se'],
        'branch1_coverage':_rec['per_branch']['1']['coverage'],
        'branch1_se':_rec['per_branch']['1']['coverage_se'],
        'mean_width':_rec['mean_width'],
        'n_branch1':_rec['per_branch']['1']['n_test']}
checks['general.branch.width_ratio']=_gb['rules']['branch_union']['mean_width']/_gb['rules']['marginal']['mean_width']
checks['general.branch.fractions']={
    'cal':_gb['cal_branch_fraction'],'test':_gb['test_branch_fraction'],
    'cal_count':_gb['cal_branch_count'],'test_count':_gb['test_branch_count']}

# Figure 7 provenance is separate from the derived-value count.
import hashlib
_fig_paths={
    'generator':ROOT/'code/make_figures.py',
    'input':R/'general_pomdp_check.json',
    'pdf':ROOT/'paper/figures/fig7-general-pomdp.pdf',
    'png':ROOT/'paper/figures/fig7-general-pomdp.png'}
_fig=json.load(open(ROOT/'paper/figures/fig7-general-pomdp.json'))
if _fig.get('comparison_scope')!=_general['config']['comparison_scope']:
    raise SystemExit('Figure 7 provenance mismatch: comparison_scope')
for _key,_path in _fig_paths.items():
    _sha=hashlib.sha256(_path.read_bytes()).hexdigest()
    if _fig['sha256'].get(_key)!=_sha:
        raise SystemExit(f'Figure 7 provenance mismatch: {_key}')
print('figure7_provenance ok')

# Theorem 2 direct-JSON bindings and SCIS figure provenance are separate guards.
_t2=load('theorem2_bounds.json')
_t2_count=0
for _policy,_methods in _t2['policies'].items():
    for _method,_rec in _methods.items():
        _slug=f'theorem2.{_policy}.{_method}.summary'
        if D.get(_slug)!=_rec['summary']:
            raise SystemExit(f'Theorem 2 binding mismatch: {_slug}')
        _t2_count+=1
print(f'theorem2_binding ok ({_t2_count} summaries)')

if 'out_of_sample' in _t2:
    def _oos_audit_summary(rows):
        import numpy as np
        coverage=np.array([r['coverage'] for r in rows])
        se=float(coverage.std(ddof=1)/math.sqrt(len(rows)))
        out={
            'n_replicates':len(rows),
            'mean_coverage':float(coverage.mean()),
            'coverage_se_across_replicates':se,
            'coverage_mean_ci95':[max(0.0,float(coverage.mean())-1.96*se),min(1.0,float(coverage.mean())+1.96*se)],
            'coverage_q05':float(np.quantile(coverage,0.05)),
            'coverage_q95':float(np.quantile(coverage,0.95))}
        for key in ('kappa_hat','two_a','two_b','cor22','mean_width'):
            vals=np.array([r[key] for r in rows])
            name='width' if key=='mean_width' else key
            out[f'mean_{name}']=float(vals.mean())
            out[f'min_{name}']=float(vals.min())
            out[f'max_{name}']=float(vals.max())
        out['mean_tightness_two_b']=float(np.mean([r['coverage']-r['two_b'] for r in rows]))
        out['mean_tightness_cor22']=float(np.mean([r['coverage']-r['cor22'] for r in rows]))
        for key in ('two_b','cor22'):
            count=sum(r[f'below_{key}'] for r in rows)
            out[f'below_{key}_count']=count
            out[f'below_{key}_fraction']=count/len(rows)
            out[f'below_{key}_fraction_ci95']=wilson(count,len(rows))
        return out
    for _direction,_methods in _t2['out_of_sample'].items():
        for _method,_rec in _methods.items():
            checks[f'theorem2_oos.{_direction}.{_method}.summary']=_oos_audit_summary(_rec['replicates'])

_l1_count=0
for _policy in ('Wide_Rebalance_2pct','Narrow_Rebalance_1pct'):
    _short='Wide' if _policy.startswith('Wide') else 'Narrow'
    for _method in ('identified_raw','identified_studentized','identified_cqr','naive_raw','naive_studentized','naive_cqr'):
        _rec=traj['summary'][_policy][_method]
        _pairs={
            f'l1.{_short}.{_method}.cov':[_rec[k] for k in ('full_window_within_coverage','full_window_loro_coverage','regime_within_coverage','regime_loro_coverage')],
            f'l1.{_short}.{_method}.width':[_rec[k] for k in ('full_window_loro_width','regime_loro_width')]}
        for _slug,_value in _pairs.items():
            if D.get(_slug)!=_value:
                raise SystemExit(f'L1 direct binding mismatch: {_slug}')
            _l1_count+=1
    for _method in ('identified_plugin_normal','naive_plugin_normal'):
        _rec=traj['summary'][_policy][_method]
        _pairs={
            f'l1.{_short}.{_method}.cov':[_rec[k] for k in ('full_window_coverage','regime_coverage')],
            f'l1.{_short}.{_method}.width':[_rec[k] for k in ('full_window_width','regime_width')]}
        for _slug,_value in _pairs.items():
            if D.get(_slug)!=_value:
                raise SystemExit(f'L1 direct binding mismatch: {_slug}')
            _l1_count+=1
print(f'l1_direct_binding ok ({_l1_count} values)')

_scis_out=ROOT/'paper/figures/scis'
_scis_generators={
    'fig0-observation-boundary':'make_scis_boundary.py',
    'fig1-regime-timeline':'make_scis_fig1.py',
    'fig3-l0-coverage':'make_scis_figures.py',
    'fig4-l1-boundary':'make_scis_figures.py',
    'fig5-l2-branches':'make_scis_figures.py',
    'fig6-l3-h':'make_scis_figures.py',
    'fig7-general-pomdp':'make_scis_figures.py'}
_scis_inputs={
    'fig1-regime-timeline':['code/synthetic_env.py'],
    'fig3-l0-coverage':['results/coverage_gap_naive_global.json','results/coverage_gap_exp1_mondrian.json'],
    'fig4-l1-boundary':['results/trajectory_value_check.json'],
    'fig5-l2-branches':['results/policy_value_check.json'],
    'fig6-l3-h':['results/observable_h_check.json','results/trajectory_value_check.json','results/trajectory_value_check_obs_h.json','results/trajectory_value_check_obs_h_z25.json','results/trajectory_value_check_obs_h_z40.json','results/trajectory_value_check_obs_h_sigma_open.json'],
    'fig7-general-pomdp':['results/general_pomdp_check.json']}
for _stem,_generator in _scis_generators.items():
    _sidecar=json.load(open(_scis_out/f'{_stem}.json'))
    if _sidecar.get('figure')!=_stem:
        raise SystemExit(f'SCIS figure provenance mismatch: {_stem} name')
    _paths={'generator':ROOT/'code'/(_generator),'pdf':_scis_out/f'{_stem}.pdf','png':_scis_out/f'{_stem}.png'}
    for _key,_path in _paths.items():
        if _sidecar['sha256'].get(_key)!=hashlib.sha256(_path.read_bytes()).hexdigest():
            raise SystemExit(f'SCIS figure provenance mismatch: {_stem} {_key}')
    _input_names=_scis_inputs.get(_stem,[])
    if len(_input_names)==1:
        _expected={'input':hashlib.sha256((ROOT/_input_names[0]).read_bytes()).hexdigest()}
    elif _input_names:
        _expected={'inputs':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in _input_names}}
    else:
        _expected={}
    for _key,_value in _expected.items():
        if _sidecar['sha256'].get(_key)!=_value:
            raise SystemExit(f'SCIS figure provenance mismatch: {_stem} {_key}')
print(f'scis_figure_provenance ok ({len(_scis_generators)} figures)')

# compare
def close(a,b):
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool): return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12)
    return a==b
ok=bad=0
for k,v in checks.items():
    if k not in D: print('MISSING DERIVED',k); bad+=1; continue
    if close(v,D[k]): ok+=1
    else:
        bad+=1; print('DIFF',k); print('  computed',json.dumps(v)[:300]); print('  derived ',json.dumps(D[k])[:300])
print('checked',ok+bad,'ok',ok,'bad',bad)
