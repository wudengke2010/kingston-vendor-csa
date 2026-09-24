# -*- coding: utf-8 -*-
"""
KINGSTON v8 重分析 (回应第五轮逐句评审, 2026-09-24)

阻断级修复：
1. Table 2 六个 per-level 90% CI 从公开个体数据程序化重算（v7 值错误）
2. C2-C5 Mann-Whitney p 修正 (0.359, complete-case)；available-case CI 修正
3. symmetric MNAR δ 重新参数化：δ = 失败层实际厂商差 (UIH-GE)
   （v7 参数化中 δ 是增量偏移，失败层实际差 = 观察差 + δ，语义与图轴标签不符）
4. P46 年龄数字修正：DCM 排除 (213→146) 与 3T 排除 (146→132) 两个步骤分别报告
5. software 版本动态读取（实际 statsmodels 0.14.6，非 0.15.0）
6. Table 1/Table 2/摘要全部数值程序化导出
7. vendor×age 交互检验（回应 P51）
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import statsmodels.formula.api as smf
from importlib.metadata import version as _ver

# Data sources:
#   PUBLIC (this repository): data/kingston_15T_per_subject_deidentified.csv (n=132)
#     reproduces the main endpoint, Table 1/2, MixedLM, tipping-point, matching,
#     tertiles, and the n=132 whole-cord analyses.
#   INTERNAL (not public; QC-failed and 146/213 cohort files): required only for the
#     cohort-flow ages (DCM-review step and 3T step) and the 146/213 sensitivity
#     cohorts. When internal files are unavailable these sections are skipped and
#     their summary values remain available in results/sensitivity_cohorts.csv.
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REPO_DATA = os.path.join(REPO_ROOT, 'data')
BASE = os.environ.get('KINGSTON_INTERNAL_DIR',
                      'E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2/')
HAVE_INTERNAL = os.path.exists(os.path.join(BASE, 'final_cohort_15T.csv'))
R = {}

R['software'] = {
    'statsmodels': _ver('statsmodels'),
    'scipy': _ver('scipy'),
    'pandas': _ver('pandas'),
    'numpy': _ver('numpy'),
    'python_note': 'MixedLM: ML (REML=False); multi-optimizer (lbfgs/bfgs/powell/cg)',
}

# ============ 0. 数据载入 ============
pub = pd.read_csv(os.path.join(REPO_DATA, 'kingston_15T_per_subject_deidentified.csv'))  # 132 (public)
if HAVE_INTERNAL:
    pl = pd.read_csv(BASE + 'perlevel_csa_moderate.csv')
    fa = pd.read_csv(BASE + 'final_cohort_no_age_limit.csv')  # 213
    fq = pd.read_csv(BASE + 'final_cohort_moderate.csv')      # 146
    params = pd.read_csv(BASE + 'supplementary_scan_parameters.csv')
else:
    pl = fa = fq = params = None

# Build the working 132-subject frame from the PUBLIC data (internal file optional cross-check)
wc = pub.rename(columns={'study_id': 'patient_id', 'whole_cord_mean_csa_mm2': 'mean_csa'})
if HAVE_INTERNAL:
    _wc_int = pd.read_csv(BASE + 'final_cohort_15T.csv')
    assert len(pub) == 132 and len(wc) == 132
    assert np.allclose(np.sort(_wc_int.mean_csa.values), np.sort(wc.mean_csa.values), atol=1e-6)

uw = wc[wc.vendor == 'UIH'].mean_csa.values
gw = wc[wc.vendor == 'GE'].mean_csa.values

# ============ 1. 队列流转与年龄 (P46 修正) ============
if not HAVE_INTERNAL:
    R['cohort_flow'] = {'note': 'internal cohort files unavailable; see '
                               'results/sensitivity_cohorts.csv for machine-readable '
                               'summary values of the 146/213/48 sensitivity cohorts.'}
    fq = fa = None

R['cohort_flow'] = {
    'pacs_exports': {'export1': 80, 'export2': 232, 'total': 312},
    'preprocessing_exclusions': {'total': 8, 'philips': 1, 'no_sagittal_T2': 3, 'under_18': 4},
    'entering_pipeline': 304,            # UIH 90 / GE 214 (Philips excluded here)
    'processing_failures': 4,
    'processed': 300,                    # UIH 90 / GE 210
    'qc_pass': {'total': 213, 'UIH': 82, 'GE': 131},
    'qc_fail': {'total': 87, 'UIH': 8, 'GE': 79},
}
if HAVE_INTERNAL:
    qids = set(fq.patient_id)
    dcm_excl = fa[~fa.patient_id.isin(qids)]                    # 213 -> 146
    fcids = set(wc.patient_id)
    t3_excl = fq[~fq.patient_id.isin(fcids)]                    # 146 -> 132 (GE 3.0T)
    R['cohort_flow']['dcm_review_exclusions'] = {   # 213 -> 146
        'total': 67, 'UIH': int(len(dcm_excl[dcm_excl.vendor == 'UIH'])),
        'GE': int(len(dcm_excl[dcm_excl.vendor == 'GE'])),
        'moderate_criteria_DCM': 55, 'non_cervical_series': 2, 'overlap_earlier_review': 10,
    }
if HAVE_INTERNAL:
    R['cohort_flow']['dcm_review_age'] = {
        'excluded_mean': float(dcm_excl.age.mean()), 'excluded_sd': float(dcm_excl.age.std()),
        'retained146_mean': float(fq.age.mean()), 'retained146_sd': float(fq.age.std()),
        'welch_p': float(stats.ttest_ind(dcm_excl.age, fq.age, equal_var=False).pvalue),
    }
    R['cohort_flow']['field_strength_exclusions_146_to_132'] = {
        'total': 14, 'all_GE_3T': bool((t3_excl.vendor == 'GE').all()),
        'age_mean': float(t3_excl.age.mean()), 'age_sd': float(t3_excl.age.std()),
    }
R['cohort_flow'].update({
    'final': {'total': 132, 'UIH': 51, 'GE': 81}
})


# ============ 2. Table 1 (cohort characteristics) ============
u52, g52 = wc[wc.vendor == 'UIH'], wc[wc.vendor == 'GE']
R['table1'] = {
    'age': {'UIH_mean': float(u52.age.mean()), 'UIH_sd': float(u52.age.std()),
            'GE_mean': float(g52.age.mean()), 'GE_sd': float(g52.age.std()),
            'welch_p': float(stats.ttest_ind(u52.age, g52.age, equal_var=False).pvalue)},
    'sex': {'UIH_women': int((u52.sex == 'F').sum()), 'UIH_n': len(u52),
            'GE_women': int((g52.sex == 'F').sum()), 'GE_n': len(g52),
            'fisher_p': float(stats.fisher_exact(
                [[(u52.sex == 'F').sum(), (u52.sex == 'M').sum()],
                 [(g52.sex == 'F').sum(), (g52.sex == 'M').sum()]])[1])},
    'whole_cord': {'UIH_mean': float(uw.mean()), 'UIH_sd': float(uw.std(ddof=1)),
                   'GE_mean': float(gw.mean()), 'GE_sd': float(gw.std(ddof=1))},
}

# ============ 3. 主结局 C2-C5 (complete-case / available-case) ============
def welch_full(a, b, alpha=0.10):
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    d = a.mean() - b.mean()
    tc = stats.t.ppf(1 - alpha / 2, dfree)
    t_, p_ = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return {'n_a': na, 'n_b': nb, 'diff': float(d), 'se': float(se), 'df': float(dfree),
            'ci_lo': float(d - tc * se), 'ci_hi': float(d + tc * se),
            't': float(t_), 'p': float(p_), 'cohens_d': float(d / sp)}

def tost_p(d, se, df, margin=5.0):
    return float(max(stats.t.cdf((d - margin) / se, df),
                     1 - stats.t.cdf((d + margin) / se, df)))

cc = pub.dropna(subset=['C2', 'C3', 'C4', 'C5'])
a_cc = cc[cc.vendor == 'UIH']['c2c5_mean_csa_mm2']
b_cc = cc[cc.vendor == 'GE']['c2c5_mean_csa_mm2']
w_cc = welch_full(a_cc, b_cc)
w_cc['mwu_p'] = float(stats.mannwhitneyu(a_cc, b_cc, alternative='two-sided').pvalue)
w_cc['tost_p'] = tost_p(w_cc['diff'], w_cc['se'], w_cc['df'])
w_cc['UIH_mean'] = float(a_cc.mean()); w_cc['UIH_sd'] = float(a_cc.std(ddof=1))
w_cc['GE_mean'] = float(b_cc.mean()); w_cc['GE_sd'] = float(b_cc.std(ddof=1))

a_av = pub[pub.vendor == 'UIH']['c2c5_mean_csa_mm2'].dropna()
b_av = pub[pub.vendor == 'GE']['c2c5_mean_csa_mm2'].dropna()
w_av = welch_full(a_av, b_av)
w_av['mwu_p'] = float(stats.mannwhitneyu(a_av, b_av, alternative='two-sided').pvalue)
w_av['tost_p'] = tost_p(w_av['diff'], w_av['se'], w_av['df'])
R['main_endpoint'] = {'complete_case': w_cc, 'available_case': w_av,
                      'note': 'v7 MWU p=0.28 and available-case CI -0.59..3.85 were incorrect'}

# whole-cord
w_wc = welch_full(uw, gw)
w_wc['tost_p'] = tost_p(w_wc['diff'], w_wc['se'], w_wc['df'])
w_wc95 = welch_full(uw, gw, alpha=0.05)
w_wc['ci95_lo'] = w_wc95['ci_lo']; w_wc['ci95_hi'] = w_wc95['ci_hi']
w_wc['mwu_p'] = float(stats.mannwhitneyu(uw, gw, alternative='two-sided').pvalue)
R['whole_cord'] = w_wc

# ============ 4. Table 2 per-level (程序化: Welch CI + MWU + ANCOVA + Holm) ============
rows = []
for lv in ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
    a = pub[pub.vendor == 'UIH'][lv].dropna()
    b = pub[pub.vendor == 'GE'][lv].dropna()
    w = welch_full(a, b)
    w['mwu_p'] = float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
    w['UIH_mean'] = float(a.mean()); w['UIH_sd'] = float(a.std(ddof=1))
    w['GE_mean'] = float(b.mean()); w['GE_sd'] = float(b.std(ddof=1))
    w['level'] = lv
    # age-adjusted ANCOVA
    sub = pub.dropna(subset=[lv, 'age'])
    import statsmodels.api as sm
    X = pd.get_dummies(sub[['vendor']], drop_first=True).astype(float)
    X['age'] = sub['age'].values; X['const'] = 1.0
    y = sub[lv].values
    fit = sm.OLS(y, X).fit()
    w['ancova_vendor_p'] = float(fit.pvalues['vendor_UIH'])
    w['ancova_age_p'] = float(fit.pvalues['age'])
    rows.append(w)

# Holm within families
pw = sorted([r['p'] for r in rows]); pa = sorted([r['ancova_vendor_p'] for r in rows])
def holm(ps):
    m = len(ps); adj = []; run = 0.0
    for i, p in enumerate(ps):
        v = min(1.0, (m - i) * p); run = max(run, v); adj.append(run)
    return adj
hw = holm(pw); ha = holm(pa)
for r in rows:
    r['holm_welch'] = hw[pw.index(r['p'])] if r['p'] in pw else None
for r in rows:
    r['holm_welch'] = None
for i, p in enumerate(pw):
    for r in rows:
        if r['p'] == p and r['holm_welch'] is None:
            r['holm_welch'] = hw[i]; break
for r in rows:
    r['holm_ancova'] = None
for i, p in enumerate(pa):
    for r in rows:
        if r['ancova_vendor_p'] == p and r['holm_ancova'] is None:
            r['holm_ancova'] = ha[i]; break
R['table2_perlevel'] = rows

# ============ 5. vendor x age 交互 (P51) ============
X = pd.get_dummies(pub[['vendor']], drop_first=True).astype(float)
X['age'] = pub['age'].values; X['vendor_UIH_x_age'] = X['vendor_UIH'] * X['age']; X['const'] = 1.0
fit_int = sm.OLS(pub['whole_cord_mean_csa_mm2'].values, X).fit()
R['vendor_x_age_interaction'] = {
    'endpoint': 'whole-cord CSA', 'interaction_p': float(fit_int.pvalues['vendor_UIH_x_age']),
    'interaction_coef': float(fit_int.params['vendor_UIH_x_age']),
    'note': 'tests effect modification; stratified estimates alone cannot establish it',
}
X2 = X.drop(columns=['vendor_UIH_x_age'])
fit_c25 = sm.OLS(pub['c2c5_mean_csa_mm2'].dropna().values,
                X2.loc[pub['c2c5_mean_csa_mm2'].notna()]).fit()
R['ancova_main'] = {
    'vendor_coef': float(fit_c25.params['vendor_UIH']),
    'vendor_p': float(fit_c25.pvalues['vendor_UIH']),
    'age_p': float(fit_c25.pvalues['age']),
}
# whole-cord ANCOVA
fit_wc = sm.OLS(pub['whole_cord_mean_csa_mm2'].values, X2).fit()
R['ancova_wholecord'] = {
    'vendor_coef': float(fit_wc.params['vendor_UIH']), 'vendor_p': float(fit_wc.pvalues['vendor_UIH']),
    'age_p': float(fit_wc.pvalues['age']), 'age_slope': float(fit_wc.params['age']),
}

# ============ 6. MixedLM (v7 不变, 重跑以程序化导出) ============
per = pub.melt(id_vars=['study_id', 'vendor', 'age'],
               value_vars=['C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
               var_name='level', value_name='csa').dropna(subset=['csa'])
per['vendor'] = per['vendor'].astype(str)

def fit_mixed(formula):
    best = None
    for meth in ['lbfgs', 'bfgs', 'powell', 'cg']:
        try:
            m = smf.mixedlm(formula, per, groups=per['study_id']).fit(
                reml=False, method=meth, maxiter=2000, disp=False)
            if m.converged and (best is None or m.llf > best.llf):
                best = m; best_meth = meth
        except Exception:
            continue
    return best, best_meth

mred, meth_red = fit_mixed('csa ~ C(vendor) + C(level) + age')
mfull, meth_full = fit_mixed('csa ~ C(vendor) * C(level) + age')

def export_model(m):
    coefs = []
    for t in m.params.index:
        coefs.append({'term': t, 'coef': float(m.params[t]), 'se': float(m.bse[t]),
                      'z': float(m.tvalues[t]), 'p': float(m.pvalues[t]),
                      'ci95_lo': float(m.conf_int().loc[t, 0]), 'ci95_hi': float(m.conf_int().loc[t, 1])})
    return {'llf': float(m.llf), 'aic': float(m.aic), 'bic': float(m.bic),
            'n_obs': int(m.nobs), 'converged': bool(m.converged),
            'k_fixed': int(len(m.fe_params)), 'k_var': int(len(m.vcomp) + 1),
            'coefficients': coefs,
            'random_effect_var': float(m.cov_re.iloc[0, 0]),
            'residual_var': float(m.scale)}

red_j = export_model(mred); full_j = export_model(mfull)
red_j['optimizer'] = meth_red; full_j['optimizer'] = meth_full

lr = 2 * (mfull.llf - mred.llf)
dfdiff = int(len(mfull.fe_params) - len(mred.fe_params))
p_lr = float(stats.chi2.sf(lr, dfdiff))

# joint Wald on 5 interaction terms
vnames = [t for t in mfull.params.index if ':' in t]
b = mfull.params[vnames].values
V = mfull.cov_params().loc[vnames, vnames].values
wald = float(b @ np.linalg.pinv(V) @ b)
p_wald = float(stats.chi2.sf(wald, len(vnames)))

vc = [t for t in mred.params.index if 'vendor' in t][0]
R['mixed_model_v8'] = {
    'reduced': red_j, 'full': full_j,
    'LR': {'chi2': lr, 'df': dfdiff, 'p': p_lr},
    'joint_Wald': {'chi2': wald, 'df': len(vnames), 'p': p_wald, 'terms': vnames},
    'vendor_effect_reduced': {'beta': float(mred.params[vc]), 'p': float(mred.pvalues[vc]),
                              'ci95': [float(mred.conf_int().loc[vc, 0]), float(mred.conf_int().loc[vc, 1])]},
}

# ============ 7. tipping-point (修正 symmetric 参数化) ============
# 统一参数化: delta = 失败层实际厂商差 (UIH - GE)
# GE-anchored: mGf = mG; mUf = mG + delta          (fail差 = delta)
# symmetric  : mUf = mU + (delta-d_obs)/2; mGf = mG - (delta-d_obs)/2
#              (fail差 = delta; 两厂商相对各自base对称移动)
mU, sU, nU = float(uw.mean()), float(uw.std(ddof=1)), len(uw)
mG, sG, nG = float(gw.mean()), float(gw.std(ddof=1)), len(gw)
nUf, nGf = 8, 79
d_obs = mU - mG

def merge_stratum(m_base, s_base, n_base, m_fail, s_fail, n_fail):
    n = n_base + n_fail
    m_pool = (n_base * m_base + n_fail * m_fail) / n
    ss_within = (n_base - 1) * s_base ** 2 + (n_fail - 1) * s_fail ** 2
    ss_between = n_base * (m_base - m_pool) ** 2 + n_fail * (m_fail - m_pool) ** 2
    var_pool = (ss_within + ss_between) / (n - 1)
    return m_pool, var_pool, n

def tipping(delta, sigma_factor, anchor):
    if anchor == 'GE':
        mGf, mUf = mG, mG + delta
    else:  # symmetric, delta = actual fail-stratum vendor difference
        mUf = mU + (delta - d_obs) / 2.0
        mGf = mG - (delta - d_obs) / 2.0
    mUp, varU, nUt = merge_stratum(mU, sU, nU, mUf, sU * sigma_factor, nUf)
    mGp, varG, nGt = merge_stratum(mG, sG, nG, mGf, sG * sigma_factor, nGf)
    dp = mUp - mGp
    se = np.sqrt(varU / nUt + varG / nGt)
    dfw = (varU / nUt + varG / nGt) ** 2 / ((varU / nUt) ** 2 / (nUt - 1) + (varG / nGt) ** 2 / (nGt - 1))
    p_t = tost_p(dp, se, dfw)
    return dp, p_t

def solve_upper(anchor, sc):
    f = lambda d: tipping(d, sc, anchor)[1] - 0.05
    try:
        return float(brentq(f, d_obs, 40))
    except Exception:
        return None

def solve_lower(anchor, sc):
    f = lambda d: tipping(d, sc, anchor)[1] - 0.05
    try:
        return float(brentq(f, -40, d_obs))
    except Exception:
        return None

tip = {}
for sc in [1.0, 1.25, 1.5]:
    for anchor in ['GE', 'symmetric']:
        tip[f'{anchor}_sigma{sc}'] = {
            'upper_delta': solve_upper(anchor, sc), 'lower_delta': solve_lower(anchor, sc)}
R['tipping_point_v8'] = {
    'parameterisation': 'delta = assumed actual vendor difference (UIH-GE) in the QC-failed stratum',
    'd_observed_base': d_obs,
    'grid_1d': tip,
    'note': 'symmetric scenario: both vendor means shift by +/-(delta-d_obs)/2 from base means, '
            'so the failed-stratum vendor difference equals delta. When delta = d_obs the shifts '
            'are zero (missing-at-random-like reference).',
}

# 2D grids for Fig. 8
grid2d = {}
deltas = np.arange(-10, 30.5, 0.5)
for anchor in ['GE', 'symmetric']:
    for sc in [1.0, 1.25, 1.5]:
        ps = [tipping(d, sc, anchor)[1] for d in deltas]
        grid2d[f'{anchor}_sigma{sc}'] = {'deltas': deltas.tolist(), 'tost_p': ps}
R['tipping_2d_v8'] = grid2d

# ============ 8. C2-C5 margin sensitivity (重算自公开数据) ============
a2 = a_cc.values; b2 = b_cc.values
w25 = welch_full(pd.Series(a2), pd.Series(b2))
margins = []
for m in [3.0, 4.0, 5.0, 6.0, 7.0]:
    margins.append({'margin_mm2': m, 'tost_p': tost_p(w25['diff'], w25['se'], w25['df'], m),
                    'equivalent': bool(tost_p(w25['diff'], w25['se'], w25['df'], m) < 0.05)})
R['margin_sensitivity_c2c5_v8'] = {
    'n': int(w25['n_a'] + w25['n_b']), 'UIH_n': int(w25['n_a']), 'GE_n': int(w25['n_b']),
    'diff': w25['diff'], 'se': w25['se'], 'df': w25['df'], 'margins': margins}

# whole-cord margins (unchanged analysis, recomputed)
mw = []
for m in [3.0, 4.0, 5.0, 6.0, 7.0]:
    mw.append({'margin_mm2': m, 'tost_p': tost_p(w_wc['diff'], w_wc['se'], w_wc['df'], m),
               'equivalent': bool(tost_p(w_wc['diff'], w_wc['se'], w_wc['df'], m) < 0.05)})
R['margin_sensitivity_wholecord_v8'] = {'diff': w_wc['diff'], 'margins': mw}

# ============ 9. 年龄匹配 + 分层 (v7 逻辑不变, 重跑) ============
rng = np.random.default_rng(42)
u_idx = wc[wc.vendor == 'UIH'].index.tolist()
g_idx = wc[wc.vendor == 'GE'].index.tolist()
pairs = []
g_avail = list(g_idx)
for ui in u_idx:  # greedy NN in UIH row order
    best_j, best_d = None, 1e9
    for gj in g_avail:
        dd = abs(wc.loc[ui, 'age'] - wc.loc[gj, 'age'])
        if dd < best_d: best_d, best_j = dd, gj
    if best_j is not None and best_d <= 3:
        pairs.append((ui, best_j)); g_avail.remove(best_j)
du = np.array([wc.loc[u, 'mean_csa'] - wc.loc[g, 'mean_csa'] for u, g in pairs])
t_p = stats.ttest_1samp(du, 0)
R['age_matching_v8'] = {
    'n_pairs': len(pairs), 'diff_mean': float(du.mean()), 'diff_sd': float(du.std(ddof=1)),
    't': float(t_p.statistic), 'p': float(t_p.pvalue),
    'ci95': [float(du.mean() - stats.t.ppf(0.975, len(du) - 1) * du.std(ddof=1) / np.sqrt(len(du))),
             float(du.mean() + stats.t.ppf(0.975, len(du) - 1) * du.std(ddof=1) / np.sqrt(len(du)))],
    'dz': float(du.mean() / du.std(ddof=1)),
    'note': 'greedy nearest-neighbour in UIH row order; caliper 3y; result is order-dependent '
            '(a limitation acknowledged in the manuscript)',
}

# tertiles
tert = []
for lo, hi in [(20, 39), (40, 59), (60, 86)]:
    sub = wc[(wc.age >= lo) & (wc.age <= hi)]
    u9 = sub[sub.vendor == 'UIH'].mean_csa.values
    g9 = sub[sub.vendor == 'GE'].mean_csa.values
    w9 = welch_full(pd.Series(u9), pd.Series(g9))
    w9['UIH_mean'] = float(u9.mean()); w9['UIH_sd'] = float(u9.std(ddof=1))
    w9['GE_mean'] = float(g9.mean()); w9['GE_sd'] = float(g9.std(ddof=1))
    w9['welch_p'] = w9['p']; w9['ci90'] = [w9['ci_lo'], w9['ci_hi']]
    w9['n_UIH'] = w9['n_a']; w9['n_GE'] = w9['n_b']
    w9['label'] = f'{lo}-{hi} y'; tert.append(w9)
R['age_stratified_tertiles_v8'] = {'strata': tert}

med = wc.age.median()
you = wc[wc.age <= med]; old = wc[wc.age > med]
wy = welch_full(you[you.vendor == 'UIH'].mean_csa, you[you.vendor == 'GE'].mean_csa)
wo = welch_full(old[old.vendor == 'UIH'].mean_csa, old[old.vendor == 'GE'].mean_csa)
R['age_stratified_median_v8'] = {'median': float(med), 'younger': wy, 'older': wo}

# ============ 保存 ============
out = os.path.join(REPO_ROOT, 'results', 'reanalysis_v8_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(f'Saved -> {out}')

print('\n===== v8 关键结果摘要 =====')
print(f"software: statsmodels {R['software']['statsmodels']}, scipy {R['software']['scipy']}")
print(f"main C2-C5 (cc n={w_cc['n_a']}+{w_cc['n_b']}): diff={w_cc['diff']:.4f} "
      f"90%CI {w_cc['ci_lo']:.4f}..{w_cc['ci_hi']:.4f} p={w_cc['p']:.4f} MWU p={w_cc['mwu_p']:.4f} "
      f"TOST p={w_cc['tost_p']:.4f}")
print(f"main C2-C5 (available n={w_av['n_a']}+{w_av['n_b']}): diff={w_av['diff']:.4f} "
      f"90%CI {w_av['ci_lo']:.4f}..{w_av['ci_hi']:.4f} p={w_av['p']:.4f}")
print('Table2 per-level 90% CI:')
for r in rows:
    print(f"  {r['UIH_mean']:.1f}: diff={r['diff']:+.2f} CI {r['ci_lo']:.2f}..{r['ci_hi']:.2f} "
          f"p={r['p']:.3f} holm={r['holm_welch']:.3f} ancova={r['ancova_vendor_p']:.3f} "
          f"holmA={r['holm_ancova']:.3f}")
print(f"vendor x age interaction p={R['vendor_x_age_interaction']['interaction_p']:.4f}")
if HAVE_INTERNAL:
    print(f"DCM excl age {R['cohort_flow']['dcm_review_age']['excluded_mean']:.1f}"
          f"±{R['cohort_flow']['dcm_review_age']['excluded_sd']:.1f} vs "
          f"retained146 {R['cohort_flow']['dcm_review_age']['retained146_mean']:.1f}"
          f"±{R['cohort_flow']['dcm_review_age']['retained146_sd']:.1f} "
          f"p={R['cohort_flow']['dcm_review_age']['welch_p']:.4f}")
    print(f"3T excl (146->132): n=14 all GE={R['cohort_flow']['field_strength_exclusions_146_to_132']['all_GE_3T']} "
          f"age {R['cohort_flow']['field_strength_exclusions_146_to_132']['age_mean']:.1f}")
print('tipping v8 (delta = actual fail-stratum diff):')
for k, v in tip.items():
    print(f"  {k}: upper={v['upper_delta']:.2f} lower={v['lower_delta']:.2f}")
print(f"whole-cord: diff={w_wc['diff']:.4f} 90CI {w_wc['ci_lo']:.4f}..{w_wc['ci_hi']:.4f} "
      f"95CI {w_wc['ci95_lo']:.4f}..{w_wc['ci95_hi']:.4f} p={w_wc['p']:.5f} TOST={w_wc['tost_p']:.5f}")
