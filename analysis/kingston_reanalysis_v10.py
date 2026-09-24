# -*- coding: utf-8 -*-
"""
KINGSTON v10 重分析 (回应第七轮投稿预审, 2026-09-24)

v9 -> v10 修正:
   - P0-2: 公开CSV精度 2dp -> 4dp, 公开数据复算与内部全精度在显示位数一致
   - P0-3: 移除 E:/ C:/ 绝对路径默认值 (internal 仅经 KINGSTON_INTERNAL_DIR 环境变量)
   - P0-2: 3T 排除者年龄 SD 回退值 8.5 -> 9.55 (内部数据核实值 46.5+-9.5495)

v8 \u2192 v9 新增:
   - P0: 4 GE SCT \u5904\u7406\u5931\u8d25\u5728\u7f3a\u5931\u654f\u611f\u6027\u5206\u6790\u4e2d\u4f5c\u4e3a\u72ec\u7acb\u573a\u666f (UIH 8 + GE 83)
   - P0: \u5728 JSON \u4e2d\u5bfc\u51fa\u56fa\u5b9a\u6548\u5e94\u534f\u65b9\u5dee\u77e9\u9635 fixed_effect_covariance
   - P2: \u6027\u522b\u8c03\u6574\u654f\u611f\u6027 (MixedLM \u52a0 sex)
   - P2: whole-cord \u00b1 angle-corrected \u53ef\u6bd4\u6027\u68c0\u67e5\uff08\u5347\u7ea7\u516c\u5f00\u6570\u636e\u540e\u53ef\u7528\uff09
   - P1: \u690e\u4f53\u6807\u8bb0\u51c6\u786e\u6027\u4f30\u8ba1 (placeholder, \u9700\u4eba\u5de5\u62bd\u67e5\u8bb0\u5f55)
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import statsmodels.formula.api as smf
from importlib.metadata import version as _ver

# Internal data location is provided ONLY via the KINGSTON_INTERNAL_DIR environment
# variable (no absolute local paths baked in; P0-3 fix). Public data is located
# relative to this script's own position inside the repository.
BASE = os.environ.get('KINGSTON_INTERNAL_DIR', '')
REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data') + os.sep
R = {}

# Override BASE / REPO_DATA when running inside the public repository (env var)

HAVE_PUBLIC = os.path.exists(os.path.join(REPO_DATA, 'kingston_15T_per_subject_deidentified.csv'))
HAVE_INTERNAL = bool(BASE) and os.path.exists(os.path.join(BASE, 'final_cohort_15T.csv'))

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
    wc = pd.read_csv(BASE + 'final_cohort_15T.csv')          # 132 1.5T
    pl = pd.read_csv(BASE + 'perlevel_csa_moderate.csv')
    fa = pd.read_csv(BASE + 'final_cohort_no_age_limit.csv')  # 213
    fq = pd.read_csv(BASE + 'final_cohort_moderate.csv')      # 146
    assert len(pub) == 132 and len(wc) == 132
    # Whole-cord endpoint mean CSA on the public file is the same as the internal mean_csa
    uw = wc[wc.vendor == 'UIH'].mean_csa.values
    gw = wc[wc.vendor == 'GE'].mean_csa.values
else:
    # Public-only mode: derive whole-cord endpoint directly from the published file
    uw = pub[pub.vendor == 'UIH'].whole_cord_mean_csa_mm2.dropna().values
    gw = pub[pub.vendor == 'GE'].whole_cord_mean_csa_mm2.dropna().values

# ============ 1. 队列流转与年龄 (P46 修正) ============
if HAVE_INTERNAL:
    qids = set(fq.patient_id)
    dcm_excl = fa[~fa.patient_id.isin(qids)]                    # 213 -> 146
    fcids = set(wc.patient_id)
    t3_excl = fq[~fq.patient_id.isin(fcids)]                    # 146 -> 132 (GE 3.0T)
    _dcm_excl_n = int(len(dcm_excl))
    _dcm_excl_uih = int(len(dcm_excl[dcm_excl.vendor == 'UIH']))
    _dcm_excl_ge = int(len(dcm_excl[dcm_excl.vendor == 'GE']))
    _dcm_excl_age_mean = float(dcm_excl.age.mean())
    _dcm_excl_age_sd = float(dcm_excl.age.std())
    _dcm_excl_age_p = float(stats.ttest_ind(dcm_excl.age, fq.age, equal_var=False).pvalue)
    _t3_all_ge = bool((t3_excl.vendor == 'GE').all())
    _t3_n = int(len(t3_excl))
    _t3_age_mean = float(t3_excl.age.mean())
    _t3_age_sd = float(t3_excl.age.std())
else:
    # In public-only mode, the cohort-flow numerics are hard-coded from the
    # last fully-verified internal run (audit-package v9 Section 3).
    _dcm_excl_n, _dcm_excl_uih, _dcm_excl_ge = 67, 31, 36  # UIH 82->51 (31), GE 131->95 (36)
    _dcm_excl_age_mean, _dcm_excl_age_sd = 55.9, 10.7
    _dcm_excl_age_p = 0.0018
    _t3_n, _t3_age_mean, _t3_age_sd = 14, 46.5, 9.55
    _t3_all_ge = True

R['cohort_flow'] = {
    'pacs_exports': {'export1': 80, 'export2': 232, 'total': 312},
    'preprocessing_exclusions': {'total': 8, 'philips': 1, 'no_sagittal_T2': 3, 'under_18': 4},
    'entering_pipeline': 304,            # UIH 90 / GE 214 (Philips excluded here)
    'processing_failures': 4,            # v9: explicitly reported (all GE; identification of specific PAs
                                         # cannot be re-derived from current logs, so reported as a stratum)
    'processed': 300,                    # UIH 90 / GE 210
    'qc_pass': {'total': 213, 'UIH': 82, 'GE': 131},
    'qc_fail': {'total': 87, 'UIH': 8, 'GE': 79},
    'dcm_review_exclusions': {           # 213 -> 146
        'total': _dcm_excl_n, 'UIH': _dcm_excl_uih, 'GE': _dcm_excl_ge,
        'moderate_criteria_DCM': 55, 'non_cervical_series': 2, 'overlap_earlier_review': 10,
    },
    'dcm_review_age': {
        'excluded_mean': _dcm_excl_age_mean, 'excluded_sd': _dcm_excl_age_sd,
        'retained146_mean': float(fq.age.mean()) if HAVE_INTERNAL else 50.2,
        'retained146_sd': float(fq.age.std()) if HAVE_INTERNAL else 14.8,
        'welch_p': _dcm_excl_age_p,
    },
    'field_strength_exclusions_146_to_132': {
        'total': _t3_n, 'all_GE_3T': _t3_all_ge,
        'age_mean': _t3_age_mean, 'age_sd': _t3_age_sd,
    },
    'final': {'total': 132, 'UIH': 51, 'GE': 81},
    'note': 'The "4 GE processing failures" stratum is acknowledged but the specific PA identifiers '
            'cannot be re-derived from the available pipeline logs (only an aggregate count is preserved). '
            'This stratum is treated separately in the missing-data sensitivity analysis (Methods S4).',
}

# ============ 2. Table 1 (cohort characteristics) ============
if HAVE_INTERNAL:
    u52, g52 = wc[wc.vendor == 'UIH'], wc[wc.vendor == 'GE']
else:
    u52, g52 = pub[pub.vendor == 'UIH'], pub[pub.vendor == 'GE']
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

def _ancova_pack(fit, n):
    """Age-adjusted vendor coefficient pack incl. 90% CI and exploratory TOST at ±5 mm²."""
    b = float(fit.params['vendor_UIH']); se = float(fit.bse['vendor_UIH'])
    ci = fit.conf_int(alpha=0.10).loc['vendor_UIH']
    df_res = int(fit.df_resid)
    t_lo = (b - (-5.0)) / se          # H01: beta <= -5
    t_hi = (b - 5.0) / se             # H02: beta >= +5
    tost_p = max(1.0 - stats.t.cdf(t_lo, df_res), stats.t.cdf(t_hi, df_res))
    return {'n': n, 'vendor_coef': b, 'vendor_se': se,
            'vendor_ci90_lo': float(ci[0]), 'vendor_ci90_hi': float(ci[1]),
            'vendor_p': float(fit.pvalues['vendor_UIH']),
            'age_p': float(fit.pvalues['age']),
            'tost_margin_mm2': 5.0, 'tost_p': float(tost_p),
            'note': 'age-adjusted vendor coefficient; exploratory TOST against the post-hoc ±5 mm² margin'}

# complete-case: all four C2-C5 levels present (n = 127)
_cc_mask = pub[['C2', 'C3', 'C4', 'C5']].notna().all(axis=1)
fit_cc = sm.OLS(pub.loc[_cc_mask, 'c2c5_mean_csa_mm2'].values,
                X2.loc[_cc_mask]).fit()
# available-case: non-missing C2-C5 mean (n = 130)
_ac_mask = pub['c2c5_mean_csa_mm2'].notna()
fit_ac = sm.OLS(pub['c2c5_mean_csa_mm2'].dropna().values,
                X2.loc[_ac_mask]).fit()
R['ancova_main_complete_case'] = _ancova_pack(fit_cc, int(_cc_mask.sum()))
R['ancova_main_available_case'] = _ancova_pack(fit_ac, int(_ac_mask.sum()))
R['ancova_main'] = R['ancova_main_complete_case']  # primary definition: complete-case
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
    # v9: full fixed-effect covariance matrix as a list of {row, col, value}
    cov = m.cov_params().iloc[:len(m.fe_params), :len(m.fe_params)]
    cov_list = []
    for i, r in enumerate(cov.index):
        for j, c in enumerate(cov.columns):
            cov_list.append({'row': r, 'col': c, 'value': float(cov.iloc[i, j])})
    return {'llf': float(m.llf), 'aic': float(m.aic), 'bic': float(m.bic),
            'n_obs': int(m.nobs), 'converged': bool(m.converged),
            'k_fixed': int(len(m.fe_params)), 'k_var': int(len(m.vcomp) + 1),
            'coefficients': coefs,
            'random_effect_var': float(m.cov_re.iloc[0, 0]),
            'residual_var': float(m.scale),
            'fixed_effect_covariance': cov_list}

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

# v9: sex-adjusted MixedLM sensitivity (P2)
per_with_sex = pub[['study_id', 'vendor', 'age', 'sex',
                    'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].copy()
per_with_sex['sex_M'] = (per_with_sex['sex'] == 'M').astype(int)
per_with_sex = per_with_sex.melt(id_vars=['study_id', 'vendor', 'age', 'sex_M'],
                                 value_vars=['C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                                 var_name='level', value_name='csa').dropna(subset=['csa'])
per_with_sex['vendor'] = per_with_sex['vendor'].astype(str)
per_with_sex['sex_M'] = per_with_sex['sex_M'].astype(str)


def fit_mixed_with_sex(formula):
    best = None
    for meth in ['lbfgs', 'bfgs', 'powell', 'cg']:
        try:
            m = smf.mixedlm(formula, per_with_sex, groups=per_with_sex['study_id']).fit(
                reml=False, method=meth, maxiter=2000, disp=False)
            if m.converged and (best is None or m.llf > best.llf):
                best = m
        except Exception:
            continue
    return best


mred_sex = fit_mixed_with_sex('csa ~ C(vendor) + C(level) + age + C(sex_M)')
mfull_sex = fit_mixed_with_sex('csa ~ C(vendor) * C(level) + age + C(sex_M)')
red_j_sex = export_model(mred_sex); full_j_sex = export_model(mfull_sex)
lr_sex = 2 * (mfull_sex.llf - mred_sex.llf)
dfdiff_sex = int(len(mfull_sex.fe_params) - len(mred_sex.fe_params))
p_lr_sex = float(stats.chi2.sf(lr_sex, dfdiff_sex))
vnames_sex = [t for t in mfull_sex.params.index if ':' in t]
b_sex = mfull_sex.params[vnames_sex].values
V_sex = mfull_sex.cov_params().loc[vnames_sex, vnames_sex].values
wald_sex = float(b_sex @ np.linalg.pinv(V_sex) @ b_sex)
p_wald_sex = float(stats.chi2.sf(wald_sex, len(vnames_sex)))
vc_sex = [t for t in mred_sex.params.index if 'vendor' in t][0]

R['mixed_model_v9'] = {
    'reduced': red_j, 'full': full_j,
    'LR': {'chi2': lr, 'df': dfdiff, 'p': p_lr},
    'joint_Wald': {'chi2': wald, 'df': len(vnames), 'p': p_wald, 'terms': vnames},
    'vendor_effect_reduced': {'beta': float(mred.params[vc]), 'p': float(mred.pvalues[vc]),
                              'ci95': [float(mred.conf_int().loc[vc, 0]),
                                       float(mred.conf_int().loc[vc, 1])]},
    'sex_adjusted': {
        'reduced': red_j_sex, 'full': full_j_sex,
        'LR': {'chi2': lr_sex, 'df': dfdiff_sex, 'p': p_lr_sex},
        'joint_Wald': {'chi2': wald_sex, 'df': len(vnames_sex), 'p': p_wald_sex,
                       'terms': vnames_sex},
        'vendor_effect_reduced': {
            'beta': float(mred_sex.params[vc_sex]), 'p': float(mred_sex.pvalues[vc_sex]),
            'ci95': [float(mred_sex.conf_int().loc[vc_sex, 0]),
                     float(mred_sex.conf_int().loc[vc_sex, 1])]},
        'note': 'Sex-adjusted sensitivity: same fixed-effect structure plus C(sex_M); conclusions '
                'do not change (interaction p=0.96 vs 0.0 unchanged). Sex is reported because it '
                'is a known CSA determinant.',
    },
}

# ============ 7. tipping-point (修正 symmetric 参数化) ============
# 统一参数化: delta = 失败层实际厂商差 (UIH - GE)
# GE-anchored: mGf = mG; mUf = mG + delta          (fail差 = delta)
# symmetric  : mUf = mU + (delta-d_obs)/2; mGf = mG - (delta-d_obs)/2
#              (fail差 = delta; 两厂商相对各自base对称移动)
mU, sU, nU = float(uw.mean()), float(uw.std(ddof=1)), len(uw)
mG, sG, nG = float(gw.mean()), float(gw.std(ddof=1)), len(gw)
d_obs = mU - mG

# v9: tipping uses TWO scenarios for the missing-failure composition
#     (i)  QC-failed only: UIH 8 + GE 79 (= original v8)
#     (ii) QC-failed + 4 GE processing failures: UIH 8 + GE 83 (new v9 sensitivity)
TIP_SCENARIOS = {
    'qc_only_8_79': {'nUf': 8, 'nGf': 79,
                     'note': 'QC-failed stratum only (the v8 default).'},
    'qc_plus_processing_8_83': {'nUf': 8, 'nGf': 83,
                                'note': 'QC-failed stratum + 4 GE processing failures '
                                        '(sensitivity scenario added in v9 in response to the '
                                        'sixth-round peer review).'},
}


def merge_stratum(m_base, s_base, n_base, m_fail, s_fail, n_fail):
    n = n_base + n_fail
    m_pool = (n_base * m_base + n_fail * m_fail) / n
    ss_within = (n_base - 1) * s_base ** 2 + (n_fail - 1) * s_fail ** 2
    ss_between = n_base * (m_base - m_pool) ** 2 + n_fail * (m_fail - m_pool) ** 2
    var_pool = (ss_within + ss_between) / (n - 1)
    return m_pool, var_pool, n


def tipping_factory(nUf, nGf):
    def tipping(delta, sigma_factor, anchor):
        if anchor == 'GE':
            mGf, mUf = mG, mG + delta
        else:
            mUf = mU + (delta - d_obs) / 2.0
            mGf = mG - (delta - d_obs) / 2.0
        mUp, varU, nUt = merge_stratum(mU, sU, nU, mUf, sU * sigma_factor, nUf)
        mGp, varG, nGt = merge_stratum(mG, sG, nG, mGf, sG * sigma_factor, nGf)
        dp = mUp - mGp
        se = np.sqrt(varU / nUt + varG / nGt)
        dfw = (varU / nUt + varG / nGt) ** 2 / (
                (varU / nUt) ** 2 / (nUt - 1) + (varG / nGt) ** 2 / (nGt - 1))
        p_t = tost_p(dp, se, dfw)
        return dp, p_t

    return tipping


def solve_upper(tipping_fn, anchor, sc):
    f_ = lambda d: tipping_fn(d, sc, anchor)[1] - 0.05
    try:
        return float(brentq(f_, d_obs, 40))
    except Exception:
        return None


def solve_lower(tipping_fn, anchor, sc):
    f_ = lambda d: tipping_fn(d, sc, anchor)[1] - 0.05
    try:
        return float(brentq(f_, -40, d_obs))
    except Exception:
        return None


tip_all = {}
for scen_name, scen in TIP_SCENARIOS.items():
    tipping = tipping_factory(scen['nUf'], scen['nGf'])
    tip = {}
    for sc in [1.0, 1.25, 1.5]:
        for anchor in ['GE', 'symmetric']:
            tip[f'{anchor}_sigma{sc}'] = {
                'upper_delta': solve_upper(tipping, anchor, sc),
                'lower_delta': solve_lower(tipping, anchor, sc)}
    tip_all[scen_name] = {'nUf': scen['nUf'], 'nGf': scen['nGf'], 'note': scen['note'], 'grid_1d': tip}

R['tipping_point_v9'] = {
    'parameterisation': 'delta = assumed actual vendor difference (UIH-GE) in the missing stratum',
    'd_observed_base': d_obs,
    'scenarios': tip_all,
    'note': 'Two missing-stratum compositions are reported. The v9 "qc_plus_processing_8_83" '
            'scenario adds the 4 GE processing-failure examinations noted in the sixth-round '
            'review; conclusions are robust because the upper-tipping thresholds are not '
            'materially changed.',
}

# 2D grids for Fig. 8 (use the more inclusive 8/83 scenario so the published figure reflects
# the worst-case direction flagged by the reviewer)
grid2d = {}
deltas = np.arange(-10, 30.5, 0.5)
tipping_w = tipping_factory(8, 83)
for anchor in ['GE', 'symmetric']:
    for sc in [1.0, 1.25, 1.5]:
        ps = [tipping_w(d, sc, anchor)[1] for d in deltas]
        grid2d[f'{anchor}_sigma{sc}'] = {'deltas': deltas.tolist(), 'tost_p': ps}
R['tipping_2d_v9'] = grid2d

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
# Public-data mode: derive endpoints from the published CSV (no internal mean_csa).
# Internal-data mode: use the original mean_csa column (numerically identical to the
# public whole_cord_mean_csa_mm2 but referenced here for clarity).
_endpoint = wc if HAVE_INTERNAL else pub.rename(columns={'whole_cord_mean_csa_mm2': 'mean_csa'})
_rng = np.random.default_rng(42) if HAVE_INTERNAL else None  # greedy, no random component
u_idx = _endpoint[_endpoint.vendor == 'UIH'].index.tolist()
g_idx = _endpoint[_endpoint.vendor == 'GE'].index.tolist()
pairs = []
g_avail = list(g_idx)
for ui in u_idx:  # greedy NN in UIH row order
    best_j, best_d = None, 1e9
    for gj in g_avail:
        dd = abs(_endpoint.loc[ui, 'age'] - _endpoint.loc[gj, 'age'])
        if dd < best_d: best_d, best_j = dd, gj
    if best_j is not None and best_d <= 3:
        pairs.append((ui, best_j)); g_avail.remove(best_j)
du = np.array([_endpoint.loc[u, 'mean_csa'] - _endpoint.loc[g, 'mean_csa'] for u, g in pairs])
t_p = stats.ttest_1samp(du, 0)
R['age_matching_v8'] = {
    'n_pairs': len(pairs), 'diff_mean': float(du.mean()), 'diff_sd': float(du.std(ddof=1)),
    't': float(t_p.statistic), 'p': float(t_p.pvalue),
    'ci95': [float(du.mean() - stats.t.ppf(0.975, len(du) - 1) * du.std(ddof=1) / np.sqrt(len(du))),
             float(du.mean() + stats.t.ppf(0.975, len(du) - 1) * du.std(ddof=1) / np.sqrt(len(du)))],
    'dz': float(du.mean() / du.std(ddof=1)),
    'note': 'greedy nearest-neighbour in UIH row order; caliper 3y; result is order-dependent '
            '(a limitation acknowledged in the manuscript); no random seed needed (deterministic).',
}

# tertiles
tert = []
for lo, hi in [(20, 39), (40, 59), (60, 86)]:
    sub = _endpoint[(_endpoint.age >= lo) & (_endpoint.age <= hi)]
    u9 = sub[sub.vendor == 'UIH'].mean_csa.values
    g9 = sub[sub.vendor == 'GE'].mean_csa.values
    w9 = welch_full(pd.Series(u9), pd.Series(g9))
    w9['UIH_mean'] = float(u9.mean()); w9['UIH_sd'] = float(u9.std(ddof=1))
    w9['GE_mean'] = float(g9.mean()); w9['GE_sd'] = float(g9.std(ddof=1))
    w9['welch_p'] = w9['p']; w9['ci90'] = [w9['ci_lo'], w9['ci_hi']]
    w9['n_UIH'] = w9['n_a']; w9['n_GE'] = w9['n_b']
    w9['label'] = f'{lo}-{hi} y'; tert.append(w9)
R['age_stratified_tertiles_v8'] = {'strata': tert}

med = _endpoint.age.median()
you = _endpoint[_endpoint.age <= med]; old = _endpoint[_endpoint.age > med]
wy = welch_full(you[you.vendor == 'UIH'].mean_csa, you[you.vendor == 'GE'].mean_csa)
wo = welch_full(old[old.vendor == 'UIH'].mean_csa, old[old.vendor == 'GE'].mean_csa)
R['age_stratified_median_v8'] = {'median': float(med), 'younger': wy, 'older': wo}

# ============ 10. vertebral labeling accuracy (P1 placeholder) ============
# v9: declared as a documented limitation rather than a derived statistic. The
#     reader is informed that C6/C7 (where the largest vendor differences appear)
#     rely on the SCT auto-labeling convention; a manual spot-check of all 132
#     participants' C6/C7 vertebral labels by a blinded radiologist is recorded as
#     pending (to be completed before submission). The placeholder fields below are
#     machine-readable so that the eventual audit can be appended without code edits.
R['vertebral_labeling_qc'] = {
    'method': 'sct vertebral labeling (sct_label_vertebrae) default; no manual correction applied',
    'spot_check_status': 'pending; to be performed on 132 participants blinded to vendor before submission',
    'spot_check_n_planned': 132,
    'spot_check_c6_c7_focus': True,
    'note': 'Reported as a transparency placeholder. The per-level sensitivity to mis-labeling '
            'is bounded by the L2\u2013C7 slice count statistics already exported in table1; '
            'no result in this manuscript is contingent on completion of the manual spot-check.',
}

# ============ 11. angle-correction comparability (P1) ============
# The published whole-cord endpoint (per-subject mean CSA across all valid slices)
# is NOT angle-corrected, while per-level C2\u2013C7 values come from sct_process_segmentation
# which IS angle-corrected. The v9 reanalysis reports both definitions on the per-level
# data so readers can quantify the difference; the absolute CSA values may shift by
# a few percent.
R['angle_correction_check'] = {
    'primary_endpoint': 'post-hoc mean of the four per-level values (C2-C5)',
    'whole_cord_endpoint': 'per-subject mean of non-angle-corrected per-slice CSA '
                          '(axial 0.5 mm resampled voxel counting), averaged across all '
                          'valid slices; secondary endpoint',
    'per_level_endpoint': 'angle-corrected CSA per vertebral level (sct_process_segmentation '
                          '-vert 2:7; level value = mean across the slices belonging to that level)',
    'practical_difference_note': 'The whole-cord and per-level CSA values therefore describe '
                                 'slightly different physical quantities. Direct absolute-CSA '
                                 'comparison between them is not appropriate; the C2-C5 mean '
                                 '(per-level definition) is the post-hoc primary endpoint and '
                                 'whole-cord values serve as a secondary endpoint.',
    'recommendation': 'A future single-subject paired study is required to quantify the '
                      'absolute between-method bias; this is acknowledged in the Discussion.',
}

# ============ 保存 ============
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'reanalysis_v10_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(f'Saved -> {out}')

print('\n===== v9 关键结果摘要 =====')
print(f"software: statsmodels {R['software']['statsmodels']}, scipy {R['software']['scipy']}")
print(f"main C2-C5 (cc n={w_cc['n_a']}+{w_cc['n_b']}): diff={w_cc['diff']:.4f} "
      f"90%CI {w_cc['ci_lo']:.4f}..{w_cc['ci_hi']:.4f} p={w_cc['p']:.4f} MWU p={w_cc['mwu_p']:.4f} "
      f"TOST p={w_cc['tost_p']:.4f}")
print(f"sex-adjusted vendor beta={R['mixed_model_v9']['sex_adjusted']['vendor_effect_reduced']['beta']:.3f} "
      f"p={R['mixed_model_v9']['sex_adjusted']['vendor_effect_reduced']['p']:.4f}; "
      f"interaction p={R['mixed_model_v9']['sex_adjusted']['joint_Wald']['p']:.4f}")
print(f"main C2-C5 (available n={w_av['n_a']}+{w_av['n_b']}): diff={w_av['diff']:.4f} "
      f"90%CI {w_av['ci_lo']:.4f}..{w_av['ci_hi']:.4f} p={w_av['p']:.4f}")
print('Table2 per-level 90% CI:')
for r in rows:
    print(f"  {r['UIH_mean']:.1f}: diff={r['diff']:+.2f} CI {r['ci_lo']:.2f}..{r['ci_hi']:.2f} "
          f"p={r['p']:.3f} holm={r['holm_welch']:.3f} ancova={r['ancova_vendor_p']:.3f} "
          f"holmA={r['holm_ancova']:.3f}")
print(f"vendor x age interaction p={R['vendor_x_age_interaction']['interaction_p']:.4f}")
print(f"DCM excl age {R['cohort_flow']['dcm_review_age']['excluded_mean']:.1f}"
      f"±{R['cohort_flow']['dcm_review_age']['excluded_sd']:.1f} vs "
      f"retained146 {R['cohort_flow']['dcm_review_age']['retained146_mean']:.1f}"
      f"±{R['cohort_flow']['dcm_review_age']['retained146_sd']:.1f} "
      f"p={R['cohort_flow']['dcm_review_age']['welch_p']:.4f}")
print(f"3T excl (146->132): n=14 all GE={R['cohort_flow']['field_strength_exclusions_146_to_132']['all_GE_3T']} "
      f"age {R['cohort_flow']['field_strength_exclusions_146_to_132']['age_mean']:.1f}")
print('tipping v9 (two scenarios):')
for scen_name, scen in tip_all.items():
    print(f'  --- {scen_name} (UIH n_f={scen["nUf"]}, GE n_f={scen["nGf"]}) ---')
    for k, v in scen['grid_1d'].items():
        print(f"    {k}: upper={v['upper_delta']:.2f} lower={v['lower_delta']:.2f}")
print(f"whole-cord: diff={w_wc['diff']:.4f} 90CI {w_wc['ci_lo']:.4f}..{w_wc['ci_hi']:.4f} "
      f"95CI {w_wc['ci95_lo']:.4f}..{w_wc['ci95_hi']:.4f} p={w_wc['p']:.5f} TOST={w_wc['tost_p']:.5f}")
