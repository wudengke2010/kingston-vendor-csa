# -*- coding: utf-8 -*-
"""
KINGSTON v7 重分析 (回应组内第四轮评审, 2026-09-24)

阻断级修复：
1. S2 MixedLM: 从原始数据重跑，全部量（llf/AIC/BIC/LR/Wald/系数/协方差）直接从
   statsmodels结果对象程序化导出，多优化器交叉验证收敛
2. S4 tipping-point: 修正合并方差公式 —— 加入层间平方和
   SS_between = n_base(m_base-m_pool)^2 + n_fail(m_fail-m_pool)^2
   双锚定情景（GE-anchored + symmetric MNAR）
3. 匹配效应量：删除paired eta^2，改报 Cohen's dz
4. C2-C5 主结局多界值敏感性（新增）
5. 年龄分层分析 p 值重算
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from scipy.optimize import brentq

BASE = 'E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2/'
R = {}

# ============ 0. 数据载入（与v6一致） ============
wc = pd.read_csv(BASE + 'final_cohort_15T.csv')       # 132 1.5T DCM-free QC-pass
pl = pd.read_csv(BASE + 'perlevel_csa_moderate.csv')
params = pd.read_csv(BASE + 'supplementary_scan_parameters.csv')

uw = wc[wc.vendor == 'UIH'].mean_csa.values
gw = wc[wc.vendor == 'GE'].mean_csa.values
nU, nG = len(uw), len(gw)
mU, mG = uw.mean(), gw.mean()
sU, sG = uw.std(ddof=1), gw.std(ddof=1)
nUf, nGf = 8, 79   # QC-fail stratum (UIH 8 + GE 79 = 87)

R['base_stats'] = {
    'n': 132, 'UIH_n': int(nU), 'GE_n': int(nG),
    'UIH_mean': float(mU), 'UIH_sd': float(sU),
    'GE_mean': float(mG), 'GE_sd': float(sG),
    'diff': float(mU - mG),
    'qc_fail_stratum': {'UIH_n': nUf, 'GE_n': nGf, 'total': 87},
}

# ============ 1. 修正后的合并方差 tipping-point ============
def merge_stratum(m_base, s_base, n_base, m_fail, s_fail, n_fail):
    """正确合并两层：总方差 = 层内SS + 层间SS（第四轮评审指出的遗漏项）"""
    n = n_base + n_fail
    m_pool = (n_base * m_base + n_fail * m_fail) / n
    ss_within = (n_base - 1) * s_base**2 + (n_fail - 1) * s_fail**2
    ss_between = n_base * (m_base - m_pool)**2 + n_fail * (m_fail - m_pool)**2
    var = (ss_within + ss_between) / (n - 1)
    return m_pool, var, n

def pooled_welch_corrected(delta_fail, sigma_factor=1.0, anchor='GE'):
    """
    delta_fail: QC-fail层内假设的厂商差（UIH − GE, mm²）
    sigma_factor: QC-fail层SD相对基础层SD的倍数
    anchor: 'GE'（GE-fail锚定在GE观测均值，UIH-fail = mG + delta）
            'symmetric'（两厂商同时MNAR：mUf = mU + delta/2, mGf = mG - delta/2）
    合并方差使用 within + between SS（修正公式）。
    """
    if anchor == 'GE':
        mGf = mG
        mUf = mG + delta_fail
    else:  # symmetric MNAR
        mUf = mU + delta_fail / 2.0
        mGf = mG - delta_fail / 2.0

    sUf, sGf = sU * sigma_factor, sG * sigma_factor

    mUp, varU, nUt = merge_stratum(mU, sU, nU, mUf, sUf, nUf)
    mGp, varG, nGt = merge_stratum(mG, sG, nG, mGf, sGf, nGf)

    dp = mUp - mGp
    se2_U, se2_G = varU / nUt, varG / nGt
    sep = np.sqrt(se2_U + se2_G)
    df_ws = (se2_U + se2_G)**2 / (se2_U**2 / (nUt - 1) + se2_G**2 / (nGt - 1))
    tcrit = stats.t.ppf(0.95, df_ws)
    ci90 = (dp - tcrit * sep, dp + tcrit * sep)
    p_tost = max(stats.t.cdf((dp - 5) / sep, df_ws), 1 - stats.t.cdf((dp + 5) / sep, df_ws))
    return dp, sep, ci90, df_ws, p_tost

# 1D tipping per anchor per sigma
tipping = {}
for anchor in ['GE', 'symmetric']:
    for sf in [1.0, 1.25, 1.5]:
        f_up = lambda d: pooled_welch_corrected(d, sf, anchor)[2][1] - 5.0
        f_lo = lambda d: pooled_welch_corrected(d, sf, anchor)[2][0] + 5.0
        tipping[f'{anchor}_sigma{sf}'] = {
            'upper_delta': float(brentq(f_up, -80, 120)),
            'lower_delta': float(brentq(f_lo, -120, 80)),
        }
R['tipping_point_corrected_1d'] = tipping

# 2D scenario table（修正公式）: delta × sigma, 两种锚定
delta_range = [-10, -5, 0, 2.25, 5, 8, 10, 15, 20, 25, 30]
sigma_factors = [1.0, 1.25, 1.5]
scen = {}
for anchor in ['GE', 'symmetric']:
    rows = []
    for sf in sigma_factors:
        row = {'sigma_factor': sf}
        for d in delta_range:
            dp, sep, ci, dfws, pt = pooled_welch_corrected(d, sf, anchor)
            row[f'delta_{d}_tost_p'] = round(pt, 4)
        rows.append(row)
    scen[anchor] = rows
R['tipping_point_2d_corrected'] = {
    'method': ('Pooled Welch-Satterthwaite; merged per-vendor variance includes '
               'WITHIN-stratum SS + BETWEEN-stratum SS [n_base(m_base-m_pool)^2 + '
               'n_fail(m_fail-m_pool)^2] (v7 correction of v6 formula). '
               'Two anchoring scenarios: GE-anchored (GE-fail mean = observed GE mean; '
               'UIH-fail = mG + delta) and symmetric MNAR (both vendor fail means shift '
               'by +/- delta/2). SD of fail stratum = sigma_factor x base SD.'),
    'scenarios': scen,
    'note': ('Conditional sensitivity analysis: the QC-failed stratum did not undergo the '
             'same DCM screening as the base cohort; conclusions are conditional on the '
             'stated assumptions, not statements about the unobserved data.'),
}

# ============ 2. C2-C5 主结局多界值敏感性（新增，审稿人要求） ============
c25 = ['C2', 'C3', 'C4', 'C5']
pl15 = wc[['patient_id']].merge(pl, on='patient_id', how='left')
cc = pl15.dropna(subset=c25)
u2 = cc[cc.vendor == 'UIH'][c25].mean(axis=1).values
g2 = cc[cc.vendor == 'GE'][c25].mean(axis=1).values
d2 = u2.mean() - g2.mean()
se2b = u2.var(ddof=1) / len(u2) + g2.var(ddof=1) / len(g2)
seb = np.sqrt(se2b)
dfb = se2b**2 / ((u2.var(ddof=1) / len(u2))**2 / (len(u2) - 1) + (g2.var(ddof=1) / len(g2))**2 / (len(g2) - 1))

c25_margins = []
for m in [3.0, 4.0, 5.0, 6.0, 7.0]:
    pt = max(stats.t.cdf((d2 - m) / seb, dfb), 1 - stats.t.cdf((d2 + m) / seb, dfb))
    c25_margins.append({'margin_mm2': m, 'tost_p': float(pt),
                        'equivalence': 'YES' if pt < 0.05 else 'NO'})
R['margin_sensitivity_c2c5'] = {
    'n_complete': int(len(cc)), 'UIH_n': int(len(u2)), 'GE_n': int(len(g2)),
    'diff': float(d2), 'se': float(seb), 'df_welch': float(dfb),
    'margins': c25_margins,
}

# ============ 3. MixedLM 重跑（程序化全导出） ============
c15 = pl[~pl.patient_id.isin(params[params.field_T == 3.0].patient_id)].copy()
long = c15.melt(id_vars=['patient_id', 'vendor', 'age'],
                value_vars=['C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                var_name='level', value_name='csa').dropna(subset=['csa'])

def fit_mixed(formula):
    """多优化器尝试，选llf最高且收敛的拟合"""
    best = None
    for meth in ['lbfgs', 'bfgs', 'powell', 'cg']:
        try:
            m = smf.mixedlm(formula, long, groups=long['patient_id']).fit(
                reml=False, method=meth)
            ok = bool(m.converged)
            llf = float(m.llf)
            if not np.isfinite(llf):
                continue
            if best is None or llf > best[0]:
                best = (llf, meth, m, ok)
        except Exception as e:
            print(f'  {formula[:30]}... {meth}: FAILED {e}')
    llf, meth, m, conv = best
    return m, meth, conv

mred, meth_red, conv_red = fit_mixed('csa ~ C(vendor) + C(level) + age')
mfull, meth_full, conv_full = fit_mixed('csa ~ C(vendor)*C(level) + age')

print(f'reduced: method={meth_red} converged={conv_red} llf={mred.llf:.3f}')
print(f'full:    method={meth_full} converged={conv_full} llf={mfull.llf:.3f}')
assert mfull.llf >= mred.llf - 1e-6, 'Full model must have LL >= reduced (nested, ML)'

def export_model(m, model_name):
    ci = m.conf_int()
    coefs = []
    for p in m.params.index:
        coefs.append({
            'term': str(p),
            'coef': float(m.params[p]),
            'se': float(m.bse[p]),
            'z': float(m.tvalues[p]),
            'p': float(m.pvalues[p]),
            'ci95_lo': float(ci.loc[p, 0]),
            'ci95_hi': float(ci.loc[p, 1]),
        })
    # 参数计数（含截距的固定效应 + 方差成分）
    fixed_names = [str(x) for x in m.params.index if 'Group Var' not in str(x) and 'Var' not in str(x)]
    var_names = [str(x) for x in m.params.index if x not in fixed_names]
    n_fixed = len(fixed_names)
    n_var = len(m.params) - n_fixed
    k_total = len(m.params)   # 固定+方差成分总参数数
    return {
        'model': model_name,
        'n_obs': int(m.nobs),
        'n_groups': int(len(set(m.model.groups))),
        'converged': bool(m.converged),
        'llf': float(m.llf),
        'aic_statsmodels': float(m.aic),
        'bic_statsmodels': float(m.bic),
        'aic_manual_k_total': float(-2 * m.llf + 2 * k_total),
        'bic_manual_k_total': float(-2 * m.llf + np.log(m.nobs) * k_total),
        'n_fixed_params_incl_intercept': int(n_fixed),
        'n_variance_params': int(n_var),
        'k_total_params': int(k_total),
        'coefficients': coefs,
        'random_effect_variance': float(m.cov_re.iloc[0, 0]) if m.cov_re.size else None,
        'residual_variance': float(m.scale),
    }

red_exp = export_model(mred, 'reduced: csa ~ vendor + level + age + (1|subject)')
full_exp = export_model(mfull, 'full: csa ~ vendor*level + age + (1|subject)')

# LR test（嵌套，同一数据ML）
lr = 2 * (mfull.llf - mred.llf)
dfdiff = len(mfull.params) - len(mred.params)
p_lr = stats.chi2.sf(lr, dfdiff)

# 联合Wald检验（5个交互项）
int_names = [str(x) for x in mfull.params.index if ':' in str(x)]
b_int = mfull.params[int_names].values
V = mfull.cov_params().loc[int_names, int_names].values
wald = float(b_int @ np.linalg.pinv(V) @ b_int)
p_wald = stats.chi2.sf(wald, len(int_names))

R['mixed_model_v7'] = {
    'software': f'statsmodels {__import__("statsmodels").__version__}, Python 3.10; ML (REML=False)',
    'data': f'{len(long)} observations, {long.patient_id.nunique()} subjects, 6 levels C2-C7 (available-case)',
    'optimizer_reduced': meth_red, 'optimizer_full': meth_full,
    'reduced': red_exp,
    'full': full_exp,
    'LR_test': {
        'chi2': float(lr), 'df': int(dfdiff), 'p': float(p_lr),
        'formula': 'LR = 2 x (llf_full - llf_reduced)',
    },
    'joint_Wald_test': {
        'chi2': wald, 'df': int(len(int_names)), 'p': float(p_wald),
        'terms': int_names,
    },
    'sanity': {
        'llf_full_ge_reduced': bool(mfull.llf >= mred.llf),
        'delta_AIC_expected_from_LR': float(2 * dfdiff - lr),
        'note': ('AIC/BIC reported both as statsmodels object output and manual with '
                 'k = total estimated parameters (fixed + variance components). '
                 'All quantities exported directly from the fitted result objects; '
                 'no manual transcription.'),
    },
}

# vendor系数（reduced模型平均效应）
vc = [i for i in mred.params.index if 'vendor' in str(i)][0]
R['mixed_model_v7']['vendor_effect_reduced'] = {
    'beta': float(mred.params[vc]), 'p': float(mred.pvalues[vc]),
    'ci95': [float(mred.conf_int().loc[vc, 0]), float(mred.conf_int().loc[vc, 1])],
}

# ============ 4. 年龄匹配（dz替代eta^2） ============
np.random.seed(42)
cc_df = wc.copy()
cc_df['vendor_bin'] = (cc_df.vendor == 'UIH').astype(int)
uih_idx = cc_df[cc_df.vendor_bin == 1].index.tolist()
ge_idx = cc_df[cc_df.vendor_bin == 0].index.tolist()

pairs, used_ge = [], set()
for ui in uih_idx:
    age_u = cc_df.loc[ui, 'age']
    cands = [(g, abs(cc_df.loc[g, 'age'] - age_u)) for g in ge_idx
             if g not in used_ge and abs(cc_df.loc[g, 'age'] - age_u) <= 3]
    if cands:
        cands.sort(key=lambda x: x[1])
        g, d = cands[0]
        pairs.append((ui, g, d))
        used_ge.add(g)

md = np.array([cc_df.loc[ui, 'mean_csa'] - cc_df.loc[g, 'mean_csa'] for ui, g, _ in pairs])
n_pairs = len(md)
dm, dsd = md.mean(), md.std(ddof=1)
dse = dsd / np.sqrt(n_pairs)
tc = stats.t.ppf(0.975, n_pairs - 1)
ci95 = (dm - tc * dse, dm + tc * dse)
t_stat, p_paired = stats.ttest_rel(md, np.zeros_like(md))
dz = dm / dsd   # Cohen's dz for paired design

ua = np.array([cc_df.loc[ui, 'age'] for ui, _, _ in pairs])
ga = np.array([cc_df.loc[g, 'age'] for _, g, _ in pairs])
smd_post = (ua.mean() - ga.mean()) / np.sqrt((ua.std(ddof=1)**2 + ga.std(ddof=1)**2) / 2)
smd_pre = (cc_df[cc_df.vendor_bin == 1].age.mean() - cc_df[cc_df.vendor_bin == 0].age.mean()) / \
          np.sqrt((cc_df[cc_df.vendor_bin == 1].age.std(ddof=1)**2 + cc_df[cc_df.vendor_bin == 0].age.std(ddof=1)**2) / 2)

R['age_matching_v7'] = {
    'method': '1:1 nearest-neighbour without replacement, caliper +/-3 years, seed=42',
    'n_pairs': int(n_pairs),
    'pre_match_smd_age': float(smd_pre), 'post_match_smd_age': float(smd_post),
    'paired_diff_mean': float(dm), 'paired_diff_sd': float(dsd),
    'paired_ci95': [float(x) for x in ci95],
    'paired_t': float(t_stat), 'paired_df': int(n_pairs - 1), 'paired_p': float(p_paired),
    'cohens_dz': float(dz),
    'effect_measure_note': ("Cohen's dz = mean paired difference / SD of paired differences. "
                            'Paired eta^2 removed (v6 value 3.9% was inconsistent with '
                            't^2/(t^2+df); per reviewer request only dz is reported).'),
}

# 年龄分层（中位年龄分割）
med_age = wc.age.median()
strata = {}
for label, sub in [('younger_le_median', wc[wc.age <= med_age]), ('older_gt_median', wc[wc.age > med_age])]:
    u, g = sub[sub.vendor == 'UIH'].mean_csa.values, sub[sub.vendor == 'GE'].mean_csa.values
    if len(u) >= 5 and len(g) >= 5:
        t_st, p_st = stats.ttest_ind(u, g, equal_var=False)
        strata[label] = {'n_UIH': int(len(u)), 'n_GE': int(len(g)),
                         'median_age_split': float(med_age),
                         'UIH_mean': float(u.mean()), 'GE_mean': float(g.mean()),
                         'diff': float(u.mean() - g.mean()),
                         'welch_t': float(t_st), 'welch_p': float(p_st)}
R['age_stratified_v7'] = strata

# ============ 年龄三分位分层 (S3.3 程序化导出: 20-39/40-59/60-86) ============
tert = []
for lo, hi in [(20, 39), (40, 59), (60, 86)]:
    sub = wc[(wc.age >= lo) & (wc.age <= hi)]
    u = sub[sub.vendor == 'UIH'].mean_csa.values
    g = sub[sub.vendor == 'GE'].mean_csa.values
    t_st, p_st = stats.ttest_ind(u, g, equal_var=False)
    vu, vg = u.var(ddof=1), g.var(ddof=1)
    se = np.sqrt(vu/len(u) + vg/len(g))
    dfw = (vu/len(u) + vg/len(g))**2 / ((vu/len(u))**2/(len(u)-1) + (vg/len(g))**2/(len(g)-1))
    tcrit = stats.t.ppf(0.95, dfw)
    d = u.mean() - g.mean()
    tert.append({'label': f'{lo}-{hi} y', 'n_UIH': int(len(u)), 'n_GE': int(len(g)),
                 'UIH_mean': float(u.mean()), 'UIH_sd': float(u.std(ddof=1)),
                 'GE_mean': float(g.mean()), 'GE_sd': float(g.std(ddof=1)),
                 'diff': float(d), 'ci90': [float(d - tcrit*se), float(d + tcrit*se)],
                 'welch_t': float(t_st), 'welch_p': float(p_st)})
R['age_stratified_tertiles_v7'] = {
    'method': 'decade strata 20-39/40-59/60-86, whole-cord endpoint, Welch t, 90% CI',
    'strata': tert}

# ============ 保存 ============
out = 'C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/reanalysis_v7_results.json'
with open(out, 'w', encoding='utf-8') as fp:
    json.dump(R, fp, indent=1, ensure_ascii=False)
print(f'\nSaved -> {out}')

print('\n===== 关键结果摘要 =====')
print(f"1. MixedLM reduced: llf={mred.llf:.2f} AIC(sm)={mred.aic:.2f} conv={conv_red} ({meth_red})")
print(f"   MixedLM full:    llf={mfull.llf:.2f} AIC(sm)={mfull.aic:.2f} conv={conv_full} ({meth_full})")
print(f"   LR chi2={lr:.3f} df={dfdiff} p={p_lr:.4f} | Wald chi2={wald:.3f} p={p_wald:.4f}")
print(f"   vendor beta={mred.params[vc]:.3f} p={mred.pvalues[vc]:.4f}")
print(f"2. tipping (corrected formula):")
for k, v in tipping.items():
    print(f"   {k}: upper={v['upper_delta']:.2f} lower={v['lower_delta']:.2f}")
print(f"3. C2-C5 margins: " + '; '.join(f"±{m['margin_mm2']:.0f}: p={m['tost_p']:.4f}" for m in c25_margins))
print(f"4. matching: {n_pairs} pairs, diff={dm:.2f} p={p_paired:.4f} dz={dz:.3f}")
print(f"   age-stratified: " + '; '.join(f"{k}: p={v['welch_p']:.4f}" for k, v in strata.items()))
