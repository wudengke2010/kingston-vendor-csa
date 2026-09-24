# -*- coding: utf-8 -*-
"""
KINGSTON v7 figures (第四轮评审修复):
- Fig 3: probplot解包bug修复 —— 参考线此前误用数据值当斜率/截距 (y≈50x+52)
- Fig 4: tipping脚注更新为修正公式后的阈值(GE-anchored +9.5/+8.2; symmetric MNAR +3.4)
- Fig 7: 双panel —— whole-cord + C2-C5主结局多界值敏感性
- Fig 8: 修正公式(层间SS)后的二维tipping heatmap, 双锚定情景(GE-anchored + symmetric MNAR)
- Fig 1/2/5/6: 与v6一致(拷贝)
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as st
import json

BASE = 'E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2/'
OUT = BASE + 'figures_v7/'
SRC = BASE + 'figures_v6/'
os.makedirs(OUT, exist_ok=True)

C_UIH, C_GE = '#1f77b4', '#d62728'

# copy unchanged figures
for f in ['fig1_c2c5_primary_boxplot', 'fig2_wholecord_vs_age',
          'fig5_strobe_flow_vendor', 'fig6_perlevel_exploratory']:
    for ext in ['.png', '.pdf']:
        shutil.copy(SRC + f + ext, OUT + f + ext)

wc = pd.read_csv(BASE + 'final_cohort_15T.csv')
pl = pd.read_csv(BASE + 'perlevel_csa_moderate.csv')
params = pd.read_csv(BASE + 'supplementary_scan_parameters.csv')
t3 = params[params.field_T == 3.0].patient_id.tolist()
c15 = pl[~pl.patient_id.isin(t3)].copy()
cc = c15.dropna(subset=['C2', 'C3', 'C4', 'C5']).copy()
cc['c2c5'] = cc[['C2', 'C3', 'C4', 'C5']].mean(axis=1)

with open('C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/reanalysis_v7_results.json') as fp:
    V7 = json.load(fp)
with open('C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/reanalysis_v6_results.json') as fp:
    V6 = json.load(fp)   # whole-cord margin sensitivity unchanged (base-132 observed cohort)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

u_w_vals = wc[wc.vendor == 'UIH'].mean_csa.values
g_w_vals = wc[wc.vendor == 'GE'].mean_csa.values

# ---------------- Fig 3: QQ plots (probplot unpacking FIXED) ----------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
ax = axes[0]
bins = np.linspace(min(u_w_vals.min(), g_w_vals.min()), max(u_w_vals.max(), g_w_vals.max()), 14)
ax.hist(u_w_vals, bins=bins, alpha=0.55, color=C_UIH, label='UIH', edgecolor='white')
ax.hist(g_w_vals, bins=bins, alpha=0.55, color=C_GE, label='GE', edgecolor='white')
ax.set_xlabel('Whole-cord mean CSA (mm$^2$)'); ax.set_ylabel('Count')
ax.set_title('Distribution', fontsize=10)
ax.legend(frameon=False, fontsize=9)

for ax, vals, c, name in [(axes[1], u_w_vals, C_UIH, 'UIH'), (axes[2], g_w_vals, C_GE, 'GE')]:
    (osm, osr), (slope, intercept, r) = st.probplot(vals, dist='norm')   # CORRECT unpacking
    ax.scatter(osm, osr, color=c, s=18, alpha=0.7, edgecolor='none')
    xs = np.array([osm.min(), osm.max()])
    ax.plot(xs, slope * xs + intercept, color='black', lw=1.2, ls='--')  # fitted reference line
    sw_p = st.shapiro(vals).pvalue
    ax.set_title(f'{name} Q\u2013Q plot\nShapiro\u2013Wilk p = {sw_p:.3f} (R$^2$ = {r**2:.3f})', fontsize=10)
    ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles (mm$^2$)')

fig.suptitle('Normality assessment of whole-cord CSA by vendor (reference line = normal-theory fit)', fontsize=11)
fig.tight_layout()
fig.savefig(OUT + 'fig3_normality_qq_per_vendor.png', dpi=300)
fig.savefig(OUT + 'fig3_normality_qq_per_vendor.pdf')
plt.close(fig)

# ---------------- Fig 4: dual-endpoint equivalence CI ----------------
def diff_ci_welch(a, b):
    d = a.mean() - b.mean()
    se2_a = a.var(ddof=1) / len(a)
    se2_b = b.var(ddof=1) / len(b)
    se = np.sqrt(se2_a + se2_b)
    df_ws = (se2_a + se2_b) ** 2 / (se2_a ** 2 / (len(a) - 1) + se2_b ** 2 / (len(b) - 1))
    tc = st.t.ppf(0.95, df_ws)
    return d, d - tc * se, d + tc * se, df_ws

def tost_p(d, se, m, df):
    return max(st.t.cdf((d - m) / se, df), 1 - st.t.cdf((d + m) / se, df))

u_c2 = cc[cc.vendor == 'UIH'].c2c5.values
g_c2 = cc[cc.vendor == 'GE'].c2c5.values
d_p, lo_p, hi_p, df_p = diff_ci_welch(u_c2, g_c2)
d_s, lo_s, hi_s, df_s = diff_ci_welch(u_w_vals, g_w_vals)
se_p = (d_p - lo_p) / st.t.ppf(0.95, df_p)
se_s = (d_s - lo_s) / st.t.ppf(0.95, df_s)
tost_p_p = tost_p(d_p, se_p, 5, df_p)
tost_p_s = tost_p(d_s, se_s, 5, df_s)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ep = [
    ('Post-hoc revised main endpoint:\nmean C2\u2013C5 CSA (complete-case, n = 127)',
     d_p, lo_p, hi_p, C_UIH, tost_p_p, df_p),
    ('Secondary endpoint:\nwhole-cord mean (n = 132)',
     d_s, lo_s, hi_s, C_GE, tost_p_s, df_s),
]
for i, (label, d, lo, hi, c, pt, dfv) in enumerate(ep):
    y = 1 - i
    ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='o', color=c, capsize=6,
                markersize=9, lw=2.2, capthick=2.2)
    ax.annotate(f'{d:.2f} mm$^2$; 90% CI {lo:.2f} to {hi:.2f}\nTOST p = {pt:.3f} (df = {dfv:.1f})',
                xy=(d, y), xytext=(0, 14), textcoords='offset points',
                ha='center', va='bottom', fontsize=8.8)
ax.axvspan(-5, 5, color='#2ca02c', alpha=0.10)
ax.axvline(5, color='#2ca02c', ls='--', lw=1.4)
ax.axvline(-5, color='#2ca02c', ls='--', lw=1.4)
ax.text(0, 1.86, '\u00b15 mm$^2$ equivalence margin', ha='center', va='top',
        fontsize=9, color='#1a7a1a', fontweight='bold')
ax.set_yticks([1, 0])
ax.set_yticklabels([e[0] for e in ep], fontsize=9)
ax.set_ylim(-0.85, 1.95)
ax.set_xlim(-6.5, 8.0)
ax.set_xlabel('Between-vendor difference in CSA, UIH \u2212 GE (mm$^2$)')
ax.set_title('Equivalence of both endpoints against the \u00b15 mm$^2$ margin\n'
             '(Welch\u2013Satterthwaite df; both TOST p < 0.05)', fontsize=10.5)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.text(0.98, 0.015,
         'Conditional tipping-point (Fig.\u00a08, corrected variance formula): under GE-anchored scenarios the whole-cord\n'
         'equivalence reverses only if the assumed QC-failed-stratum vendor difference exceeds +9.5 mm$^2$ (SD \u00d71.0)\n'
         'or +8.2 mm$^2$ (SD \u00d71.5); under symmetric MNAR shifting the threshold is +3.4 mm$^2$ (SD \u00d71.0)',
         ha='right', va='bottom', fontsize=7.6, color='#444444')
fig.savefig(OUT + 'fig4_equivalence_dual_endpoint.png', dpi=300)
fig.savefig(OUT + 'fig4_equivalence_dual_endpoint.pdf')
plt.close(fig)

# ---------------- Fig 7: margin sensitivity, both endpoints ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
margins = [3.0, 4.0, 5.0, 6.0, 7.0]

# Panel A: whole-cord n=132
tosts_w = [s['base_132']['tost_p'] for s in V6['margin_sensitivity_wholecord']]
ax = axes[0]
colors = ['#d62728' if t >= 0.05 else '#2ca02c' for t in tosts_w]
bars = ax.bar([f'\u00b1{m:.0f}' for m in margins], tosts_w, color=colors, edgecolor='black', lw=0.6)
ax.axhline(0.05, color='black', ls='--', lw=1)
ax.set_xlabel('Equivalence margin (mm$^2$)')
ax.set_ylabel('TOST p value')
ax.set_title('Whole-cord endpoint (n = 132)', fontsize=10.5)
for b, t in zip(bars, tosts_w):
    lab = 'p < 0.001' if t < 0.0005 else f'p = {t:.3f}'
    ax.text(b.get_x() + b.get_width() / 2, t + 0.006, lab, ha='center', fontsize=8.5)
ax.set_ylim(0, max(tosts_w) * 1.2)

# Panel B: C2-C5 n=127 (new)
tosts_c = [m['tost_p'] for m in V7['margin_sensitivity_c2c5']['margins']]
ax = axes[1]
colors = ['#d62728' if t >= 0.05 else '#2ca02c' for t in tosts_c]
bars = ax.bar([f'\u00b1{m:.0f}' for m in margins], tosts_c, color=colors, edgecolor='black', lw=0.6)
ax.axhline(0.05, color='black', ls='--', lw=1)
ax.set_xlabel('Equivalence margin (mm$^2$)')
ax.set_title('Post-hoc revised main endpoint C2\u2013C5 (complete-case, n = 127)', fontsize=10.5)
for b, t in zip(bars, tosts_c):
    lab = 'p < 0.001' if t < 0.0005 else f'p = {t:.3f}'
    ax.text(b.get_x() + b.get_width() / 2, t + 0.006, lab, ha='center', fontsize=8.5)
ax.set_ylim(0, max(tosts_c) * 1.2)

handles = [Patch(fc='#d62728', label='equivalence NOT supported (p \u2265 0.05)'),
           Patch(fc='#2ca02c', label='equivalence supported (p < 0.05)')]
axes[0].legend(handles=handles, frameon=False, fontsize=8, loc='upper right')
fig.suptitle('Sensitivity of equivalence conclusions to margin choice', fontsize=11)
fig.tight_layout()
fig.savefig(OUT + 'fig7_margin_sensitivity.png', dpi=300)
fig.savefig(OUT + 'fig7_margin_sensitivity.pdf')
plt.close(fig)

# ---------------- Fig 8: corrected 2D tipping heatmap, two anchoring panels ----------------
deltas = [-10, -5, 0, 2.25, 5, 8, 10, 15, 20, 25, 30]
sigmas = [1.0, 1.25, 1.5]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9))

panels = [
    ('GE', 'GE-anchored: GE-fail mean = observed GE mean;\nUIH-fail mean = anchor + \u03b4'),
    ('symmetric', 'Symmetric MNAR: UIH-fail = m$_{UIH}$ + \u03b4/2;\nGE-fail = m$_{GE}$ \u2212 \u03b4/2'),
]
for k, (anchor, subtitle) in enumerate(panels):
    scen = V7['tipping_point_2d_corrected']['scenarios'][anchor]
    matrix = np.zeros((len(sigmas), len(deltas)))
    for i, s in enumerate(scen):
        for j, d in enumerate(deltas):
            matrix[i, j] = s[f'delta_{d}_tost_p']
    ax = axes[k]
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.1,
                   extent=[-0.5, len(deltas) - 0.5, len(sigmas) - 0.5, -0.5])
    for i in range(len(sigmas)):
        for j in range(len(deltas)):
            v = matrix[i, j]
            col = 'white' if (v < 0.015 or v > 0.09) else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.6, color=col)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([str(d) for d in deltas], fontsize=8.5)
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f'\u00d7{s}' for s in sigmas], fontsize=8.5)
    ax.set_xlabel('Assumed vendor difference in QC-failed stratum (\u03b4, mm$^2$)', fontsize=9)
    if k == 0:
        ax.set_ylabel('QC-failed stratum\nSD scale factor', fontsize=9)
    ax.set_title(f'({chr(65 + k)}) {subtitle}', fontsize=9)

fig.suptitle('Conditional tipping-point sensitivity: TOST p for the whole-cord comparison '
             '(merged variance = within + between stratum SS)', fontsize=10.5)
fig.text(0.5, 0.015,
         'Green = equivalence supported (p < 0.05); red = reversed (p \u2265 0.05). '
         'The QC-failed stratum did not undergo the same DCM screening as the base cohort; '
         'results are conditional on the stated assumptions.',
         ha='center', va='bottom', fontsize=8, color='#444444')
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT + 'fig8_tipping_2d.png', dpi=300)
fig.savefig(OUT + 'fig8_tipping_2d.pdf')
plt.close(fig)

print('saved v7 figures ->', OUT)
print('regenerated: fig3 (probplot fix), fig4 (tipping footnote), fig7 (2-panel), fig8 (corrected formula)')
