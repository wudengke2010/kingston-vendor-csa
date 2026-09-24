# -*- coding: utf-8 -*-
"""KINGSTON v8 figures: redraw Fig.5 (STROBE flow, corrected) + Fig.8 (reparameterised symmetric
MNAR) + Fig.4 (updated footnote values). Others copied from v7 (unchanged)."""
import os, shutil, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = 'E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2/'
OUT = BASE + 'figures_v8/'
SRC = BASE + 'figures_v7/'
os.makedirs(OUT, exist_ok=True)

for f in ['fig1_c2c5_primary_boxplot', 'fig2_wholecord_vs_age',
          'fig3_normality_qq_per_vendor', 'fig6_perlevel_exploratory',
          'fig7_margin_sensitivity']:
    for ext in ['.png', '.pdf']:
        shutil.copy(SRC + f + ext, OUT + f + ext)

V8 = json.load(open('C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/reanalysis_v8_results.json', encoding='utf-8'))
C_UIH, C_GE = '#1f77b4', '#d62728'

# ================= Fig 5: STROBE flow (REDRAWN, all numbers verified) =================
fig, ax = plt.subplots(figsize=(9.2, 11.5))
ax.set_xlim(-2.6, 15.0)
ax.set_ylim(-0.3, 12.5)
ax.axis('off')

BX, BW = 1.6, 8.0  # main column

def box(x, y, w, h, text, fc='#eef3fb', ec='#1f4e79', fs=9, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')

def arrow_v(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color='#1f4e79', lw=1.4))

def note_right(y, text, fs=8.0):
    ax.annotate('', xy=(BX + BW + 0.12, y), xytext=(BX + BW + 0.75, y),
                arrowprops=dict(arrowstyle='<|-', color='#8b0000', lw=1.2))
    ax.text(BX + BW + 0.85, y, text, ha='left', va='center', fontsize=fs, color='#8b0000')

def note_left(y, text, fs=7.9):
    ax.annotate('', xy=(BX - 0.12, y), xytext=(BX - 0.75, y),
                arrowprops=dict(arrowstyle='<|-', color='#8b0000', lw=1.2))
    ax.text(BX - 0.85, y, text, ha='right', va='center', fontsize=fs, color='#8b0000')

# Level 1
box(BX, 11.0, BW, 1.4,
    'Two non-overlapping PACS exports of cervical spine MRI\n'
    'export 1: 80 examinations (Jan\u2013Jun 2026, \u226550 y)\n'
    'export 2: 232 examinations (Jul 2023\u2013Jul 2026, no age limit)\n'
    'deduplication: no shared identifier; cross-check (age, sex, scan date)',
    fs=8.0, bold=True)
note_right(11.7, 'excluded before processing (n = 8):\n1 non-UIH/GE (Philips); 3 without\nusable sagittal T2; 4 aged <18 y')
arrow_v(5.6, 11.0, 10.35)

# Level 2: 312 - 8 = 304, vendor split = UIH 90 / GE 214 (Philips already removed)
box(BX, 9.45, BW, 0.9, 'Eligible examinations entering pipeline\nn = 304   (UIH 90 / GE 214)', fs=9.5, bold=True)
note_right(9.9, 'processing incomplete (n = 4):\narchive series unmatchable')
arrow_v(5.6, 9.45, 8.75)

# Level 3
box(BX, 7.85, BW, 0.9, 'Processed with SCT pipeline\nn = 300   (UIH 90 / GE 210)', fs=10, bold=True)

# split arrows
ax.annotate('', xy=(3.4, 7.1), xytext=(4.2, 7.85),
            arrowprops=dict(arrowstyle='-|>', color='#1f4e79', lw=1.4))
ax.annotate('', xy=(7.8, 7.1), xytext=(7.0, 7.85),
            arrowprops=dict(arrowstyle='-|>', color='#1f4e79', lw=1.4))

# Level 4: QC
box(0.2, 5.65, 6.0, 1.45,
    'Automated QC passed: n = 213 (UIH 82 / GE 131)\nper-vendor normalised mean CSA within\n30\u2013120 mm$^2$ physiologic range;\n\u2265100 analysed cord slices', fs=8.4)
box(6.6, 5.65, 4.4, 1.45,
    'Automated QC failed: n = 87 (UIH 8 / GE 79)\nalgorithmic CSA outside range\n(median 21\u201322 mm$^2$);\nsegmentation under-performance',
    fs=8.4, fc='#fff5f5', ec='#8b0000')
note_left(4.55, 'QC-failed examinations were not\nscreened for degenerative compression;\nthe tipping-point sensitivity analysis\naddresses this stratum (Fig. 8)', fs=7.6)

arrow_v(3.2, 5.65, 4.85)

# Level 5: DCM review on source images (not segmentation)
box(BX, 3.55, BW, 1.3,
    'Neuroradiological review of source images\n(single reader, 13 y cervical-MRI experience,\nblinded to vendor and CSA)\n'
    '213 \u2192 146 (excluded 67: degenerative cord\ncompression/T2 signal change 55;\nnon-cervical series 2; earlier \u226550-y\nreview overlap 10)',
    fs=8.0, fc='#fff8e1', ec='#b8860b')
arrow_v(5.6, 3.55, 2.85)

# Level 6: 1.5T
box(BX, 1.95, BW, 0.9, '1.5 T analytic cohort\nn = 132   (UIH 51 / GE 81)', fs=9.5, bold=True)
note_right(2.4, 'GE 3.0 T (DISCOVERY MR750)\nexcluded after review: n = 14\n(all QC-passed, all GE)')
arrow_v(5.6, 1.95, 1.2)

box(BX, 0.25, BW, 0.9, 'C2\u2013C5 complete-case population\nn = 127   (UIH 51 / GE 76)', fs=9.5, bold=True)

ax.set_title('Participant and examination flow (STROBE, vendor-stratified)',
             fontsize=12.5, fontweight='bold', y=0.995)
fig.tight_layout()
fig.savefig(OUT + 'fig5_strobe_flow_vendor.png', dpi=300)
fig.savefig(OUT + 'fig5_strobe_flow_vendor.pdf')
plt.close(fig)
print('Fig.5 redrawn: 304 = 90 UIH + 214 GE; Philips excluded pre-processing; '
      '3T excluded after review (146->132); no placeholder')

# ================= Fig 8: tipping 2D (reparameterised symmetric) =================
grid = V8['tipping_2d_v8']
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
panels = [('GE', 'A. GE-anchored scenario'), ('symmetric', 'B. Symmetric shift scenario')]
for ax, (anchor, title) in zip(axes, panels):
    g = grid[f'{anchor}_sigma1.0']
    deltas = np.array(g['deltas']); ps = np.array(g['tost_p'])
    im = ax.imshow(np.array(ps).reshape(1, -1), aspect='auto', origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=0.25, extent=[deltas.min(), deltas.max(), 0, 1])
    # overlay tipping contours per sigma
    for sc, ls in [(1.0, '-'), (1.25, '--'), (1.5, ':')]:
        gg = grid[f'{anchor}_sigma{sc}']
        dd = np.array(gg['deltas']); pp = np.array(gg['tost_p'])
        ax.plot(dd[np.argsort(dd)][::-1] * 0 + dd, np.where(pp >= 0.05, 0.5, np.nan), ls=ls,
                color='#333333', lw=0)  # placeholder no-op
        # vertical line at tipping point
        tip_v = V8['tipping_point_v8']['grid_1d'][f'{anchor}_sigma{sc}']['upper_delta']
        if tip_v is not None:
            ax.axvline(tip_v, color='k', ls=ls, lw=1.4)
    ax.set_xlabel('Assumed vendor difference in QC-failed stratum,\nUIH \u2212 GE (mm$^2$)', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_yticks([])
    # annotate tipping values
    for sc, yy in [(1.0, 0.80), (1.25, 0.55), (1.5, 0.30)]:
        tip_v = V8['tipping_point_v8']['grid_1d'][f'{anchor}_sigma{sc}']['upper_delta']
        ax.annotate(f'\u00d71.0: +{tip_v:.1f}' if sc == 1.0 else
                    (f'\u00d71.25: +{tip_v:.1f}' if sc == 1.25 else f'\u00d71.5: +{tip_v:.1f}'),
                    xy=(tip_v, yy), fontsize=8, color='k',
                    xytext=(tip_v + 0.8, yy), va='center')
    ax.axvline(V8['tipping_point_v8']['d_observed_base'], color='#1f77b4', lw=1.2, ls='-.',
               alpha=0.8)
    ax.annotate('observed\nbase diff', xy=(V8['tipping_point_v8']['d_observed_base'], 0.12),
                fontsize=7.5, color='#1f77b4', ha='center',
                xytext=(V8['tipping_point_v8']['d_observed_base'] + 0.2, 0.04))
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label('TOST p value (\u00b15 mm\u00b2)', fontsize=9)
fig.suptitle('Conditional tipping-point sensitivity: TOST p \u2265 0.05 (equivalence lost) '
             'right of each vertical line', fontsize=10.5)
fig.savefig(OUT + 'fig8_tipping_2d.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT + 'fig8_tipping_2d.pdf', bbox_inches='tight')
plt.close(fig)
print('Fig.8 redrawn with unified delta = actual fail-stratum vendor difference')

# ================= Fig 4: equivalence dual endpoint (update footnote values) =================
# regenerate with v8 values (structure identical to v7, values from JSON)
me = V8['main_endpoint']['complete_case']
wcx = V8['whole_cord']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (name, d, lo, hi, pv) in zip(axes, [
        ('C2\u2013C5 mean (complete case, n = 127)', me['diff'], me['ci_lo'], me['ci_hi'], me['p']),
        ('Whole-cord mean (n = 132)', wcx['diff'], wcx['ci_lo'], wcx['ci_hi'], wcx['p'])]):
    ax.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt='o', color='#1f4e79',
                capsize=5, markersize=8, lw=1.8)
    ax.axvline(-5, color='#8b0000', ls='--', lw=1.2)
    ax.axvline(5, color='#8b0000', ls='--', lw=1.2)
    ax.axvline(0, color='#666666', ls='-', lw=0.8)
    ax.fill_betweenx([-1, 1], -5, 5, color='#2e7d32', alpha=0.07)
    ax.set_xlim(-7, 7); ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('UIH \u2212 GE difference (mm\u00b2)', fontsize=9.5)
    ax.set_title(f'{name}\ndiff = {d:.2f} mm\u00b2; Welch p = {pv:.3f}', fontsize=9.5)
    ax.set_yticks([])
axes[0].annotate(f'TOST p = {me["tost_p"]:.4f}', xy=(0.5, -0.55), xycoords='axes fraction',
                 fontsize=9, ha='center', color='#2e7d32')
axes[1].annotate(f'TOST p = {wcx["tost_p"]:.4f}', xy=(0.5, -0.55), xycoords='axes fraction',
                 fontsize=9, ha='center', color='#2e7d32')
fig.suptitle('Equivalence testing against \u00b15 mm\u00b2 (90% CIs)', fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + 'fig4_equivalence_dual_endpoint.png', dpi=300)
fig.savefig(OUT + 'fig4_equivalence_dual_endpoint.pdf')
plt.close(fig)
print('Fig.4 regenerated with v8 values')
print('DONE figures_v8 ->', OUT)
