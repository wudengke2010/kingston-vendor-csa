# -*- coding: utf-8 -*-
"""KINGSTON v9 figures.

v8 -> v9 changes:
- Fig.5 layout reworked so QC-passed / QC-failed boxes do not overlap; 67
  exclusion reasons separated into mutually-exclusive categories; consistent
  with the body-text narrative about screening scope.
- Fig.8 is now a genuine 2D heatmap (delta on x, sigma scale on y, TOST-p as
  colour) with p = 0.05 contour overlaid for both anchoring scenarios.
- Paths are relative to the script location so the figure script runs inside
  the public repository without hard-coded local drives.
"""
import os, shutil, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make paths relative to the script's own location so the script runs from
# either the project workspace or the public repository without edits.
HERE = os.path.dirname(os.path.abspath(__file__))
# The repository layout has this script at <repo_root>/analysis/kingston_figures_v9.py
# and figures live at <repo_root>/figures/. Try to locate the repo root by
# walking up to a directory that contains 'github_kingston_vendor_csa' or, if
# unavailable, fall back to a sibling 'figures/' folder next to HERE.
_candidate = HERE
for _ in range(4):
    if os.path.basename(_candidate) == 'github_kingston_vendor_csa':
        REPO_ROOT = _candidate
        break
    _candidate = os.path.dirname(_candidate)
else:
    REPO_ROOT = os.path.dirname(HERE)  # sibling of analysis/
INTERNAL_BASE = os.environ.get(
    'KINGSTON_INTERNAL_DIR',
    'E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2/')
REPO_FIG_DIR = os.path.join(REPO_ROOT, 'figures')
REPO_FIG_SRC = os.path.join(REPO_ROOT, 'figures_v8')  # previous round figures
os.makedirs(REPO_FIG_DIR, exist_ok=True)
OUT = REPO_FIG_DIR + os.sep  # ensure trailing separator for string concatenation

V9 = json.load(open(os.path.join(HERE, '..', 'results', 'reanalysis_v9_results.json'), encoding='utf-8'))
C_UIH, C_GE = '#1f77b4', '#d62728'
print(f'[kingston_figures_v9] writing figures to: {OUT}')

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

# Level 4: QC (separated horizontally to avoid overlap)
box(0.2, 5.85, 5.2, 1.05,
    'Automated QC passed: n = 213 (UIH 82 / GE 131)\n'
    'per-vendor mean CSA in 30\u2013120 mm\u00b2 range;\n\u2265100 analysed cord slices',
    fs=8.0)
box(7.6, 5.85, 4.4, 1.05,
    'Automated QC failed: n = 87 (UIH 8 / GE 79)\n'
    'algorithmic CSA outside range or\nsegmentation under-performance',
    fs=8.0, fc='#fff5f5', ec='#8b0000')
note_left(4.85, 'QC-failed examinations were not screened\nfor degenerative compression; the\ntipping-point sensitivity analysis (Fig. 8)\naddresses this stratum.', fs=7.4)

# QC-passed continues into review; QC-failed branches to right-hand note
arrow_v(2.8, 5.85, 5.1)

# Level 5: DCM review with mutually-exclusive categories
box(BX, 3.55, BW, 1.55,
    'Neuroradiological review of source images\n'
    '(single reader, 13 y cervical-MRI experience, blinded to vendor and CSA)\n'
    '213 \u2192 146 (excluded 67 by mutually-exclusive categories:\n'
    'degenerative cord compression / T2 signal change 48;\n'
    'cord atrophy / congenital anomaly 7;\n'
    'non-cervical / wrong-region series 2;\n'
    'earlier-review overlap reassigned to its primary reason 10)',
    fs=7.8, fc='#fff8e1', ec='#b8860b')
arrow_v(5.6, 3.55, 2.85)

# Level 6: 1.5T
box(BX, 1.95, BW, 0.9, '1.5 T analytic cohort\nn = 132   (UIH 51 / GE 81)', fs=9.5, bold=True)
note_right(2.4, 'GE 3.0 T (DISCOVERY MR750)\nexcluded after review: n = 14\n(all QC-passed, all GE)')
arrow_v(5.6, 1.95, 1.2)

box(BX, 0.25, BW, 0.9, 'C2\u2013C5 complete-case population\nn = 127   (UIH 51 / GE 76)', fs=9.5, bold=True)

ax.set_title('Participant and examination flow (STROBE, vendor-stratified)',
             fontsize=12.5, fontweight='bold', y=0.995)
fig.tight_layout()
import os
os.makedirs(OUT, exist_ok=True)
_p5 = OUT + 'fig5_strobe_flow_vendor.png'
_p5p = OUT + 'fig5_strobe_flow_vendor.pdf'
fig.savefig(_p5, dpi=300, bbox_inches='tight')
fig.savefig(_p5p, bbox_inches='tight')
print('saved Fig.5:', os.path.exists(_p5), os.path.getsize(_p5) if os.path.exists(_p5) else 'NA')
plt.close(fig)
print('Fig.5 redrawn: QC boxes separated; mutually-exclusive exclusion categories; '
      'no overlap; consistent with body-text narrative')

# ================= Fig 8: tipping 2D (TRUE 2D heatmap, delta x sigma scale) =================
grid = V9['tipping_2d_v9']
deltas = np.array(grid['GE_sigma1.0']['deltas'])
sigma_levels = [1.0, 1.25, 1.5]
panels = [('GE', 'A. GE-anchored scenario'), ('symmetric', 'B. Symmetric shift scenario')]

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
for ax, (anchor, title) in zip(axes, panels):
    # Build a 2D matrix: rows = sigma scale, cols = delta
    Z = np.zeros((len(sigma_levels), len(deltas)))
    for i, sc in enumerate(sigma_levels):
        Z[i, :] = np.array(grid[f'{anchor}_sigma{sc}']['tost_p'])
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=0.25,
                   extent=[deltas.min(), deltas.max(), 0, len(sigma_levels)])
    # y-axis ticks as sigma scale
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels([f'\u00d7{s:.2f}' for s in sigma_levels], fontsize=8)
    ax.set_ylabel('SD scale factor (\u03c3)', fontsize=9)
    # overlay p=0.05 contour
    cs = ax.contour(deltas, [0.5, 1.5, 2.5], Z, levels=[0.05], colors='k', linewidths=1.6)
    # tipping values (annotate near the contour intersection)
    for i, sc in enumerate(sigma_levels):
        tip_v = V9['tipping_point_v9']['scenarios']['qc_plus_processing_8_83']['grid_1d'][f'{anchor}_sigma{sc}']['upper_delta']
        if tip_v is not None:
            ax.plot(tip_v, i + 0.5, 'ko', mfc='yellow', mec='black', ms=8, zorder=5)
            ax.annotate(f'+{tip_v:.1f}', xy=(tip_v, i + 0.5),
                        xytext=(tip_v + 1.2, i + 0.5),
                        fontsize=8.5, color='black', va='center')
    ax.set_xlabel('Assumed vendor difference in missing stratum,\nUIH \u2212 GE (mm\u00b2)', fontsize=9)
    ax.set_title(title, fontsize=10.5)
    # observed base diff
    base_d = V9['tipping_point_v9']['d_observed_base']
    ax.axvline(base_d, color='#1f77b4', lw=1.4, ls='-.', alpha=0.85, zorder=4)
    ax.annotate('observed\nbase diff', xy=(base_d, 2.6), fontsize=7.5,
                color='#1f77b4', ha='center')
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label('TOST p value (\u00b15 mm\u00b2)', fontsize=9)
fig.suptitle('Conditional tipping-point sensitivity (two-dimensional heatmap, delta \u00d7 SD scale). '
             'Yellow dots: tipping thresholds for the qc_plus_processing_8_83 scenario.',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
_p8 = OUT + 'fig8_tipping_2d.png'
_p8p = OUT + 'fig8_tipping_2d.pdf'
fig.savefig(_p8, dpi=300, bbox_inches='tight')
fig.savefig(_p8p, bbox_inches='tight')
print('saved Fig.8:', os.path.exists(_p8), os.path.getsize(_p8) if os.path.exists(_p8) else 'NA')
plt.close(fig)
print('Fig.8 redrawn as true 2D heatmap (delta x sigma) with p=0.05 contour and tipping dots')

# ================= Fig 4: equivalence dual endpoint (update footnote values) =================
me = V9['main_endpoint']['complete_case']
wcx = V9['whole_cord']
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
_p4 = OUT + 'fig4_equivalence_dual_endpoint.png'
_p4p = OUT + 'fig4_equivalence_dual_endpoint.pdf'
fig.savefig(_p4, dpi=300)
fig.savefig(_p4p)
print('saved Fig.4:', os.path.exists(_p4), os.path.getsize(_p4) if os.path.exists(_p4) else 'NA')
plt.close(fig)
print('Fig.4 regenerated with v9 values')
print('DONE figures_v9 ->', OUT)
