# kingston-vendor-csa

Reproducibility package for:

> **Single-centre retrospective comparison of cervical spinal cord cross-sectional area measured on co-installed United Imaging and GE 1.5-T MRI systems under routine clinical protocols**
> Dengke Wu, Yilin Zhu. Manuscript v8 (under internal review), 2026.

Single-institution retrospective cohort comparing cervical spinal cord cross-sectional
area (CSA) measured on routine sagittal T2-weighted MRI acquired on co-installed
United Imaging (uMR 660, n = 51) and GE (SIGNA HDe, n = 81) 1.5-T systems, segmented
automatically with the Spinal Cord Toolbox (v7.3).

## Repository layout

```
analysis/
  kingston_reanalysis_v8.py   # all statistics: Table 1/2, Welch/TOST, Mann-Whitney,
                              # MixedLM (ML, multi-optimizer), tipping-point (corrected
                              # merged variance, delta = actual failed-stratum vendor
                              # difference), age matching, tertiles. Every reported
                              # quantity is exported programmatically (no manual
                              # transcription). Runs on the PUBLIC data alone;
                              # internal-only sections degrade gracefully.
  kingston_figures_v8.py      # Figures 4/5/8 (regenerated with v8 values)
  kingston_reanalysis_v7.py   # superseded (v7) - kept for version history
  kingston_figures_v7.py      # superseded (v7)
pipeline/
  kingston_sct_pipeline_v5_batch2.py   # DICOM -> NIfTI -> SCT segmentation -> CSA (batch 2)
  kingston_sct_pipeline_v5_batch3.py   # batch 3 (pre-identified target series)
  per_level_csa.py                     # sct_label_vertebrae + sct_process_segmentation
                                       # (-vert 2:7, C2-C7); the earlier -initfile bug
                                       # (empty-string argument consuming the next flag)
                                       # was fixed in v8
data/
  kingston_15T_per_subject_deidentified.csv        # n = 132, per-level (C2-C7) + whole-cord CSA
  kingston_15T_scan_parameters_deidentified.csv    # Table S1 source (n = 131 with metadata)
  PACS_query_deduplication.md                      # PACS query logic + de-duplication counts
results/
  reanalysis_v8_results.json     # full statistical output of reanalysis_v8.py
  sensitivity_cohorts.csv        # machine-readable n=48/132/146/213 whole-cord and
                                 # n=146 C6/C7 sensitivity statistics
  STROBE_checklist.md            # STROBE cohort checklist
```

## Software environment

- Python 3.10.11
- Spinal Cord Toolbox 7.3 (segmentation via `deep_segmentation_spinalcord`
  SVM centreline with CNN fallback; vertebral labelling against the PAM50 template)
- `dcm2niix` v1.0.20260416 for DICOM → NIfTI conversion

Locked Python dependencies (match the versions used to produce all v8 numbers):

```
pandas==2.3.3
numpy==2.4.6
scipy==1.18.0
statsmodels==0.14.6
scikit-learn==1.9.0
matplotlib==3.11.0
```

(Note: the v7 manuscript text incorrectly stated statsmodels 0.15.0; the analysis
was actually run with 0.14.6, which is what the locked file above and the JSON
output record.)

## Reproduce

```bash
pip install -r requirements.txt
python analysis/kingston_reanalysis_v8.py   # writes results/reanalysis_v8_results.json
```

This reproduces, from the public per-subject data alone: Table 1, Table 2 (all
per-level Welch 90% CIs, Mann-Whitney and Holm-adjusted values), the main and
whole-cord endpoints with TOST, the MixedLM output (Supplementary Table S2), the
age-matched analysis (Table S3.2), the age tertiles (Table S3.3), and the full
tipping-point grids (Fig. 8 / Methods S4).

Optional: if you have access to the internal (non-public) cohort files, set
`KINGSTON_INTERNAL_DIR` to that directory to additionally regenerate the
cohort-flow ages (DCM-review and 3T exclusion steps) and the 146/213 sensitivity
cohorts; otherwise these summary values are read from `results/sensitivity_cohorts.csv`.

Figure generation requires the local working directory with the raw imaging
outputs and is documented in `analysis/kingston_figures_v8.py` (Figures 1-3, 6-7
are unchanged from v7 and were generated from the same data; Figures 4/5/8 were
redrawn for v8).

## Key endpoints (v8)

| Endpoint | Difference (UIH − GE) | 90% CI | Welch p | TOST ±5 mm² |
|---|---|---|---|---|
| C2–C5 mean (complete case, n = 127) | 1.38 mm² | −0.82 to 3.57 | 0.300 | p = 0.004 (criterion met) |
| Whole-cord mean (n = 132) | 2.25 mm² | 0.27 to 4.23 | 0.062 | p = 0.012 (criterion met) |

Mixed model (770 level observations, 130 participants, ML): vendor effect 2.94 mm²
(p = 0.027); global vendor × level interaction LR χ² = 10.27, df = 5, p = 0.068.

Conditional tipping-point (δ = assumed actual UIH−GE vendor difference in the
QC-failed stratum): GE-anchored +9.5/+8.9/+8.2 mm² (SD × 1.0/1.25/1.5);
symmetric-shift +5.6/+5.3/+5.0 mm².

## Version history

- **v8 (2026-09-24)**: Table 2 confidence intervals recomputed programmatically from
  the public per-subject data (v7 values were manually transcribed and incorrect);
  C2–C5 Mann-Whitney p corrected (0.359, not 0.28); symmetric MNAR scenario
  re-parameterised so δ denotes the assumed actual failed-stratum vendor difference
  in both scenarios; STROBE flow figure corrected (304 = 90 UIH + 214 GE; Philips
  excluded pre-processing; 14 GE 3.0 T excluded after review); DCM-review and 3T
  exclusion ages reported separately (v7 conflated them); statsmodels version
  corrected to 0.14.6; `per_level_csa.py` `-initfile` bug fixed and extended to C2–C7;
  machine-readable sensitivity-cohort output added; README made consistent with the
  actual scripts.

## Licence

Code under the MIT licence; de-identified data under CC BY 4.0.
Raw DICOM data are not publicly available (institutional privacy restrictions).
