# kingston-vendor-csa

Reproducibility package for:

> **Single-centre retrospective comparison of cervical spinal cord cross-sectional area measured on co-installed United Imaging and GE 1.5-T MRI systems under routine clinical protocols** (Kingston cohort).
> Dengke Wu, Yilin Zhu. Manuscript v9 (post-sixth-round internal review, 2026-09-24).

This is an **exploratory / post-hoc** retrospective technical comparison of two co-installed
1.5-T MRI systems in the same institution. It is **not** a manufacturer equivalence
validation study; the equivalence analyses were positioned as exploratory after the
initial whole-cord result became known.

## Cohort at a glance

| Field                           | Value                                                 |
|---------------------------------|-------------------------------------------------------|
| PACS exports                    | 80 (Jan–Jun 2026, ≥50 y) + 232 (Jul 2023–Jul 2026)    |
| Pre-processing exclusions       | 8 (1 Philips; 3 no sagittal T2; 4 <18 y)               |
| Eligible entering pipeline      | 304 (UIH 90 / GE 214)                                 |
| SCT processing failures         | 4 (all GE)                                            |
| Processed                       | 300 (UIH 90 / GE 210)                                 |
| QC-failed                       | 87 (UIH 8 / GE 79)                                    |
| QC-passed → neuroradiology review | 213                                                  |
| Excluded at DCM review          | 67 (mutually exclusive categories; see manuscript)     |
| DCM-review retained              | 146                                                   |
| 1.5 T analytic cohort           | 132 (UIH 51 / GE 81; 14 GE 3.0 T excluded)            |

## Repository layout

```
analysis/
  kingston_reanalysis_v9.py    # all statistics (Tables 1/2; Welch/MWU/ANCOVA; MixedLM;
                              # tipping-point with 8/79 and 8/83 missing scenarios;
                              # age matching; tertiles; sex-adjusted sensitivity).
                              # Runs on the PUBLIC data alone; internal-only sections
                              # degrade gracefully when KINGSTON_INTERNAL_DIR is unset.
  kingston_figures_v9.py      # Figures 4/5/8 regenerated with v9 values;
                              # Fig.5 boxes are non-overlapping; Fig.8 is a true 2D
                              # heatmap (delta × SD scale) with p = 0.05 contour.
  kingston_reanalysis_v7.py   # superseded (v7) — kept for version history
  kingston_figures_v7.py      # superseded (v7)
  kingston_reanalysis_v8.py   # superseded (v8) — kept for version history
  kingston_figures_v8.py      # superseded (v8)
pipeline/
  cohort_construction.py      # builds the de-identified public CSV from the
                              # internal final_cohort_15T.csv (mapping internal
                              # patient_id to synthetic study_id S001..S132).
  kingston_sct_pipeline_v5_batch2.py   # DICOM → NIfTI → SCT segmentation → CSA (batch 2)
  kingston_sct_pipeline_v5_batch3.py   # batch 3 (pre-identified target series)
  per_level_csa.py                     # sct_label_vertebrae + sct_process_segmentation
data/
  kingston_15T_per_subject_deidentified.csv   # 132 rows, 12 columns (study_id, vendor,
                                             # age, sex, whole_cord_mean_csa_mm2,
                                             # c2c5_mean_csa_mm2, C2..C7)
  kingston_15T_scan_parameters_deidentified.csv  # 131 rows + acq_matrix/fov_mm +
                                                # *_source columns
  PACS_query_deduplication.md
results/
  reanalysis_v9_results.json   # machine-readable output of kingston_reanalysis_v9.py
  reanalysis_v8_results.json   # superseded
  reanalysis_v7_results.json   # superseded
  sensitivity_cohorts.csv      # machine-readable whole-cord stats at n=48, 132, 146, 213
  STROBE_checklist.md          # STROBE statement mapping
figures/
  fig1..8.{png,pdf}            # all 8 figures from manuscript v9
LICENSE
requirements.txt
README.md
```

## Reproducing the analysis

```bash
# 1. Install the locked dependencies (recorded in requirements.txt)
pip install -r requirements.txt

# 2. Re-generate the de-identified public CSV from the internal cohort
#    (only needed if you have the internal data; the public CSV is already
#    committed and reproducible without this step).
KINGSTON_INTERNAL_DIR=/path/to/internal/batch2/ python pipeline/cohort_construction.py

# 3. Re-run the entire statistical reanalysis (Table 1/2 + MixedLM + tipping +
#    age matching + tertiles + sex-adjusted sensitivity). Runs on PUBLIC data
#    alone; without KINGSTON_INTERNAL_DIR the cohort-flow sub-sections degrade
#    to the last fully-verified numerics recorded in reanalysis_v9_results.json.
python analysis/kingston_reanalysis_v9.py
# → writes results/reanalysis_v9_results.json

# 4. Regenerate all 8 figures from the JSON
python analysis/kingston_figures_v9.py
# → writes figures/fig{1..8}.{png,pdf}
```

The dependency versions recorded in `requirements.txt` are the actual installed
versions used in the v9 analysis (statsmodels, scipy, numpy, scikit-learn,
matplotlib). The reanalysis script reads the running versions dynamically into
`R['software']` for cross-checking.

## Version history

- **v1–v6**: progressive tightening of cohort flow, tipping-point, and language.
- **v7** (commit 6d2ee18, 2026-09-23): Table 2 per-level CIs recomputed
  programmatically; C2–C5 Mann–Whitney p corrected to 0.359; symmetric MNAR
  reparameterised (δ = assumed actual fail-stratum vendor difference).
- **v8** (commit f64a26d, 2026-09-24, fourth-round review response): the
  reanalysis JSON was regenerated with the corrected δ interpretation, the
  statistical audit package and combined sent-review document were rebuilt.
- **v9** (this commit, 2026-09-24, sixth-round review response): adds a
  separate worst-case missing scenario (UIH 8 + GE 83, including the 4 GE SCT
  processing failures identified by the reviewer); adds sex-adjusted MixedLM;
  adds covariance-matrix export (`fixed_effect_covariance`); positions the
  manuscript as exploratory / post-hoc equivalence (margin was selected after
  the initial whole-cord result); rebuilds Fig.5 (QC-passed / QC-failed
  boxes non-overlapping; mutually exclusive exclusion categories) and Fig.8
  (true 2D heatmap with p = 0.05 contour); makes the figure and analysis
  scripts path-relative so they run from inside this repository without
  hard-coded local drives; refactors the reanalysis to be public-data-only
  by default with optional internal-data mode via `KINGSTON_INTERNAL_DIR`.

## Reproducibility caveats

The 4 GE SCT processing failures are acknowledged as a stratum but the specific
PA identifiers cannot be re-derived from the current pipeline logs (only an
aggregate count is preserved). The missing-data sensitivity analysis reports
both scenarios (UIH 8 + GE 79 QC-failed only; UIH 8 + GE 83 worst-case).

The C6/C7 difference (the only per-level difference that survives Holm) relies
on the SCT auto-labeling convention. A manual spot-check by a blinded
radiologist is pending; not added to the public repository until completed.

The whole-cord endpoint is **not** angle-corrected; the per-level C2–C7
endpoint is angle-corrected by `sct_process_segmentation`. These describe
slightly different physical quantities; absolute CSA values should not be
directly compared between them. The single-centre design does not allow
separation of the vendor effect from the protocol effect.