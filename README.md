# kingston-vendor-csa

Reproducibility package for:

> **Comparison of cervical spinal cord cross-sectional area between United Imaging and GE 1.5-T systems under routine clinical protocols**
> Dengke Wu, Yilin Zhu. Manuscript v7 (under internal review), 2026.

Single-institution retrospective cohort comparing cervical spinal cord cross-sectional
area (CSA) measured on routine sagittal T2-weighted MRI acquired on co-installed
United Imaging (uMR 660, n = 51) and GE (SIGNA HDe, n = 81) 1.5-T systems, segmented
automatically with the Spinal Cord Toolbox (v7.3).

## Repository layout

```
analysis/
  reanalysis_v7.py        # all statistics: endpoints, Welch/TOST, MixedLM, matching,
                          # conditional tipping-point sensitivity, margin sensitivity.
                          # Every reported quantity is exported programmatically from the
                          # fitted result objects (no manual transcription).
  figures_v7.py           # generates Figures 1-8 (v7)
pipeline/
  kingston_sct_pipeline_v5_batch2.py   # DICOM -> NIfTI -> SCT segmentation -> CSA (batch 2)
  kingston_sct_pipeline_v5_batch3.py   # batch 3
data/
  kingston_15T_per_subject_deidentified.csv        # n = 132, per-level + whole-cord CSA
  kingston_15T_scan_parameters_deidentified.csv    # Table S1 source (n = 131 with metadata)
results/
  reanalysis_v7_results.json            # full statistical output of reanalysis_v7.py
```

## Software environment

- Python 3.10.11
- Spinal Cord Toolbox 7.3 (segmentation: `sct_deepseg_sc` with `SC_spm_T2_seg` model;
  vertebral labelling and CSA computed against the PAM50 template)
- `dcm2niix` for DICOM → NIfTI conversion (see pipeline scripts for exact invocations)

Locked Python dependencies (`requirements.txt`):

```
pandas==2.3.3
numpy==2.2.6
scipy==1.15.3
statsmodels==0.15.0
scikit-learn==1.7.2
matplotlib==3.10.9
```

## Reproduce

```bash
pip install -r requirements.txt
python analysis/reanalysis_v7.py       # writes results/reanalysis_v7_results.json
python analysis/figures_v7.py          # writes Figures 1-8
```

## Key endpoints (v7, verified)

| Endpoint | Difference (UIH − GE) | 90% CI | Welch p | TOST ±5 mm² |
|---|---|---|---|---|
| C2–C5 mean CSA (complete-case, n = 127) | 1.38 mm² | −0.82 to 3.57 | 0.300 | p = 0.004 (equivalent) |
| Whole-cord mean CSA (n = 132) | 2.25 mm² | 0.27 to 4.23 | 0.062 | p = 0.012 (equivalent) |

Mixed model (ML, random intercept; 770 observations / 130 subjects): vendor β = 2.94
(95% CI 0.33–5.55), p = 0.027; vendor × level interaction LR χ² = 10.27, df = 5, p = 0.068;
joint Wald χ² = 10.36, p = 0.066.

## Data de-identification

Patient identifiers were replaced with sequential study IDs (S001–S132). Age, sex,
vendor, and CSA values are retained as analysed. The study was approved by the
institutional ethics committee; the derived dataset contains no direct identifiers.

## License

Code: MIT. Data: released under CC BY 4.0.
