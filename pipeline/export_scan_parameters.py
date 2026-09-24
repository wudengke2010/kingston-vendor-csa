# -*- coding: utf-8 -*-
"""Export the de-identified per-subject scan-parameter table (Table S1 source).

Reads the internal DICOM-derived parameter extraction
(supplementary_scan_parameters.csv) and the final 1.5-T cohort
(final_cohort_15T.csv) from the directory given by KINGSTON_INTERNAL_DIR
(or --internal-dir), and writes data/kingston_15T_scan_parameters_deidentified.csv.

Field definitions and sources (per row):
  - TR_ms, TE_ms, ETL, flip_deg, slice_thick_mm, spacing_mm:
        copied from the DICOM header extraction (no imputation).
  - acq_matrix: "<rows>x<cols>" built from the DICOM rows/cols header fields.
  - fov_mm: GE systems report the field of view in the header (copied directly);
        UIH headers do not report FOV in mm, so it is DERIVED per examination as
        rows x pixel_row_mm (equivalently cols x pixel_col_mm) and flagged in
        fov_mm_source. No field is imputed from protocol records.
  - bandwidth_Hzpx: DICOM receive bandwidth (Hz/pixel).
  - inter_slice_gap_mm: spacing_mm - slice_thick_mm (derived).

One GE 1.5-T examination of the final cohort (B1_PA60 in internal IDs; its
MRI series was lost during a historical PACS re-export, see cohort history)
has no retrievable series and therefore no row in this file: 131/132 rows.

Run:
    python pipeline/export_scan_parameters.py [--internal-dir DIR] [--out CSV]
"""
import os
import argparse
import numpy as np
import pandas as pd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--internal-dir', default=os.environ.get('KINGSTON_INTERNAL_DIR', ''))
    ap.add_argument('--out', default=os.path.join(
        here, '..', 'data', 'kingston_15T_scan_parameters_deidentified.csv'))
    args = ap.parse_args()

    cohort_csv = os.path.join(args.internal_dir, 'final_cohort_15T.csv')
    params_csv = os.path.join(args.internal_dir, 'supplementary_scan_parameters.csv')
    if not (os.path.exists(cohort_csv) and os.path.exists(params_csv)):
        raise SystemExit(
            'Internal data not found. Set --internal-dir or KINGSTON_INTERNAL_DIR '
            'to a directory containing final_cohort_15T.csv and '
            'supplementary_scan_parameters.csv.')

    wc = pd.read_csv(cohort_csv)
    params = pd.read_csv(params_csv)

    # Same stable, order-based anonymisation as pipeline/cohort_construction.py:
    # study_id S001..S132 are assigned in the row order of the frozen internal
    # final_cohort_15T.csv (this order is part of the frozen analysis and must
    # not change, otherwise study_id references across released files diverge).
    id_map = {pid: f'S{i + 1:03d}' for i, pid in enumerate(wc.patient_id)}

    params['study_id'] = params['patient_id'].map(id_map)
    pm = params[params['study_id'].notna()].drop_duplicates(subset='study_id').copy()
    pm = pm.sort_values('study_id').reset_index(drop=True)

    # --- per-row fields with explicit sources -------------------------------
    pm['acq_matrix'] = (pm['rows'].astype(int).astype(str) + 'x'
                        + pm['cols'].astype(int).astype(str))
    pm['acq_matrix_source'] = 'DICOM header (rows/cols fields)'

    fov = []
    fov_src = []
    for _, r in pm.iterrows():
        if r['vendor'] == 'GE' and pd.notna(r['fov_mm']):
            fov.append(round(float(r['fov_mm']), 1))
            fov_src.append('DICOM header (FOV reported in mm)')
        elif pd.notna(r['rows']) and pd.notna(r['pixel_row_mm']):
            fov.append(round(float(r['rows']) * float(r['pixel_row_mm']), 1))
            fov_src.append('Derived: DICOM rows x pixel spacing (FOV not reported in header)')
        else:
            fov.append(np.nan)
            fov_src.append('Not available (not reported in header; cannot be derived)')
    pm['fov_mm'] = fov
    pm['fov_mm_source'] = fov_src
    pm['inter_slice_gap_mm'] = (pm['spacing_mm'] - pm['slice_thick_mm']).round(2)

    keep = ['study_id', 'vendor', 'manufacturer', 'model', 'field_T',
            'TR_ms', 'TE_ms', 'ETL', 'flip_deg', 'slice_thick_mm', 'spacing_mm',
            'inter_slice_gap_mm', 'acq_matrix', 'acq_matrix_source',
            'fov_mm', 'fov_mm_source', 'bandwidth_Hzpx']
    out = pm[keep].reset_index(drop=True)

    # --- verification --------------------------------------------------------
    subj = pd.read_csv(os.path.join(here, '..', 'data',
                                    'kingston_15T_per_subject_deidentified.csv'))
    chk = out.merge(subj[['study_id', 'vendor']], on='study_id', suffixes=('', '_subj'))
    assert (chk['vendor'] == chk['vendor_subj']).all(), 'vendor mismatch vs per-subject file'
    assert len(out) == 131, f'expected 131 rows, got {len(out)}'
    for c in ['TR_ms', 'TE_ms', 'ETL', 'flip_deg', 'slice_thick_mm', 'spacing_mm',
              'acq_matrix', 'fov_mm', 'bandwidth_Hzpx']:
        n_missing = int(out[c].isna().sum())
        print(f'  {c:18s} non-missing {len(out) - n_missing}/131')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Saved scan-parameter table -> {args.out} ({len(out)} rows, {len(keep)} cols)')


if __name__ == '__main__':
    main()
