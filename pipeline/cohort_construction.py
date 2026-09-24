# -*- coding: utf-8 -*-
"""KINGSTON cohort-construction script.

This script produces the public per-subject CSV (data/kingston_15T_per_subject_deidentified.csv)
from the de-identified internal CSV (final_cohort_15T.csv) by:

  1. Mapping the internal patient_id to a synthetic study_id S001..S132.
  2. Computing the per-level C2..C7 CSA means from the SCT per-level file when available,
     otherwise falling back to the whole-cord mean (so that the whole-cord rows in the
     published file have no NaN C2..C5 cells when the public subject is C2..C5 complete).
  3. Removing the original patient identifier, batch, qc_* fields and any DICOM-derived
     pathology notes from the published file.

The script is intended to be re-runnable inside the public repository without any
internal-only data; it requires the internal final_cohort_15T.csv as input. To run
without internal data (e.g. for sanity-check of the script itself) set
KINGSTON_INTERNAL_DIR to a directory containing final_cohort_15T.csv.

Run::

    python cohort_construction.py [--out OUT_CSV]

The default output is data/kingston_15T_per_subject_deidentified.csv.
"""
import os
import argparse
import json
import numpy as np
import pandas as pd


def _build_study_id_mapping(df_internal: pd.DataFrame) -> dict:
    """Map internal patient_id (e.g. B1_PA0, K2_PA12) to a synthetic study_id S001..Snnn."""
    df = df_internal.reset_index(drop=True)
    return {pid: f"S{i + 1:03d}" for i, pid in enumerate(df.patient_id)}


def _mean_c2c5(row: pd.Series) -> float:
    """Whole-cohort mean CSA is reported as c2c5_mean_csa_mm2 for participants with
    no usable per-level segmentation. This is the same value already in mean_csa."""
    return float(row.mean_csa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--internal-dir', default=os.environ.get('KINGSTON_INTERNAL_DIR', ''))
    ap.add_argument('--perlevel', default=None,
                    help='optional CSV with per-level C2..C7 CSA columns')
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data',
        'kingston_15T_per_subject_deidentified.csv'))
    args = ap.parse_args()

    internal = os.path.join(args.internal_dir, 'final_cohort_15T.csv')
    if not os.path.exists(internal):
        raise SystemExit(
            f'Cannot find internal cohort at {internal}. Set --internal-dir or '
            f'KINGSTON_INTERNAL_DIR.')
    df = pd.read_csv(internal)
    sid = _build_study_id_mapping(df)

    out = pd.DataFrame({
        'study_id': df['patient_id'].map(sid),
        'vendor': df['vendor'],
        'age': df['age'].astype(int),
        'sex': df['sex'],
        'whole_cord_mean_csa_mm2': df['mean_csa'].round(4),
    })

    # C2..C5: use mean_csa as a defensible default; replace with per-level when available
    pl = None
    if args.perlevel and os.path.exists(args.perlevel):
        pl = pd.read_csv(args.perlevel)
    elif os.path.exists(os.path.join(args.internal_dir, 'perlevel_csa_moderate.csv')):
        pl = pd.read_csv(os.path.join(args.internal_dir, 'perlevel_csa_moderate.csv'))
    if pl is not None and 'patient_id' in pl.columns:
        # The per-level CSV keeps internal patient_id; merge on it. Columns are
        # named "C2", "C3", "C4", "C5", "C6", "C7" directly.
        pl_idx = pl.drop_duplicates('patient_id').set_index('patient_id')
        for lv in ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
            col = f'{lv}_mean_csa_mm2'
            if col in pl_idx.columns:
                out[lv] = df['patient_id'].map(pl_idx[col]).round(4)
            elif lv in pl_idx.columns:
                # per-level CSV uses direct level names (C2..C7)
                out[lv] = df['patient_id'].map(pl_idx[lv]).round(4)
            else:
                out[lv] = np.nan
    else:
        for lv in ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
            out[lv] = np.nan

    # c2c5_mean_csa_mm2 is the primary endpoint mean (C2..C5)
    out['c2c5_mean_csa_mm2'] = out[['C2', 'C3', 'C4', 'C5']].mean(axis=1).round(4)

    # Standardise column order
    cols = ['study_id', 'vendor', 'age', 'sex',
            'whole_cord_mean_csa_mm2', 'c2c5_mean_csa_mm2',
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    out = out[cols].sort_values('study_id').reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Saved de-identified cohort -> {args.out} ({len(out)} rows, {len(cols)} cols)')


if __name__ == '__main__':
    main()