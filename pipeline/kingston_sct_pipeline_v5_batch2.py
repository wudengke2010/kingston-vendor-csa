"""
KINGSTON SCT Pipeline v5 — Batch 2: New 232 patients
Processes eligible new patients (>=50yo + UIH/GE + T2w sagittal) from KINGSTON/KINGSTON/DICOM/

Adapted from kingston_sct_pipeline_v5.py with:
- New DICOM_ROOT: F:/bayymri/KINGSTON/KINGSTON/DICOM
- Patient info read from DICOM headers (no metadata file)
- Eligibility filter from new_data_analysis.json
- Patient IDs prefixed with K2_ to avoid collision with original PA0-PA79
"""

import os
import sys
import json
import glob
import logging
import traceback
import numpy as np
import pandas as pd
import pydicom

# SCT imports
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.deepseg_.sc import (
    deep_segmentation_spinalcord,
    THR_DEEPSEG,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('kingston_pipeline_v5_batch2')

# === Configuration ===
DICOM_ROOT = "F:/bayymri/KINGSTON/KINGSTON/DICOM"
OUTPUT_ROOT = "E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2"
ELIGIBILITY_FILE = "C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/new_data_analysis.json"
CSA_STANDARD_RESOLUTION = 0.5  # mm


def load_eligible_patients():
    """Load eligible patients from analysis JSON"""
    with open(ELIGIBILITY_FILE, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)
    
    eligible = []
    for p in all_patients:
        if not p['is_new']:
            continue
        if p['age_num'] < 50:
            continue
        mfr = p['manufacturer']
        if 'UIH' not in mfr and 'GE' not in mfr:
            continue
        if not p['has_t2w_sag']:
            continue
        eligible.append(p)
    
    logger.info(f"Eligible patients: {len(eligible)} (UIH {sum(1 for p in eligible if 'UIH' in p['manufacturer'])}, "
                f"GE {sum(1 for p in eligible if 'GE' in p['manufacturer'])})")
    return eligible


def scan_dicom_series(pa_dir):
    """Scan actual DICOM directories for a patient."""
    base = os.path.join(DICOM_ROOT, pa_dir, "ST0")
    if not os.path.exists(base):
        return []

    se_dirs = sorted(
        [d for d in os.listdir(base) if d.startswith('SE')],
        key=lambda x: int(x[2:])
    )

    series_info = []
    for se_dir in se_dirs:
        se_path = os.path.join(base, se_dir)
        if not os.path.isdir(se_path):
            continue

        dcm_files = [f for f in os.listdir(se_path) if not f.endswith('.json')]
        if not dcm_files:
            continue

        try:
            ds = pydicom.dcmread(
                os.path.join(se_path, dcm_files[0]),
                stop_before_pixels=True
            )
            desc = str(ds.get('SeriesDescription', ''))
            thick = float(ds.get('SliceThickness', 0))
            mfr = str(ds.get('Manufacturer', ''))
            
            iop = ds.get('ImageOrientationPatient', [0, 0, 0, 0, 0, 0])
            is_sag = False
            if len(iop) >= 6:
                row_x, row_y, row_z = iop[0], iop[1], iop[2]
                col_x, col_y, col_z = iop[3], iop[4], iop[5]
                nx = row_y * col_z - row_z * col_y
                ny = row_z * col_x - row_x * col_z
                nz = row_x * col_y - row_y * col_x
                if abs(nx) > 0.7 and abs(ny) < 0.5 and abs(nz) < 0.5:
                    is_sag = True

            series_info.append({
                'se_dir': se_dir,
                'se_idx': int(se_dir[2:]),
                'description': desc,
                'n_files': len(dcm_files),
                'slice_thickness': thick,
                'manufacturer': mfr,
                'is_sagittal': is_sag,
                'path': se_path,
            })
        except Exception as e:
            continue

    return series_info


def find_t2w_sag_from_dicom(series_list):
    """Find T2w sagittal series from actual DICOM headers."""
    candidates = []
    
    for s in series_list:
        desc = s['description'].lower()
        if 't2' not in desc:
            continue
        exclude_patterns = [
            'loc', 'fgre', 'stir', '3-pl', '3pl', 'water', 'ideal',
            'inphase', 'outphase', 'calib', 'scout', 'easy', 
            'mars', 'tra', 'ax', 'cor', 't1', 'pd'
        ]
        if any(x in desc for x in exclude_patterns):
            continue
        is_sag_desc = 'sag' in desc
        is_sag_orient = s['is_sagittal']
        if not (is_sag_desc or is_sag_orient):
            continue
        if s['slice_thickness'] < 2 or s['slice_thickness'] > 6:
            continue
        if s['n_files'] < 7 or s['n_files'] > 25:
            continue
        candidates.append(s)
    
    if not candidates:
        for s in series_list:
            desc = s['description'].lower()
            if 't2' not in desc:
                continue
            exclude_patterns = [
                'loc', 'fgre', 'stir', '3-pl', '3pl', 'water', 'ideal',
                'inphase', 'outphase', 'calib', 'scout', 'easy', 
                'mars', 'tra', 'ax', 'cor', 't1', 'pd'
            ]
            if any(x in desc for x in exclude_patterns):
                continue
            if 'sag' in desc or s['is_sagittal']:
                if s['n_files'] >= 7:
                    candidates.append(s)
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x['n_files'], reverse=True)
    return candidates[0]


def convert_dicom_to_nifti(patient_id, se_path):
    """Convert DICOM to NIfTI and return the correct 3D volume"""
    import dcm2niix
    nifti_dir = os.path.join(OUTPUT_ROOT, "nifti", patient_id)
    os.makedirs(nifti_dir, exist_ok=True)

    try:
        dcm2niix.main([
            '-o', nifti_dir, '-f', f'{patient_id}_T2w',
            '-z', 'y', '-x', 'y', se_path
        ])
    except Exception as e:
        logger.error(f"  dcm2niix failed: {e}")
        return None

    all_nifti = sorted(glob.glob(os.path.join(nifti_dir, "*.nii.gz")))
    best_path = None
    best_score = -1

    for nf in all_nifti:
        try:
            im = Image(nf)
            dx, dy, dz = im.dim[0], im.dim[1], im.dim[2]
            vx, vy, vz = im.dim[4], im.dim[5], im.dim[6]
            if dx < 100 or dy < 100:
                continue
            score = 0
            if 200 <= dx <= 600 and 200 <= dy <= 600:
                score += 10
            if 7 <= dz <= 25:
                score += 20
            if 2 <= vz <= 6:
                score += 10
            if 0.5 <= vx <= 2 and 0.5 <= vy <= 2:
                score += 5
            if score > best_score:
                best_score = score
                best_path = nf
        except:
            continue

    if best_path is None or best_score < 20:
        logger.error(f"  No valid 3D NIfTI found (best score={best_score})")
        return None

    im = Image(best_path)
    logger.info(f"  NIfTI: {os.path.basename(best_path)} "
                f"{im.dim[0]}x{im.dim[1]}x{im.dim[2]}, "
                f"voxel={im.dim[4]:.2f}x{im.dim[5]:.2f}x{im.dim[6]:.2f}mm, "
                f"orient={im.orientation}")
    return best_path


def process_patient(nifti_path, patient_id, manufacturer, age, sex):
    """Full pipeline: SVM centerline -> segmentation -> CSA at 0.5mm"""

    im_image = Image(nifti_path)
    logger.info(f"  Image: {im_image.dim[0]}x{im_image.dim[1]}x{im_image.dim[2]}, "
                f"orient={im_image.orientation}, "
                f"voxel={im_image.dim[4]:.2f}x{im_image.dim[5]:.2f}x{im_image.dim[6]:.2f}mm")

    for ctr_algo in ['svm', 'cnn']:
        logger.info(f"  Trying centerline: {ctr_algo}")
        try:
            im_seg, im_image_res, im_seg_rpi = deep_segmentation_spinalcord(
                im_image=Image(nifti_path),
                contrast_type='t2',
                ctr_algo=ctr_algo,
                brain_bool=True,
                kernel_size='2d',
                threshold_seg=THR_DEEPSEG['t2'],
                remove_temp_files=1,
                verbose=0
            )
            logger.info(f"  Segmentation succeeded with {ctr_algo}")
            break
        except Exception as e:
            logger.warning(f"  {ctr_algo} failed: {str(e)[:100]}")
            if ctr_algo == 'cnn':
                logger.error(f"  Both SVM and CNN failed, skipping")
                return None
            continue

    seg_dir = os.path.join(OUTPUT_ROOT, "segmentation", patient_id)
    os.makedirs(seg_dir, exist_ok=True)
    seg_file = os.path.join(seg_dir, f"{patient_id}_T2w_seg.nii.gz")
    im_seg.save(seg_file)

    im_seg_rpi = im_seg.copy().change_orientation('RPI')
    im_seg_std = resample_nib(
        im_seg_rpi,
        new_size=[CSA_STANDARD_RESOLUTION, CSA_STANDARD_RESOLUTION, im_seg_rpi.dim[6]],
        new_size_type='mm',
        interpolation='linear',
        preserve_codes=True
    )
    im_seg_std.data = (im_seg_std.data > 0.5).astype(np.uint8)

    voxel_area = CSA_STANDARD_RESOLUTION ** 2
    csa_values = []
    for z in range(im_seg_std.data.shape[2]):
        n_voxels = np.sum(im_seg_std.data[:, :, z] > 0)
        if n_voxels > 0:
            csa_mm2 = n_voxels * voxel_area
            csa_values.append({'slice_z': z, 'CSA_mm2': csa_mm2, 'n_voxels': n_voxels})

    if not csa_values:
        logger.warning("  No slices with spinal cord detected")
        return None

    csa_df = pd.DataFrame(csa_values)
    csa_dir = os.path.join(OUTPUT_ROOT, "csa", patient_id)
    os.makedirs(csa_dir, exist_ok=True)
    csa_file = os.path.join(csa_dir, f"{patient_id}_csa.csv")
    csa_df.to_csv(csa_file, index=False)

    mean_csa = csa_df['CSA_mm2'].mean()
    logger.info(f"  CSA (0.5mm std): mean={mean_csa:.2f}, "
                f"min={csa_df['CSA_mm2'].min():.2f}, "
                f"max={csa_df['CSA_mm2'].max():.2f}, "
                f"slices={len(csa_values)}")

    if mean_csa < 30 or mean_csa > 150:
        logger.warning(f"  CSA suspicious: {mean_csa:.2f} mm^2")

    return {
        'patient_id': patient_id,
        'original_pa': patient_id.replace('K2_', ''),
        'manufacturer': manufacturer,
        'age': age,
        'sex': sex,
        'ctr_algo': ctr_algo,
        'nifti_file': nifti_path,
        'segmentation_file': seg_file,
        'csa_file': csa_file,
        'mean_csa': float(mean_csa),
        'n_slices': len(csa_values),
        'min_csa': float(csa_df['CSA_mm2'].min()),
        'max_csa': float(csa_df['CSA_mm2'].max()),
        'std_csa': float(csa_df['CSA_mm2'].std()),
    }


def main():
    logger.info("=" * 60)
    logger.info("KINGSTON SCT Pipeline v5 — Batch 2: New 232 patients")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    eligible = load_eligible_patients()

    results = []
    successful = 0
    failed = 0
    skipped = 0
    total = len(eligible)

    for i, pinfo in enumerate(eligible):
        pa_dir = pinfo['pa_dir']
        patient_id = f"K2_{pa_dir}"
        mfr = pinfo['manufacturer']
        age = pinfo['age_num']
        sex = pinfo['sex']
        vname = 'UIH' if 'UIH' in mfr else 'GE'

        logger.info(f"\n[{i+1}/{total}] {patient_id} ({vname}, {age}yo {sex})")

        # Step 1: Scan DICOM series
        series_list = scan_dicom_series(pa_dir)
        if not series_list:
            logger.warning(f"  No DICOM series found, skipping")
            skipped += 1
            continue

        # Step 2: Find T2w sagittal
        t2w_series = find_t2w_sag_from_dicom(series_list)
        if t2w_series is None:
            descs = [f"{s['se_dir']}=\"{s['description']}\"" for s in series_list]
            logger.warning(f"  No T2w sagittal found. Available: {', '.join(descs)}")
            skipped += 1
            continue

        logger.info(f"  T2w sag: {t2w_series['se_dir']} \"{t2w_series['description']}\" "
                    f"({t2w_series['n_files']} files, {t2w_series['slice_thickness']:.1f}mm)")

        # Step 3: Convert DICOM -> NIfTI
        nifti_path = convert_dicom_to_nifti(patient_id, t2w_series['path'])
        if not nifti_path:
            failed += 1
            continue

        # Step 4: Process patient
        try:
            result = process_patient(nifti_path, patient_id, mfr, age, sex)
            if result is None:
                failed += 1
                continue
            results.append(result)
            successful += 1
        except Exception as e:
            logger.error(f"  Unexpected error: {e}")
            traceback.print_exc()
            failed += 1
            continue

        # Save intermediate results every 10 patients
        if (successful + failed) % 10 == 0:
            with open(os.path.join(OUTPUT_ROOT, "batch2_results_partial.json"), 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"  [Checkpoint] Saved {len(results)} results")

    # === Summary ===
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results: {successful} OK, {failed} FAIL, {skipped} SKIP / {total} total")

    with open(os.path.join(OUTPUT_ROOT, "batch2_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        summary_rows = []
        for r in results:
            summary_rows.append({
                'patient_id': r['patient_id'],
                'manufacturer': r['manufacturer'],
                'age': r['age'],
                'sex': r['sex'],
                'ctr_algo': r['ctr_algo'],
                'mean_csa': r['mean_csa'],
                'std_csa': r['std_csa'],
                'min_csa': r['min_csa'],
                'max_csa': r['max_csa'],
                'n_slices': r['n_slices'],
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUTPUT_ROOT, "batch2_csa_summary.csv"), index=False)

        uih = summary_df[summary_df['manufacturer'].str.contains('UIH', na=False)]
        ge = summary_df[summary_df['manufacturer'].str.contains('GE', na=False)]
        logger.info(f"\nUIH: n={len(uih)}, mean CSA={uih['mean_csa'].mean():.2f} +/- {uih['mean_csa'].std():.2f} mm2")
        logger.info(f"GE:  n={len(ge)}, mean CSA={ge['mean_csa'].mean():.2f} +/- {ge['mean_csa'].std():.2f} mm2")

        if len(uih) > 1 and len(ge) > 1:
            from scipy import stats
            t, p = stats.ttest_ind(uih['mean_csa'], ge['mean_csa'])
            logger.info(f"t-test: t={t:.3f}, p={p:.4f}")

    logger.info("Pipeline v5 Batch 2 complete!")


if __name__ == "__main__":
    main()
