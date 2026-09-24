"""
KINGSTON SCT Pipeline v5 — Batch 3: New cases without age limit
Processes 108 cases (>=18yo, <50yo, UIH/GE, cervical T2w sag) from KINGSTON/KINGSTON/DICOM/

Key fix: Uses pre-identified cervical T2w series from rescreen to avoid the 
series selection bug (picking lumbar instead of cervical).
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
logger = logging.getLogger('kingston_pipeline_v5_batch3')

# === Configuration ===
DICOM_ROOT = "F:/bayymri/KINGSTON/KINGSTON/DICOM"
OUTPUT_ROOT = "E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2"
CASES_FILE = "C:/Users/admin/WorkBuddy/2026-07-05-05-44-50/cases_to_process_no_age_limit.json"
CSA_STANDARD_RESOLUTION = 0.5  # mm


def load_cases():
    """Load cases to process from JSON"""
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    logger.info(f"Cases to process: {len(cases)}")
    return cases


def find_dicom_series(pa_dir, target_se_dir):
    """Find the DICOM series path for a specific SE directory."""
    base = os.path.join(DICOM_ROOT, pa_dir, "ST0")
    if not os.path.exists(base):
        return None, None
    
    se_path = os.path.join(base, target_se_dir)
    if not os.path.isdir(se_path):
        # Try case-insensitive match
        for d in os.listdir(base):
            if d.lower() == target_se_dir.lower():
                se_path = os.path.join(base, d)
                break
    
    if not os.path.isdir(se_path):
        return None, None
    
    dcm_files = [f for f in os.listdir(se_path) if not f.endswith('.json')]
    if not dcm_files:
        return None, None
    
    try:
        ds = pydicom.dcmread(os.path.join(se_path, dcm_files[0]), stop_before_pixels=True)
        desc = str(ds.get('SeriesDescription', ''))
        thick = float(ds.get('SliceThickness', 0))
    except:
        desc = ''
        thick = 0
    
    return se_path, {'description': desc, 'n_files': len(dcm_files), 'slice_thickness': thick}


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
    logger.info("KINGSTON SCT Pipeline v5 - Batch 3: No age limit (108 cases)")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    cases = load_cases()

    results = []
    successful = 0
    failed = 0
    skipped = 0
    total = len(cases)

    for i, pinfo in enumerate(cases):
        pa_dir = pinfo['pa_dir']
        patient_id = f"K2_{pa_dir}"
        mfr = 'UIH' if pinfo['vendor'] == 'UIH' else 'GE MEDICAL SYSTEMS'
        age = pinfo['age']
        sex = pinfo['sex']
        vname = pinfo['vendor']
        target_se = pinfo['c_t2w_series']  # e.g. "SE1", "SE2"

        logger.info(f"\n[{i+1}/{total}] {patient_id} ({vname}, {age}yo {sex}) -> {target_se}")

        # Step 1: Find the pre-identified DICOM series
        se_path, se_info = find_dicom_series(pa_dir, target_se)
        if se_path is None:
            logger.warning(f"  Series {target_se} not found for {pa_dir}, skipping")
            skipped += 1
            continue

        logger.info(f"  T2w sag: {target_se} \"{se_info['description']}\" "
                    f"({se_info['n_files']} files, {se_info['slice_thickness']:.1f}mm)")

        # Step 2: Convert DICOM -> NIfTI
        nifti_path = convert_dicom_to_nifti(patient_id, se_path)
        if not nifti_path:
            failed += 1
            continue

        # Step 3: Process patient
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
            with open(os.path.join(OUTPUT_ROOT, "batch3_results_partial.json"), 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"  [Checkpoint] Saved {len(results)} results ({successful} OK, {failed} FAIL)")

    # === Summary ===
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results: {successful} OK, {failed} FAIL, {skipped} SKIP / {total} total")

    with open(os.path.join(OUTPUT_ROOT, "batch3_results.json"), 'w') as f:
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
        summary_df.to_csv(os.path.join(OUTPUT_ROOT, "batch3_csa_summary.csv"), index=False)

        uih = summary_df[summary_df['manufacturer'].str.contains('UIH', na=False)]
        ge = summary_df[summary_df['manufacturer'].str.contains('GE', na=False)]
        logger.info(f"\nUIH: n={len(uih)}, mean CSA={uih['mean_csa'].mean():.2f} +/- {uih['mean_csa'].std():.2f} mm2")
        logger.info(f"GE:  n={len(ge)}, mean CSA={ge['mean_csa'].mean():.2f} +/- {ge['mean_csa'].std():.2f} mm2")

    logger.info("Pipeline v5 Batch 3 complete!")


if __name__ == "__main__":
    main()
