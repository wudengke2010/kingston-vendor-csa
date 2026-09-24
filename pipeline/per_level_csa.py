"""
Phase 6: Per-level CSA analysis
Uses sct_label_vertebrae to label vertebral bodies, then sct_process_segmentation
to extract per-level CSA at C2/C3/C4/C5.

Runs on the COMBINED cohort (batch1 Moderate-48 + batch2 QC+DCM passed).
"""
import os
import json
import subprocess
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('per_level_csa')

# SCT binary path
SCT_BIN = "C:/Users/admin/.workbuddy/binaries/python/versions/3.10.11/Scripts"
SCT_PYTHON = "C:/Users/admin/.workbuddy/binaries/python/versions/3.10.11/python.exe"

# Paths
BATCH1_ROOT = "E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5"
BATCH2_ROOT = "E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/batch2"
OUTPUT_ROOT = "E:/boshi/spine-generic-multi-subject/results/kingston_sct_output_v5/per_level"


def run_sct_label_vertebrae(nifti_path, seg_path, output_dir, patient_id):
    """Run sct_label_vertebrae to label vertebral bodies"""
    label_file = os.path.join(output_dir, f"{patient_id}_labels.nii.gz")
    
    # Try using the Python API first
    try:
        from spinalcordtoolbox.vertebra import label_vertebrae
        # sct_label_vertebrae -i <nifti> -s <seg> -c t2 -initfile <init> -ofile <label>
        # Need to use subprocess for this as the API is complex
        pass
    except:
        pass
    
    # Use sct_label_vertebrae command
    cmd = [
        os.path.join(SCT_BIN, "sct_label_vertebrae"),
        "-i", nifti_path,
        "-s", seg_path,
        "-c", "t2",
        "-initfile", "",  # May need init label
        "-ofile", label_file,
        "-v", "0",
    ]
    
    # Remove empty initfile if not needed
    cmd = [c for c in cmd if c != ""]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"  sct_label_vertebrae failed: {result.stderr[:200]}")
            return None
        return label_file
    except Exception as e:
        logger.warning(f"  sct_label_vertebrae error: {e}")
        return None


def run_sct_process_segmentation_perlevel(seg_path, label_file, output_csv, patient_id):
    """Run sct_process_segmentation with -vert flag for per-level CSA"""
    cmd = [
        os.path.join(SCT_BIN, "sct_process_segmentation"),
        "-i", seg_path,
        "-vert", "2:5",  # C2 to C5
        "-vertfile", label_file,
        "-o", output_csv,
        "-v", "0",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"  sct_process_segmentation failed: {result.stderr[:200]}")
            return None
        
        # Parse output CSV
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            return df
        return None
    except Exception as e:
        logger.warning(f"  sct_process_segmentation error: {e}")
        return None


def process_patient_perlevel(patient_id, nifti_path, seg_path, vendor, age, sex, output_dir):
    """Process a single patient for per-level CSA"""
    logger.info(f"  Processing {patient_id}...")
    
    # Step 1: Label vertebrae
    label_file = run_sct_label_vertebrae(nifti_path, seg_path, output_dir, patient_id)
    if label_file is None:
        return None
    
    # Step 2: Per-level CSA
    output_csv = os.path.join(output_dir, f"{patient_id}_perlevel_csa.csv")
    df = run_sct_process_segmentation_perlevel(seg_path, label_file, output_csv, patient_id)
    if df is None:
        return None
    
    # Add patient metadata
    df['patient_id'] = patient_id
    df['vendor'] = vendor
    df['age'] = age
    df['sex'] = sex
    
    return df


def main():
    logger.info("=" * 60)
    logger.info("Per-level CSA Analysis (C2/C3/C4/C5)")
    logger.info("=" * 60)
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Load combined cohort
    combined_file = os.path.join(BATCH2_ROOT, "combined_cohort_summary.csv")
    if not os.path.exists(combined_file):
        logger.error("combined_cohort_summary.csv not found. Run QC analysis first.")
        return
    
    cohort = pd.read_csv(combined_file)
    logger.info(f"Combined cohort: {len(cohort)} patients")
    
    all_results = []
    
    for idx, row in cohort.iterrows():
        patient_id = row['patient_id']
        vendor = row['vendor']
        age = row.get('age', 0)
        sex = row.get('sex', '')
        
        # Find nifti and segmentation files
        if patient_id.startswith('K2_'):
            # Batch 2
            nifti_dir = os.path.join(BATCH2_ROOT, "nifti", patient_id)
            seg_dir = os.path.join(BATCH2_ROOT, "segmentation", patient_id)
        else:
            # Batch 1
            nifti_dir = os.path.join(BATCH1_ROOT, "nifti", patient_id)
            seg_dir = os.path.join(BATCH1_ROOT, "segmentation", patient_id)
        
        nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')] if os.path.exists(nifti_dir) else []
        seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')] if os.path.exists(seg_dir) else []
        
        if not nifti_files or not seg_files:
            logger.warning(f"  {patient_id}: Missing nifti or segmentation")
            continue
        
        nifti_path = os.path.join(nifti_dir, nifti_files[0])
        seg_path = os.path.join(seg_dir, seg_files[0])
        
        result = process_patient_perlevel(
            patient_id, nifti_path, seg_path, 
            vendor, age, sex, OUTPUT_ROOT
        )
        
        if result is not None:
            all_results.append(result)
            logger.info(f"  {patient_id}: per-level CSA extracted")
        else:
            logger.warning(f"  {patient_id}: per-level CSA failed")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(os.path.join(OUTPUT_ROOT, "perlevel_csa_all.csv"), index=False)
        logger.info(f"\nSaved: perlevel_csa_all.csv ({len(combined_df)} rows)")
        
        # Summary by level and vendor
        for vert in [2, 3, 4, 5]:
            level_df = combined_df[combined_df['VertLevel'] == vert]
            for v in ['UIH', 'GE']:
                v_df = level_df[level_df['vendor'] == v]
                if len(v_df) > 0:
                    logger.info(f"  C{vert} {v}: n={len(v_df)}, "
                               f"CSA={v_df['CSA'].mean():.2f} +/- {v_df['CSA'].std():.2f} mm2")
    
    logger.info("Per-level CSA analysis complete!")


if __name__ == "__main__":
    main()
