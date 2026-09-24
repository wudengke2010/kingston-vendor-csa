# PACS query logic and de-duplication

This document records the picture archiving and communication system (PACS)
query strategy and the de-duplication procedure for the two retrospective
exports analysed in the manuscript (v8).

## Exports

| Export | Query date | Acquisition window | Age restriction | Examinations |
|---|---|---|---|---|
| 1 | June 2026 | Jan–Jun 2026 | ≥ 50 years | 80 |
| 2 | July 2026 | Jul 2023–Jul 2026 | none | 232 |
| **Total** | | | | **312** |

## Overlap prevention (identifier NOT-IN filter)

Because export 2 spans the acquisition period of export 1, the export 2 PACS
query was executed with an explicit exclusion filter on the patient identifiers
returned by export 1 (identifier `NOT IN` export 1). Patient-level overlap
between the two exports is therefore **zero by construction**, and the unique
examination total is 312 = 80 + 232.

## De-duplication checks

- Within each export, hospital identifiers were cross-checked for internal
  duplicates using the tuple (age, sex, scan date) plus the identifier string.
- Two pairs of examinations shared a patient name but had distinct hospital
  identifiers; clinical details (referring service, acquisition date, body
  region) confirmed these were different individuals. At most one examination
  per pair entered any analysis cohort.
- No other duplicate patient identifiers were found across the 312 examinations.

## Pre-processing exclusions (312 → 304)

| Reason | n |
|---|---|
| Non-UIH/GE system (Philips) | 1 |
| No usable sagittal T2-weighted series | 3 |
| Age < 18 years | 4 |
| **Total** | **8** |

Note: the Philips examination was excluded at this stage, so the 304
examinations entering the pipeline comprise UIH 90 and GE 214 only.

## Verification

The per-examination list underlying this accounting (study ID, export, query
date, exclusion reason where applicable) is retained in the institutional
project record; de-identified per-subject data for the 132 analysed
examinations are in `kingston_15T_per_subject_deidentified.csv`. Raw PACS
query logs and DICOM headers cannot be released publicly for privacy reasons
but are available to auditors on reasonable request.
