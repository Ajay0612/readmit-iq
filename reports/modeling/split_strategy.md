# Phase 2 patient-grouped split design

Historical design, now implemented in Phase 3. See the
[frozen split report](data_split_report.md) and [manifest](split_manifest.json) for actual assignments
and diagnostics. Statements below describe the Phase 2 state before partition creation.

Recommend **70% train / 15% validation / 15% test by unique patient**, using seed **42**
from `configs/config.yaml`. Every eligible encounter for a patient follows that patient's
assignment. No partition IDs or train/validation/test datasets have been created in Phase 2.

The primary cohort has 65,044 patients and 90,702 encounters: approximately 45.5k / 9.8k / 9.8k
patients per partition. Encounter counts and prevalence will only be approximately balanced
because group sizes and within-patient outcomes differ.

## Repeated encounters

| Eligible encounters per patient | Patients | Encounters |
|---|---:|---:|
| 1 | 50,481 | 50,481 |
| 2 | 9,238 | 18,476 |
| 3 | 2,868 | 8,604 |
| 4+ | 2,457 | 13,141 |

14,563 patients have repeated eligible encounters, accounting for **44.34%** of encounters.
The maximum is 39 eligible encounters per patient. The observed encounter-level positive rate
is 19.51% for repeat patients versus 4.37% for single-encounter patients. This is associated
with dataset coverage and future encounters; it is not a known-at-discharge predictor.

The raw source has 54,745 / 10,434 / 3,328 / 3,011 patients with 1 / 2 / 3 / 4+ encounters,
respectively. Repeat patients contribute 46.21% of raw encounters.

Under hypothetical independent 70/15/15 encounter allocation:

- Expected patients spanning partitions: **8,180.07** in the eligible cohort.
- A randomly selected test encounter has a **37.21%** probability of its patient also appearing
  in training. The corresponding raw-data figure is 38.98%.

For patient size `k`, the probability of spanning partitions is
`1 - (0.70**k + 0.15**k + 0.15**k)`. Conditional on one encounter being in test, the chance
of at least one other encounter being in training is `1 - 0.30**(k-1)`. Weight the latter by
`k` and divide by all encounters for the test-encounter probability. These are expectations
under independent allocation, not an actual split, a fixed-size simulation, or metric inflation.

## Proposed Phase 3 procedure

1. Freeze the cohort policy, scoring time, source checksum, seed and feature exclusions.
2. Construct an allocation-only table with one row per patient, eligible encounter count,
   and an indicator of any positive eligible encounter.
3. Use joint strata `any_positive × encounter_count_bucket(1, 2, 3, 4+)`. These are allocation
   metadata, never predictors. If a stratum cannot populate every partition, merge size buckets
   by a documented count-based rule before allocation; do not search seeds for favorable results.
4. Reserve 15% of the **unique-patient table** using stratified selection. From the remaining
   85%, reserve `15/85` for validation. Expand assignments back to encounters. Stratifying the
   patient table preserves groups; `train_test_split` directly on encounter rows does not.
   Balanced patient strata do not guarantee exactly balanced encounter prevalence.
5. Assert pairwise-disjoint patient sets and encounter IDs, complete eligible-row coverage,
   one assignment per encounter, both classes in every partition, and reproducibility. Report
   patient/encounter counts and prevalence. Large discrepancies require a documented redesign
   based only on allocation diagnostics, never model metrics or seed shopping.
6. Save and hash one assignment manifest in ignored data storage, with aggregate diagnostics
   tracked in Git. Freeze the test manifest before fitting any preprocessing or model.
7. Fit encoding, rare-level grouping, scaling, transformations and any resampling on training
   only, inside reusable pipelines. Use group-aware CV within training. Future calibration and
   threshold work must use training out-of-fold predictions or validation, never test labels.
   Do not treat one repeatedly reused validation set as unlimited experimental evidence.
8. Evaluate the frozen final pipeline on test once after choices are fixed. Keep experiments
   away from test and use patient-cluster uncertainty when reporting evaluation.

The proposed allocation strata have adequate observed support; the smallest contains 690
patients. Counts below describe the cohort and do not assign anyone to a partition.

| Encounter-count bucket | No positive encounter | At least one positive encounter |
|---|---:|---:|
| 1 | 48,273 | 2,208 |
| 2 | 6,591 | 2,647 |
| 3 | 1,559 | 1,309 |
| 4+ | 690 | 1,767 |

## Limits

- This estimates encounter-level performance for patients unseen in training. A returning-patient
  deployment population is different and cannot be validated by casually mixing patient IDs.
- No reliable dates or hospital IDs exist. Temporal and hospital external validation are unavailable.
  Encounter ID order cannot substitute for dates.
- Within-patient chronology cannot be reconstructed. Future encounter counts, full-dataset repeat
  flags and patient-wide outcome aggregates are forbidden predictors. Use the provided prior-year
  utilization measures instead.
- **Full-cohort EDA has already examined outcomes.** A future test can be untouched by fitting
  and tuning after freezing, but is not a fully unseen confirmatory dataset. Retain this limitation
  in final reporting; contemporary external/prospective data would provide stronger confirmation.
- Settle mixed rehabilitation and unknown-destination policies before allocating groups; do not
  redefine the cohort repeatedly against test results.

Evidence: `reports/eda/repeat_patient_counts.csv`, `repeat_patient_rates.csv`, `summary.json`,
and the analytic overlap function in `analysis/statistics.py`.
