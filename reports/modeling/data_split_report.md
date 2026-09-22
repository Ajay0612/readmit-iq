# Frozen patient-grouped partitions

| partition | encounters | patients | positives | negatives | prevalence_pct | encounter_pct | patient_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 63563 | 45530 | 7068 | 56495 | 11.120 | 70.079 | 69.999 |
| validation | 13590 | 9757 | 1499 | 12091 | 11.030 | 14.983 | 15.001 |
| test | 13549 | 9757 | 1487 | 12062 | 10.975 | 14.938 | 15.001 |

All three pairwise patient overlaps are **zero**. Every eligible patient and encounter occurs in one partition. Seed 42; outcome × encounter-count patient strata; absolute prevalence tolerance 1 percentage point, fixed before allocation. No seed search. Groups take precedence over exact row proportions.

The test is locked. Its labels were used only for allocation diagnostics; no test features, predictions or performance guide Phase 3. Full-cohort Phase 2 EDA previously viewed outcomes indirectly; this is not a fully unseen confirmatory sample. No timestamps support temporal validation.

Compressed CSV partitions and assignments live in ignored data/processed/phase3. SHA-256 hashes, source/cohort provenance and freeze time are in split_manifest.json. Reproduction must match those hashes. Development loaders reject test access. This is an application guard, not an operating-system access boundary.
