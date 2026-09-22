# Cleaning and preprocessing contract for Phase 3

This records the Phase 2 plan. The implemented Phase 3 allowlist and transformations are in the
[scoring contract](scoring_time_contract.md) and [feature manifest](model_feature_manifest.md).

Only fixed eligibility, literal `? → missing` normalization, and the validated binary outcome
have been applied to a copied table. Raw files remain checksum-verified and untouched. Source
outcome and identifiers remain for audit; no learned preprocessing exists.

| Area | Decision and boundary |
|---|---|
| Cohort | Apply `configs/phase2.yaml` through `data/cohort.py`; 90,702 primary encounters. Save exclusion reasons; fail on an unreviewed code. |
| Target | Existing strict `binary_target`: `<30=1`, `>30=0`, `NO=0`. Validate before exclusions; preserve the original field. |
| `?` | Normalize exactly this marker to NA on a copy. Encode categorical NA as Unknown later; no full-data mode imputation. |
| Lab `None` | Preserve as Not measured, distinct from unknown and normal; avoid pandas default NA parsing. |
| Administrative missing codes | Map only documented NULL/not-available/not-mapped/invalid codes to Unknown in the future transformer. Retain raw codes for audit. |
| Weight | Omit from initial predictors for sparse coverage; no fabricated weights or numeric midpoints. Retain for sensitivity. |
| Specialty / payer | Keep explicit Unknown and test ablation if timing or site/access proxies undermine reliability. |
| Race / gender | Preserve unknown/invalid, never infer. Retain for subgroup audits; decide predictive inclusion through a documented development-only review. |
| Categories | Admission codes are categories, not continuous numbers. Fit encoders and any rare-level grouping on training only; handle unseen levels explicitly. |
| Numbers | No missing numeric counts. Keep valid counts and tails; fit any scaling/transform on training only; no arbitrary clipping. |
| IDs | Patient ID only for grouping/audit; encounter ID only for traceability. Exclude both from predictor matrices. |
| Diagnoses | Keep strings/Unknown; verify and version broad mappings; test decimal, short integer, E/V and missing cases before implementation. |
| Constants | Exclude `examide` and `citoglipton` from models; retain source fields and spelling. |
| Timing-sensitive fields | Preserve disposition, diagnosis, lab and full-stay treatment fields with flags. Verify scoring-time availability and compare documented feature sets on development data. |
| Repeats | Keep eligible encounters together by patient; no predictors derived from future rows. |
| Imbalance | 11.08% positive; no resampling or weighting now. Any future resampling is confined to training folds. |

Load the analytical CSV with `keep_default_na=False`, `na_values=['']`, and string dtypes for
IDs and diagnoses. Empty fields encode normalized missing values; lab `None` survives. Regenerating
from verified raw data is preferable to treating a hand-edited analytical CSV as canonical.

Numeric bounds validate source/count constraints, not every clinical plausibility question.
For example, a maximum of 76 emergency visits is unusual but not established as impossible.
Review sparse/ambiguous cases rather than silently correcting them. No target-dependent row
exclusion or learned full-data transformation has been applied.
