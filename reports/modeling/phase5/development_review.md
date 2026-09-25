# Phase 5 pretest development review

Review began at `38283dc2b710680904a9e869ccdc951e454f143d`, the pushed Phase 4 state.
The existing 158 tests passed before Phase 5 analysis. The Phase 4 verifier confirmed its
executed 13-cell notebook, seven frozen development pipelines, 63 local MLflow runs,
training/validation cache provenance and unchanged patient partitions. Existing reports,
configuration, feature/scoring contracts and model-selection decisions were reviewed.
Historical Phase 4 verification artifacts retain their original bytes and execution claims.

The primary and challenger reproduce Phase 4 validation AP exactly: 0.1967806996565404
and 0.1987213611362178. Original train/validation/test counts remain 63,563/13,590/13,549;
patient overlap remains zero. The test count is allocation evidence only. No Phase 5 test
record, prediction or outcome analysis occurred during development review or validation.
Earlier Phase 2 full-cohort EDA remains an acknowledged limitation of the broader study.

Original model SHA-256 values:

* Logistic: `ff82996f1f49008dec373655d53cf95a2ce5d940cb7dbc2bd1e25ea6fe447fa8`
* Boosting: `66ac3b086a0092a6c45e27c50888edda10a429550751a3074e8e99f063c8c8b0`

Phase 5's pretest local checks: **196 tests passed**, Ruff check/format passed, dependency
check passed, notebook 05 executed **8 code cells with 8 embedded figures and no errors**.
All eight validation figures were visually reviewed. Three SHAP legacy-colormap deprecation
warnings occurred in tests; these concern unused plotting helpers, not fit or explanation
correctness. Linear and actual-booster permutation explanations reproduce frozen outputs.
The unsupported TreeSHAP adapter is rejected rather than used in the findings.

The 10% encounter capacity was selected as an explicitly assumed portfolio scenario, using
validation tradeoffs after the user accepted that recommendation. It is not measured staffing
or an optimized clinical benefit/cost policy. The explanation, deep-error, operational and
responsible-ML reports are complete before the formal pretest freeze.

The evaluation code/configuration/dependencies and model specification will be bound to
commits by `final_evaluation_protocol.md` and its lock before the explicit one-time scoring
command. No API, dashboard, Docker image or deployment work is part of this review.
