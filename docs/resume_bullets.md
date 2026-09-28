# ReadmitIQ resume bullets

Use the version that matches the role. All results describe a retrospective portfolio project;
the application is a local demonstration, not a deployed hospital service.

## Data Scientist

- Built a patient-grouped readmission study on 90,702 eligible encounters, selecting logistic
  regression through model comparison, paired bootstrap uncertainty and calibration assessment;
  achieved held-out AP 0.208 and ROC-AUC 0.652.
- Evaluated an assumed top-10% outreach scenario that surfaced 23.54% of recorded readmissions
  at 2.36× lift over random targeting; translated results into workload and precision tradeoffs.
- Used SHAP and error analysis to explain predictive drivers and expose a key limitation:
  7.6% recall without prior inpatient use versus 39.6% with prior use.

## Machine Learning Engineer

- Packaged a frozen scikit-learn pipeline behind a versioned FastAPI service with strict
  ten-feature validation, cohort eligibility checks, startup hash verification and deterministic
  batch prioritization.
- Built a Streamlit-to-API demonstration and separate non-root Docker containers with a
  read-only model mount, structured logs and pinned, isolated runtime environments.
- Verified 296 local tests at Phase 6, including real-artifact integration and exact synthetic
  prediction parity; configured GitHub Actions for contract tests, Docker builds and artifact rejection.

## Data Analyst / Business Analyst

- Analyzed 90,702 eligible encounters from 65,044 patients to examine readmission patterns,
  missingness, prior utilization and discharge context using reproducible Python notebooks.
- Quantified an assumed 10% outreach scenario: approximately 258 recorded readmissions surfaced
  per 10,000 eligible discharges versus 110 with random targeting, without claiming prevention or savings.
- Communicated risk segmentation, targeting yield and low-utilization failure modes through
  annotated charts, a documented case study and a synthetic interactive demonstration.

Evidence: [final test report](../reports/modeling/final_test_report.md),
[scenario calculation](../reports/modeling/phase5/test/business_scenario.json),
[Phase 6 verification](../reports/serving/phase6_verification.md).
