# Dataset selection and intended use

Decision recorded September 21, 2026. ReadmitIQ will investigate prioritizing eligible
patients for care-management outreach at discharge. It is an analytical decision-support
portfolio project, not a clinical diagnostic tool.

## Public-data candidates

| Candidate | Fit to the problem | Practical constraint | Decision |
|---|---|---|---|
| [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | Real encounters with explicit readmission categories and patient keys | Historical diabetes cohort; limited outcome visibility and no usable dates | Selected for reproducible, open portfolio work |
| [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) | Detailed hospital records; readmission labels could be derived from admissions | Credentialing, required training and signed data-use agreement; access is not immediately open | Strong future option if authorized access and a more involved extraction are available |
| [AHRQ HCUP Nationwide Readmissions Database](https://hcup-us.ahrq.gov/nrdoverview.jsp) | Designed for readmission analyses with linked hospital stays | Encounter-level database requires purchase; public HCUPnet summaries do not replace individual records | Unsuitable for the requested immediate, openly reproducible download |
| [Synthea](https://synthetichealth.github.io/synthea/) | Freely generated synthetic longitudinal patient histories | Outcomes reflect simulator assumptions rather than observed real-world readmissions | Useful for later demo/test inputs, unsuitable as the principal evidence for model effectiveness |

Selection is a practical project judgment, not a claim that UCI is the most clinically
representative dataset. Its original release supplies 101,766 encounters and 50 columns:
47 potential predictors, two identifiers and one outcome. The downloaded files match that
inventory. A clone can retrieve the same release without credentials.

## Source and reproducibility

- Creators: John Clore, Krzysztof Cios, Jon DeShazo and Beata Strack (2014).
- Dataset DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated by UCI.
- Source period: 1999–2008; 130 US hospitals and integrated delivery networks.
- Original files: `diabetic_data.csv` and `IDS_mapping.csv` inside the UCI ZIP.
- Raw bytes stay in ignored `data/raw/`. SHA-256 pins are in `configs/config.yaml`;
  verified sizes, hashes, URLs and first verification time are in
  `reports/data_quality/source_manifest.json`.
- Descriptions come from UCI metadata and the
  [original study by Strack et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/).
  The API metadata is downloaded separately and its observed checksum is recorded.

## Cohort, label and prediction time

The source selects inpatient encounters with a diabetes diagnosis, a stay of 1–14 days,
laboratory testing and medication administration. It is not a sample of all hospital discharges.
The target is the source `readmitted` category. For the proposed binary task, `<30` is positive;
`>30` and `NO` are negative. `NO` denotes no record of readmission. UCI describes `<30` as less
than 30 days: do not silently redefine it as an independently verified inclusive 30-day label.
Exact dates and the day-30 boundary cannot be reconstructed from the public extract.

The proposed decision point is discharge. Counts and medication changes recorded across the
stay would not all be available at admission. A later feature audit must verify timing against
that discharge decision. Patient and encounter IDs are retained for audit/grouping only.

## Limits and ethical considerations

- Historical care patterns and a selected diabetes cohort limit generalization to present-day
  US hospitals and patients with other conditions. No national representativeness is claimed.
- Recorded outcomes cannot establish complete follow-up or capture all readmissions outside
  contributing systems. Planned/unplanned and preventable readmissions are not distinguished
  by the supplied target. Capturing a label does not establish an avoidable readmission.
- Multiple encounters per patient make independent row splitting inappropriate for evaluating
  generalization to unseen patients. A patient-grouped strategy needs design in a later phase.
- No admission/discharge timestamps or hospital identifier appear in the raw columns. Neither
  an encounter-ID chronology nor a hospital holdout should be invented.
- Death and hospice dispositions require explicit eligibility review. The original study used
  a filtered one-encounter-per-patient cohort of 69,984; that is not this unfiltered raw file.
- Demographics, missingness and care access can encode inequities. Retain source categories
  for later subgroup audits; do not infer sensitive traits or claim fairness before evaluation.
- The medication field name `citoglipton` is preserved exactly as released. The article's
  medication terminology differs; no unverified renaming or drug identity inference is made.
- All eventual feature associations are observational. Any intervention impact or savings
  would require explicit assumptions and separate causal evidence.
- Real deployment would need contemporary external validation, workflow review, outcome
  ascertainment and governance beyond this portfolio project. No individual care decisions
  or real patient records are being supplied by this repository.

## Phase 1 decisions

Preserve raw source tokens and records. Quantify `?`, administrative unknowns and unmeasured
labs separately. Make no imputation, outlier-removal, cohort-filtering or resampling choice
before reviewing the raw findings. EDA, feature engineering and modeling remain future work.
