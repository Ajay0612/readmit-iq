# ReadmitIQ

Predicting hospital readmission risk to help prioritize limited care-management outreach.

**Business question:** Which patients should a hospital prioritize at discharge for follow-up
when its care-management team cannot contact everyone?

**Current status: Phase 2 complete.** Verified ingestion and raw-data inspection are followed
by an explicit eligible cohort, a 50-variable leakage audit, an executed business-driven EDA
notebook, seven reviewed figures and a patient-grouped split design. No partitions, fitted
preprocessing, models or prediction services exist. There are no performance or impact claims.

This is an analytical decision-support portfolio project, not a clinical diagnostic tool.

## Data understanding

| Measure | Raw source | Eligible cohort |
|---|---:|---:|
| Hospital encounters | 101,766 | 90,702 |
| Unique patients | 71,518 | 65,044 |
| `<30` readmission labels | 11,357 (11.16%) | 10,054 (11.08%) |
| Negative labels (`>30` / `NO`) | 90,409 | 80,648 |
| Patients with multiple encounters | 16,773 | 14,563 |
| Encounters from repeat patients | 46.21% | 44.34% |
| Missing weight | 96.86% | 96.68% |
| Missing medical specialty | 49.08% | 48.28% |
| Missing payer code | 39.56% | 36.48% |

The source has 50 columns (47 candidate features, two IDs and the outcome), with zero duplicate
rows or encounter IDs. Raw bytes remain unchanged. The analytical copy retains every original
column, normalizes literal `?` to missing and adds the validated binary target. Lab `None` means
not measured and is preserved; no mean/mode imputation or learned transformation has been applied.

**Cohort:** remove 11,064 encounters: 1,652 death-coded, 771 hospice, 3,874 continued inpatient
transfers, 87 without confirmed discharge and 4,680 unknown destinations. Hospice requires a
separate care-goal pathway; it is not assumed incapable of readmission. Selected postacute
destinations remain eligible for coordinated outreach. The mixed rehabilitation category and
unknown-destination policy have explicit sensitivity analyses in the
[cohort report](reports/data_quality/cohort_report.md) and reusable rules in
[phase2.yaml](configs/phase2.yaml).

**Observed associations in the eligible cohort:**

- Prior inpatient use: 26.20% readmission for 3+ visits versus 8.34% for none; difference
  **17.85 percentage points** (95% patient-cluster interval 16.44–19.27).
- Prior emergency use: 25.40% for 3+ visits versus 10.34% for none. Outpatient use levels off;
  keep the three counts separate. Exact-count tails are not strictly monotonic.
- Length of stay: 13.44% for 8–14 days versus 9.08% for 1–2 days. Medication-count rates rise
  from 8.61% at 1–9 to 13.00% at 20–29 and level off at 12.79% for 30+.
- Rehabilitation discharges: 27.70% versus 9.30% at home. This is a care-setting association,
  not evidence that changing destination changes risk; mixed rehab settings need workflow review.

![Prior utilization and observed readmission](reports/figures/eda/02_prior_utilization.png)

**Missingness:** initially omit weight (only 3,013 observed eligible values); retain explicit
Unknown for specialty/payer, with no claim that missingness is random. Preserve lab Not measured
separately. All 50 fields, including timing-sensitive diagnoses, medications and disposition,
are reviewed in the [feature audit](reports/data_quality/feature_audit.md).

![Eligible-cohort missingness](reports/figures/eda/07_missingness.png)

**Evaluation design:** repeat patients have a 19.51% encounter readmission rate versus 4.37%
for single-encounter patients. Full-dataset repeat status is future-informed and cannot be a
predictor. Under hypothetical independent encounter allocation, 37.21% of test encounters
would share a patient with training. Plan a **70/15/15 split by patient, seed 42**, with approximate
outcome/group-size stratification; no split has been created. Fit all preprocessing on training
only, and freeze test before modeling. No dates or hospital IDs support temporal/site validation.
Full-cohort EDA has already examined outcomes: future test is isolated from subsequent fitting
and tuning, but is not a fully unseen confirmatory sample.

Read the [raw-data report](reports/data_quality/raw_data_report.md),
[executed notebook](notebooks/01_data_understanding.ipynb),
[dataset comparison and limitations](docs/dataset_selection.md), and
[data dictionary](docs/data_dictionary.md).

Phase 2: [executed EDA notebook](notebooks/02_eda.ipynb),
[missingness decisions](reports/data_quality/missingness_strategy.md),
[split design](reports/modeling/split_strategy.md),
[cleaning contract](reports/modeling/cleaning_strategy.md), and
[feature-engineering discovery plan](reports/feature_engineering_plan.md).

Local validation: **38 tests passed**, all 11 EDA code cells executed with zero error outputs,
and Ruff/environment checks passed. The [verification record](reports/eda/verification.json)
includes raw-integrity checks, table reconciliation and reviewed figure hashes.

## Dataset

[UCI Diabetes 130-US Hospitals, 1999–2008](https://doi.org/10.24432/C5230J),
by Clore, Cios, DeShazo and Strack (2014), licensed CC BY 4.0. Chosen after comparing
MIMIC-IV, HCUP NRD and Synthea for target fit, access and reproducibility.
The population is selected inpatient encounters with diabetes, not all hospital patients.

The validated positive label is `readmitted == '<30'`. `>30` and `NO` are negative.
`NO` means no recorded readmission, which does not ensure complete follow-up elsewhere.
The source categories do not allow the exact day-30 boundary to be independently verified.

## Reproduce the executed work

From the repository root on macOS/Linux (Python 3.11+; verified locally with 3.12.2):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation .
make phase1
make phase2
```

On Windows, activate with `.venv\Scripts\Activate.ps1` and use the equivalent commands:

```bash
python -m readmit_iq.data.download
python -m readmit_iq.data.inspect
python scripts/execute_notebook.py
python scripts/execute_notebook.py notebooks/02_eda.ipynb
python scripts/verify_environment.py
python -m pip check
python -m pytest
```

`requirements.txt` pins the environment used by both phases. No Phase 2 dependency was added.
Platform-only packages use markers.
Exact pinned versions were tested on Python 3.12/macOS arm64; other Python/platform combinations
are not verified locally. If a pin is unavailable, resolve `python -m pip install '.[dev]'`
in a fresh environment and record a separate lock instead of silently changing this one.
`pyproject.toml` declares direct dependencies and future modeling/API/demo extras. Those extras
are deferred and have not been installed, resolved together or tested.

Use `make download`, `make inspect`, `make notebook`, `make verify`, `make test`, or `make lint`
for Phase 1 steps. `make eda` rebuilds the cohort/reports/figures; `make eda-notebook` executes
the complete Phase 2 analysis and embeds its outputs. `make phase2` executes that notebook,
checks the environment, lints and runs the complete test suite. It expects Phase 1 source files
and reports to exist. To work interactively, run `.venv/bin/jupyter lab`; the Python kernel
must use this project's virtual environment. No global kernel registration is required.
`make notebook` registers the ReadmitIQ kernel inside `.venv` and executes with that interpreter.

Source settings live in `configs/config.yaml`; Phase 2 policy lives in `configs/phase2.yaml`.
Source hashes enforce the reviewed release. A changed
download fails verification and requires review. Raw data, `.venv`, credentials, caches and
model files are excluded from Git. No credentials or `.env` are required. Optional overrides
are documented in `.env.example`. The planned split seed is 42; current analysis is deterministic.

## Structure

```text
configs/                     Source contract, explicit cohort rules and split design
data/{raw,interim,processed}/ Verified raw; ignored cohort/membership; processed still empty
docs/                        Dataset selection and source dictionary
notebooks/                   Executed 01_data_understanding.ipynb and 02_eda.ipynb
src/readmit_iq/data/          Download, load, validate and inspect modules
src/readmit_iq/analysis/      Feature audit, patient-cluster statistics, reports and plots
reports/data_quality/        Raw checks, cohort report, feature audit and missingness decisions
reports/eda/                 Reproducible descriptive tables and summary JSON
reports/figures/eda/          Seven reviewed EDA figures
reports/modeling/            Split and cleaning design only; no model results
scripts/                     Environment verification and notebook execution
tests/                       Source, cohort, normalization, audit and statistical checks
models/                      Empty until modeling is authorized
```

The `src/readmit_iq` package separates imports from the repository root. A regular package
installation is used because this local macOS environment hides editable-install `.pth` files.
After modifying source modules, reinstall with `python -m pip install --no-deps --no-build-isolation .`.
Run commands from the repository root, or set `READMITIQ_CONFIG` to the YAML file's full path.
Empty API, training scripts and future notebooks are intentionally not scaffolded.
Docker, advanced ML dependencies and service CI will be added when those phases begin.
Local Git is initialized on `main`; no GitHub remote is configured. The planned remote name
is `readmit-iq`, to be connected when the user creates it.
The GitHub Actions workflow runs source inspection, executed EDA, lint and tests on Python 3.12.
It is configured but cannot run remotely until a GitHub repository is connected and pushed.

## Phase 3, pending authorization

Confirm the mixed postacute/unknown-destination policy and live scoring-time availability.
Freeze patient partitions, then establish simple baselines and test only justified feature ideas:
separate prior-use counts, optional any-prior indicators, nonlinear burden effects and a verified
clinical diagnosis mapping. Evaluate disposition with/without ablations on development data.
No later phase starts automatically.
