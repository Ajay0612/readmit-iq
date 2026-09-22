# ReadmitIQ

Predicting hospital readmission risk to help prioritize limited care-management outreach.

**Business question:** Which patients should a hospital prioritize at discharge for follow-up
when its care-management team cannot contact everyone?

**Current status: Phase 1 only.** The repository contains a reproducible Python environment,
verified UCI data ingestion, raw-data validation, an executed data-understanding notebook,
and documented findings. EDA, modeling, intervention simulations and prediction services
have not been built. There are no model-performance or business-impact claims yet.

This is an analytical decision-support portfolio project, not a clinical diagnostic tool.

## Actual raw-data findings

| Measure | Result |
|---|---:|
| Hospital encounters | 101,766 |
| Unique patients | 71,518 |
| Raw columns | 50 (47 candidate features, 2 IDs, 1 target) |
| `<30` readmission labels | 11,357 (11.16%) |
| Patients with multiple encounters | 16,773 |
| Duplicate rows / encounter IDs | 0 / 0 |
| Missing weight | 96.86% |
| Missing medical specialty | 49.08% |
| Missing payer code | 39.56% |

Repeated patients require care in evaluation design. `None` in A1c/glucose fields means
not measured and is preserved. `?` and administrative unknown codes require separate review.
The extract provides no encounter dates for temporal validation. All rows remain unchanged.

Read the [raw-data report](reports/data_quality/raw_data_report.md),
[executed notebook](notebooks/01_data_understanding.ipynb),
[dataset comparison and limitations](docs/dataset_selection.md), and
[data dictionary](docs/data_dictionary.md).

## Dataset

[UCI Diabetes 130-US Hospitals, 1999–2008](https://doi.org/10.24432/C5230J),
by Clore, Cios, DeShazo and Strack (2014), licensed CC BY 4.0. Chosen after comparing
MIMIC-IV, HCUP NRD and Synthea for target fit, access and reproducibility.
The population is selected inpatient encounters with diabetes, not all hospital patients.

The proposed positive label is `readmitted == '<30'`. `>30` and `NO` are negative.
`NO` means no recorded readmission, which does not ensure complete follow-up elsewhere.
The source categories do not allow the exact day-30 boundary to be independently verified.

## Reproduce Phase 1

From the repository root on macOS/Linux (Python 3.11+; verified locally with 3.12.2):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation .
make phase1
```

On Windows, activate with `.venv\Scripts\Activate.ps1` and use the equivalent commands:

```bash
python -m readmit_iq.data.download
python -m readmit_iq.data.inspect
python scripts/execute_notebook.py
python scripts/verify_environment.py
python -m pip check
python -m pytest
```

`requirements.txt` pins the installed Phase 1 environment. Platform-only packages use markers.
Exact pinned versions were tested on Python 3.12/macOS arm64; other Python/platform combinations
are not verified locally. If a pin is unavailable, resolve `python -m pip install '.[dev]'`
in a fresh environment and record a separate lock instead of silently changing this one.
`pyproject.toml` declares direct dependencies and future modeling/API/demo extras. Those extras
are deferred and have not been installed, resolved together or tested.

Use `make download`, `make inspect`, `make notebook`, `make verify`, `make test`, or `make lint`
for individual steps. To work interactively, run `.venv/bin/jupyter lab`; the Python kernel
must use this project's virtual environment. No global kernel registration is required.
`make notebook` registers the ReadmitIQ kernel inside `.venv` and executes with that interpreter.

Settings live in `configs/config.yaml`. Source hashes enforce the reviewed release. A changed
download fails verification and requires review. Raw data, `.venv`, credentials, caches and
model files are excluded from Git. No credentials or `.env` are required. Optional overrides
are documented in `.env.example`. The random seed is 42; Phase 1 inspection is deterministic.

## Structure

```text
configs/                     Validated YAML source contract and random seed
data/{raw,interim,processed}/ Original download; later stages remain empty
docs/                        Dataset selection and source dictionary
notebooks/                   Executed 01_data_understanding.ipynb only
src/readmit_iq/data/          Download, load, validate and inspect modules
reports/data_quality/        Actual summaries, checks, manifest and environment record
reports/{figures,metrics}/    Reserved for future analytical outputs
scripts/                     Environment verification and notebook execution
tests/                       Source integrity and data-contract regression tests
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
The Phase 1 GitHub Actions workflow runs lint, tests and source inspection on Python 3.12.
It is configured but cannot run remotely until a GitHub repository is connected and pushed.

## Next phase, pending review

Agree on eligible discharge encounters (including death/hospice handling), review missingness
and feature timing, then perform business-question-driven EDA. Decide a patient-grouped split
before any preprocessing is fitted. No later phase starts automatically.
