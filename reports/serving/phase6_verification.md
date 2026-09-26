# Phase 6 verification

Verified locally on 2026-09-26, Python 3.12.2 / macOS arm64 and Docker Engine 28.0.4
(Linux arm64 containers). Phase 6 packages the existing Phase 5 model. No model, preprocessing,
feature, calibration, capacity policy or final-test decision was changed. No final-test record
or final prediction pass was used in this phase's serving checks.

## Integrity and inference

- The original 4,127-byte model matches SHA-256
  `ff82996f1f49008dec373655d53cf95a2ce5d940cb7dbc2bd1e25ea6fe447fa8`.
- Startup verifies committed specification/metadata/code/cohort hashes, exact features,
  Python 3.12 and frozen NumPy/pandas/SciPy/scikit-learn/joblib versions before loading once.
  Original Phase 5 freeze and protected implementation hashes pass unchanged.
- The API's exact ten-field feature matrix and 77-column transformed matrix match the direct
  frozen pipeline. Across 1,000 synthetic inputs, maximum probability difference is **0.0**
  at an absolute tolerance of `1e-12`; repeated responses are identical.
- All five required endpoints, Swagger and OpenAPI were exercised. Real Docker-to-host
  probabilities also differ by **0.0**. Documented curl examples returned success.
- Original ranking/ties match exactly. Batches of 9, 20 and 1,000 select **0, 2 and 100**.
  Single and score-only batch responses do not assign outreach decisions.

Machine-readable evidence: [parity and latency](verification.json),
[real container smoke](local_container_verification.json).

## Tests and local UI

**296 distinct tests passed locally:** 196 existing tests plus 100 Phase 6 tests.
The backend run has **287 passed**, with the UI module intentionally skipped; the isolated
UI run has **9 passed**. The 100 new tests comprise 87 API/schema/ranking/loader tests with
explicit doubles or negative artifact cases, **4 real-artifact integration tests**, and
9 CSV/UI tests through a mocked HTTP boundary. No fixture pretends to be the frozen model:
its version is `contract-test-double` and `artifact_verified` is false.

Coverage includes required/extra fields, strict integers, known missing markers, invalid codes,
eligibility, empty/duplicate/oversized batches, model load-once, missing/corrupt/runtime-mismatched
artifacts, invalid inference output, privacy-safe logs/errors, exact floor policy, deterministic
ties, metadata immutability, preprocessing parity and UI error handling. Ruff check/format and
dependency checks pass. Local `make phase6` requires the real artifact and runs no modeling job.

Headless Chrome exercised the running Docker UI and real HTTP API. The single synthetic
example displayed **6.5%**, with batch-context language; the batch tab displayed **20 eligible,
2 prioritized, 10% assumed capacity**, ranks and a risk chart. Screenshots were inspected locally.
CSV parser tests verify the bundled CSV equals the hand-authored JSON batch. No real records
were used or saved as examples.

## Dependencies and Docker

Both `api` and `demo` targets build and run as UID **10001**, with healthy readiness checks.
The digest-pinned Python 3.12 slim base and exact runtime pins are recorded in the Dockerfile
and requirements files. Compose restricts host ports to loopback and uses read-only filesystems,
dropped capabilities and a temporary `/tmp`.

The original ignored model is a **read-only runtime mount** in the API only; it is absent
from Git, the build context and image layers. An unmounted image returns **503** for health,
metadata and prediction, reports `model_missing`, and does not load a substitute. Corrupted
artifact rejection occurs before deserialization in loader tests. Startup verifies the mounted
real artifact again. The images contain no source datasets, test predictions, MLflow runs or
notebooks. No cloud service was created.

An integration check caught a real environment conflict: installing Streamlit/PyArrow beside
the historical modeling stack changed pandas' automatic string backend and broke split tests.
The final design separates `.venv-demo` and the UI container. Backend tests pass without
changing the frozen splitting implementation or scientific pins. The original `requirements.txt`
is protected by the final evaluation lock; Phase 6 uses separate serving, demo and development
overlay lock files rather than modifying it.

## Warmed synthetic request latency

Five warm-up requests and 20 measured requests per size, one concurrent caller, two native
threads. TestClient measurements include request validation, frozen inference, ranking for
batches and JSON responses; they exclude network transport and cold startup.

| Records | Endpoint | Median | p95 |
|---:|---|---:|---:|
| 1 | `/v1/predict` | 8.453 ms | 9.576 ms |
| 100 | `/v1/prioritize` | 10.075 ms | 10.216 ms |
| 1,000 | `/v1/prioritize` | 23.558 ms | 63.257 ms |

The 1,000-record JSON request was 299,343 bytes. The tested API cap is 1,000 records and
1 MiB, enforced before inference. These modest local synthetic measurements do not establish
production throughput or latency under concurrency. Re-running the benchmark naturally changes timings.

## CI and remaining scope

The workflow retains all earlier notebook and frozen-final protections and adds a separate
serving job: contract/UI tests, lint, dependency checks, both Docker builds and missing-model
503 smoke. The ignored real model is absent in CI, so its four integration cases explicitly
skip. Real-artifact and live UI/container evidence above is local. Remote Phase 6 execution
will be recorded here after the authorized push and completed Actions run.

Existing dependency deprecation notices (Starlette TestClient/httpx and SHAP colormap methods)
are non-failing; frozen dependencies were not upgraded to suppress them. The demonstration
has no authentication/TLS, contemporary clinical validation, complete eligibility ascertainment,
patient-level repeat-contact policy or hospital monitoring. No claim of HIPAA compliance is made.
See the [model card](../../docs/model_card.md) for low-utilization and subgroup limitations.

Phase 7 may refine the portfolio walkthrough and narrative after separate authorization.
It has not started. Cloud hosting and clinical use remain outside this phase.
