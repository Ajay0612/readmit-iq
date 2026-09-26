# Run the frozen ReadmitIQ demonstration

Portfolio demonstration only. Built using historical 1999–2008 hospital data. Not validated
for clinical decision-making or patient care. No claim of HIPAA compliance is made.
Use synthetic demonstration inputs; this local application has no authentication or TLS.

## Restore and verify the artifact

Use Python **3.12** and run commands from the repository root:

```bash
git clone https://github.com/Ajay0612/readmit-iq.git
cd readmit-iq
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-serving.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation .
mkdir -p models/final
```

Restore the existing, trusted Phase 5 `readmit_iq_logistic.joblib` to
`models/final/readmit_iq_logistic.joblib`. Obtain those original bytes from the project owner
or your trusted Phase 5 backup; a clone alone does **not** include them. Do not download an
arbitrary pickle, refit a replacement, or modify the expected hash to accept another file.

Expected size: **4,127 bytes**. Expected SHA-256:
`ff82996f1f49008dec373655d53cf95a2ce5d940cb7dbc2bd1e25ea6fe447fa8`.

```bash
make verify-model
make api
```

The verifier checks the specification, metadata, feature order, original custom pipeline
code, cohort mappings, Python minor version and scientific package versions. It verifies
the model's size/hash **before** deserializing the same in-memory byte buffer. Startup loads
one model per API process. Metadata is immutable; there is no fallback or bypass switch.
Missing, changed or incompatible artifacts leave readiness and inference unavailable (503).
Restart the API after restoring the original artifact.

Open [health](http://localhost:8000/health), [Swagger](http://localhost:8000/docs) or
[safe provenance](http://localhost:8000/v1/model). The API binds to loopback by default.
`READMITIQ_MODEL_PATH` optionally chooses the artifact location;
`READMITIQ_SERVING_ROOT` chooses the root containing the committed bundle files.
`.env.example` documents settings but the program does not automatically load `.env`.

## UI environment

In a second terminal:

```bash
make install-demo
make demo
```

Open [Streamlit](http://localhost:8501). Choose one of three hand-authored examples and
estimate its risk, or prioritize the bundled 20-record batch. CSV upload uses the exact
header in [the synthetic batch](../examples/synthetic/batch.csv). The UI shows probabilities,
ranks, two selections from the 20-record batch, and a ranked risk chart. It links existing
explainability reports; it does not compute SHAP or load a second model.

Streamlit calls the same HTTP API used by REST clients. Set `READMITIQ_API_URL` only if that
API uses another address. The UI's `.venv-demo` is deliberately separate: adding Streamlit's
PyArrow dependency to the modeling environment changes pandas string storage and breaks the
existing frozen split regression tests. Do not merge these environments. Docker also separates
the API and UI. No changes to splitting, preprocessing or estimator weights were made.

## Verified synthetic REST examples

```bash
curl --fail-with-body http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/synthetic/single.json

curl --fail-with-body http://localhost:8000/v1/prioritize \
  -H 'Content-Type: application/json' \
  --data-binary @examples/synthetic/batch.json
```

All example files are **synthetic demonstration inputs**, not sampled patient rows. The
single example has no prior utilization; the other patterns illustrate prior utilization
and rehabilitation discharge context. These are illustrative associations, not causes.

## API v1 contract

| Endpoint | Behavior |
|---|---|
| `GET /health` | 200 only after successful model verification; otherwise 503 |
| `GET /v1/model` | Safe model/version/features/provenance, verified hash and portfolio policy |
| `POST /v1/predict` | Exactly ten feature fields; full-precision probability, percentage and null outreach decision |
| `POST /v1/predict/batch` | `records` containing unique `request_id` and nested `features`; input-order probabilities |
| `POST /v1/prioritize` | Same input; ranked results, selection flags and capacity summary |

Input field names and their order are fixed:

| Field | Accepted v1 values |
|---|---|
| `time_in_hospital` | JSON integer 1–14, the documented source range |
| `number_inpatient`, `number_emergency`, `number_outpatient` | Nonnegative JSON integers; signed-int32 upper limit is a transport bound, not a medical range |
| `age` | Exact ten-year source bands `[0-10)` through `[90-100)` |
| `gender` | `Female`, `Male`, `Unknown/Invalid`, `Unknown`, or a known missing marker |
| `admission_type_id`, `admission_source_id` | Integer codes from the committed historical mapping, or known missing markers |
| `discharge_disposition_id` | Integer historical code, followed by original Phase 2 eligibility validation |
| `medical_specialty` | Exact training vocabulary in `serving/contract.json`, `Other`, or a known missing marker |

Known missing markers for the nullable categorical fields are JSON `null`, `""`, `"?"` and
`"Unknown"`. Fields remain required. Numeric stay/counts, age and destination cannot be
missing. Existing normalization and training-fitted rare-category pooling remain unchanged.
Unrecognized spellings/codes are rejected; a caller may deliberately provide `Other` for
specialty. The UI offers documented category/code choices. JSON numeric strings, floats in
integer fields, booleans, extra predictors, IDs within `features`, and outcomes are rejected.
CSV is a textual transport: its seven integer columns are explicitly parsed as unsigned
decimal integers before submission; fractions and malformed numbers are rejected.

Eligible destination codes are **1, 3, 4, 6, 7, 8, 16, 17, 22 and 24**, directly governed
by `configs/phase2.yaml` and the original cohort implementation. Death, hospice, transfers,
unconfirmed departures and unknown destinations are rejected. A mixed-eligibility batch
fails as a whole before inference; records are never silently dropped. The ten-feature
contract cannot independently confirm diabetes-cohort membership, hospital coverage or
field-arrival timing. The caller must supply an appropriate synthetic cohort context and
score only after the discharge destination is confirmed.

Each batch needs a stable unique ASCII `request_id` (1–64 characters, letters/digits followed
by letters/digits/`_ . : -`). This is a non-predictive encounter tie key, not a patient feature.
Preserve it when reordering/resubmitting a batch. There is no repeat-patient deduplication or
patient-level contact policy. Do not use real identifiers in this demonstration.

Ranking calls the original Phase 5 implementation: descending unrounded risk, then ascending
SHA-256 of `42:` followed by the request ID. Select exactly `floor(0.10 × N)`. Fewer than ten
records select **zero**. The cutoff depends on the available batch; a single score cannot
determine membership in an unknown future batch. The 10% capacity is a portfolio assumption.

The service accepts **1–1,000 records**, with a **1 MiB** request-body limit checked before
JSON parsing, including chunked bodies. These bounds cover the locally tested 1,000-record
synthetic workload; they are not a throughput guarantee. Invalid input/eligibility yields
structured 422, excessive bytes 413, unavailable model 503, and inference failures a sanitized
500. Responses never expose stack traces. Logging records event/version/count/latency/error
codes without payloads or request IDs; default access logs are disabled. Request payloads
are not saved to disk. This does not establish hospital security or privacy certification.

## Docker

Install/start Docker Desktop or Docker Engine with Compose, then:

```bash
make docker-build
make docker-run
docker compose ps
# API http://localhost:8000/docs ; UI http://localhost:8501
make docker-stop
```

Stop native processes first if using the same ports. `docker-build` verifies the local model
before building. The images intentionally contain **no model binary**. Compose mounts only
the original artifact read-only at `/model/readmit_iq_logistic.joblib` in the API container;
startup verifies it again. The UI has no model mount. A fresh clone cannot serve predictions
until the trusted artifact is restored. No automatic training/download occurs.

Both images use the digest-pinned official Python 3.12 slim base, pinned dependencies, UID
10001, health checks, no reload/debug and loopback host bindings. Compose uses read-only
filesystems, temporary `/tmp`, dropped capabilities and no privilege escalation. The Docker
context allowlist excludes datasets, model binaries, saved predictions, notebooks, MLflow,
secrets, Git and virtual environments. Plain `docker build .` defaults to the API; Compose
explicitly selects the separate `api` and `demo` targets. No cloud deployment is configured.

## Verify and reproduce

For the full development checks, preserve the Phase 1–5 pins:

```bash
make install-phase6
make install-demo
make phase6
```

`requirements.txt` is byte-frozen by the final evaluation lock. Instead of invalidating that
guard, `requirements-serving.txt` pins the minimal package-compatible service runtime,
`requirements-demo.txt` pins its isolated UI, and `requirements-phase6.txt` combines the
original lock with serving dependencies. The broad historical modeling extra is never needed.
After edits reinstall the package in each environment with `pip install --no-deps
--no-build-isolation .`; editable installs are not used.

`make phase6` checks integrity, lint, dependencies, existing tests, real-artifact synthetic
integration, isolated UI tests and warmed synthetic parity/latency. It never trains, accesses
final test records or runs `make final-eval`. `make phase6-test` runs serving contract tests;
real-artifact tests skip when absent, and UI tests run separately when Streamlit is absent.
`READMITIQ_REQUIRE_REAL_MODEL=1` makes missing real weights an integration-test failure.

CI retains earlier notebook/freeze checks. Its separate serving job runs mocked contract/UI
tests and loader rejection tests, builds both images, and verifies missing-model 503 behavior.
Its fixture explicitly reports `artifact_verified: false` and `contract-test-double`.
It does not claim real-model inference success. The real binary smoke/parity checks are local.
See the [verification report](../reports/serving/phase6_verification.md) and
[model card](model_card.md) for evidence and limitations.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/),
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
[Docker build practices](https://docs.docker.com/build/building/best-practices/).
