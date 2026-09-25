"""Create and verify a committed, immutable decision contract before any test parsing."""

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime

import joblib

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.decision_support.context import settings
from readmit_iq.optimization.contract import digest, git_revision, verify_contract
from readmit_iq.optimization.pipelines import schema

PROTOCOL = "reports/modeling/final_evaluation_protocol.md"
SPECIFICATION = "reports/modeling/final_model_specification.json"
LOCK = "reports/modeling/phase5/final_evaluation_lock.json"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "shap", "numba", "llvmlite"]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def implementation_files(root):
    names = [
        "context",
        "ranking",
        "uncertainty",
        "subgroups",
        "evaluation",
        "freeze",
        "final_evaluation",
    ]
    paths = [f"src/readmit_iq/decision_support/{name}.py" for name in names]
    paths += [
        f"src/readmit_iq/modeling/{name}.py"
        for name in ["features", "preprocessing", "diagnoses", "metrics", "splitting"]
    ]
    paths += [f"src/readmit_iq/optimization/{name}.py" for name in ["contract", "pipelines"]]
    paths += [
        "configs/phase5.yaml",
        "configs/phase4.yaml",
        "configs/phase3.yaml",
        "configs/phase2.yaml",
        "requirements.txt",
        "reports/modeling/final_feature_policy.md",
    ]
    return {path: sha256(root / path) for path in paths}


def write_freeze():
    """Run after committing verified development code; then commit these three freeze artifacts."""
    root, policy = settings()
    if (root / LOCK).exists() or (root / "reports/modeling/phase5/test/summary.json").exists():
        raise ValueError("A final freeze/evaluation already exists; refuse silent replacement")
    frozen = verify_contract(root)
    code_commit = git_revision(root)
    implementations = implementation_files(root)
    for path in implementations:
        if git(root, "show", f"{code_commit}:{path}") != (root / path).read_bytes():
            raise ValueError("Commit all decision/evaluation code before creating the final freeze")
    validation = json.loads((root / policy["report_dir"] / "validation/summary.json").read_text())
    if validation["targeting"] != policy["targeting"]:
        raise ValueError("Validation did not evaluate the selected targeting rule")
    fitted = json.loads((root / "models/development/phase4/fit_manifest.json").read_text())
    training = json.loads((root / "models/development/phase4/training_selection.json").read_text())
    models = {}
    numeric, categorical = schema("raw")
    for name in policy["final_test"]["models"]:
        path = root / "models/development/phase4" / f"{name}.joblib"
        verify_file(path, fitted["models"][name]["sha256"])
        model = joblib.load(path)
        preprocess = model.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["categorical"]
        scaler = preprocess.named_transformers_["numeric"]
        models[name] = dict(
            **fitted["models"][name],
            path=str(path.relative_to(root)),
            estimator_class=type(model[-1]).__name__,
            exact_estimator_parameters=model[-1].get_params(deep=False),
            numeric_features=numeric,
            categorical_features=categorical,
            preprocessing=dict(
                feature_configuration=model.named_steps["features"].configuration,
                numeric_scaling="StandardScaler" if name.startswith("logistic") else "none",
                numeric_mean=scaler.mean_.tolist() if name.startswith("logistic") else None,
                numeric_scale=scaler.scale_.tolist() if name.startswith("logistic") else None,
                numeric_imputation="none; approved numeric source columns are complete",
                categorical_mode=encoder.mode,
                rare_min_count=encoder.min_count,
                categorical_fit_rows=encoder.n_fit_rows_,
                categorical_levels=dict(zip(categorical, encoder.categories_, strict=True)),
                unseen_category="Other after source-code normalization",
                onehot_drop="none; fully encoded" if encoder.mode == "onehot" else None,
                output_columns=preprocess.get_feature_names_out().tolist(),
                column_order="numeric followed by categorical; remainder dropped",
            ),
        )
    specification = dict(
        version="1.0.0-frozen-portfolio",
        primary=policy["primary"],
        challenger=policy["challenger"],
        models=models,
        effective_logistic_penalty="L2 (l1_ratio=0; sklearn penalty argument is deprecated)",
        calibration="uncalibrated",
        random_seed=42,
        fit_partition="train",
        fit_encounters=63563,
        scoring_moment="Confirmed discharge once destination is known",
        target_definition="1 if recorded readmitted is <30; 0 for >30 or NO",
        eligibility_policy_sha256=sha256(root / "configs/phase2.yaml"),
        training_commit=training["git_commit"],
        pretest_code_commit=code_commit,
        library_versions={name: importlib.metadata.version(name) for name in PACKAGES},
        python=platform.python_version(),
        platform=platform.platform(),
        targeting=policy["targeting"],
        frozen_partition_hashes=frozen["files"],
        validation_metrics=validation["metrics"],
    )
    write_json(root / SPECIFICATION, specification)
    text = f"""# Frozen final evaluation protocol

This decision record was prepared **before opening the frozen test table**. Its Git commit
must exist before `make final-eval` is permitted. Model/feature/calibration/policy choices
cannot change in response to test results. The source/evaluation code commit is
`{code_commit}`; the training commit is `{training["git_commit"]}`. The freeze commit itself
is resolved and verified from this file's Git history and recorded with the final result.

## Fixed models and data

Primary: **uncalibrated Logistic Regression**, raw utilization configuration, ten original
allowed source predictors, L2/lbfgs, C=0.09988151348099303, seed 42, no class weights.
Challenger: **uncalibrated Histogram Gradient Boosting**, explicitly included for a single
prespecified comparison. Exact parameters, preprocessing, feature names, library versions,
training provenance and both binary SHA-256 values are in
[the model specification](final_model_specification.json).
The actual trained pipelines are reused byte-for-byte; no train+validation refit occurs.
The [final feature policy](final_feature_policy.md) keeps timing-uncertain families excluded.

The original frozen patient split and all hashes remain binding. Test contains 13,549
encounters and 9,757 patients according to the existing allocation record. No test outcomes,
probabilities or errors were used to choose any decision here. Earlier full-cohort EDA did
examine eventual test outcomes before partitioning; this is a held-out model evaluation,
not a fully untouched confirmatory study.

## Frozen operational policy

Prioritize the **top 10% highest-risk eligible discharge encounters** in an available outreach
batch. Capacity is a **portfolio scenario assumption**, not measured hospital staffing.
Rank descending by predicted probability; equal scores use ascending SHA-256 of
`42:<encounter_id>`. The key is only an outcome-independent tie-break, never a predictor or
chronology. Select exactly `floor(0.10 * N)` encounters. Report unique people separately;
the data cannot identify simultaneous caseload or a daily/weekly batch schedule.

The corresponding probability cutoff may differ in test because ranking uses that batch's
unlabeled scores. This is an application of the frozen capacity rule, not post-test threshold
optimization. Do not switch to a validation probability cutoff, optimize F1, use subgroup
thresholds or change capacity after test. The pooled historical evaluation cannot establish
performance in real-time batches with a changing mix of discharges.

Validation: 1,359 targeted encounters / 871 unique historical patients captured 321 of 1,499
readmissions (21.4143% recall, 23.6203% precision, 2.14143x lift). Top 5% has higher precision
but captures only 13.2755%; larger capacities increase coverage at declining precision.
Ten percent is an interpretable limited-outreach demonstration, not an empirically optimal
cost/utility choice. Under this policy 732 readmissions without prior inpatient use were
missed; this limitation does not trigger a model or policy change. Risk prioritization must
not be interpreted as denying standard care to unflagged patients.

## Metrics, uncertainty and diagnostics declared before test

Report AP (non-interpolated), ROC-AUC, Brier, mean risk/prevalence, quantile reliability,
precision/recall curves, selected-policy recall/precision/specificity, confusion counts,
encounters/unique people flagged, lift and gains. Report the fixed 5/10/15/20/25% capacity
grid and ten risk groups as descriptive sensitivity summaries only; 10% stays primary.
Scale encounter-level yields to an illustrative 10,000-discharge scenario. No prevented
readmissions, causal effect, cost savings or scaled unique-person counts are estimated.

Use 1,000 paired patient-cluster bootstrap draws, seed 42, resampling all encounters of
each patient together. Apply the frozen ranking rule within each weighted draw; report
percentile 95% intervals for AP, ROC-AUC, Brier, recall, precision and lift. The intervals
condition on fitted models and exclude retraining/selection and temporal/site uncertainty.

Apply identical validation/test subgroup definitions from `configs/phase5.yaml`: no/any
prior inpatient and emergency use, 0–2/3+ inpatient use, recorded age bands/gender/race,
stay lengths 1–2/3–4/5–7/8–14 days, historical destination/type/source codes, and specialty/
payer missingness. Payer and race are audit-only. Preserve Unknown and counts; suppress
performance below 200 encounters, 100 patients, 30 positives or 30 negatives. Report
cluster-sandwich 95% ratio intervals for recall, precision, prevalence and mean prediction
minus observed outcome. Flag mean probability bias as material only at >=2 percentage
points with an interval excluding zero. These criteria neither certify fairness nor rule
out within-group miscalibration. No demographic recalibration or new subgroup cuts follow.

Limited error diagnostics compare false negatives/positives, low-utilization misses,
destination, stay and other declared groups with validation. Characterization does not
authorize refitting. Explanations use the already-frozen models and validation examples.
Linear SHAP is checked against logistic scores; permutation SHAP of the actual categorical
booster replaces its unsupported TreeSHAP adapter. Explanations are predictive, not causal.

## One-time execution and repeatable review

`make final-eval` requires this committed protocol, the committed JSON lock/specification,
unchanged evaluation code/configuration/dependencies, exact original model bytes and the
original partition hashes. It writes an exclusive start marker before parsing test and
calls each prespecified model's prediction method once. Saved predictions remain ignored.
Subsequent local analysis reuses verified cached predictions; an incomplete scoring attempt
cannot automatically retry. A fresh checkout containing published final results refuses
new test scoring. CI verifies the archived results and synthetic guard/reproducibility tests;
it does not repeat the real test prediction pass.

Final model metadata and an ignored copy of the primary pipeline are created after successful
evaluation. Test findings cannot change these decisions. No service or deployment is built.
"""
    (root / PROTOCOL).write_text(text)
    lock = dict(
        created_utc=datetime.now(UTC).isoformat(),
        pretest_code_commit=code_commit,
        protocol_path=PROTOCOL,
        protocol_sha256=sha256(root / PROTOCOL),
        specification_sha256=sha256(root / SPECIFICATION),
        implementation=implementations,
        policy=policy,
        policy_sha256=digest(policy),
        model_sha256={name: item["sha256"] for name, item in models.items()},
        frozen_files=frozen["files"],
        split_manifest_sha256=sha256(root / "reports/modeling/split_manifest.json"),
        test_evaluated_at_freeze=False,
    )
    write_json(root / LOCK, lock)
    return lock


def verify_freeze(root, require_models=True):
    """Verify the committed protocol and code BEFORE any caller can parse test data."""
    for path in [PROTOCOL, SPECIFICATION, LOCK]:
        if not (root / path).exists():
            raise ValueError("Final evaluation requires the committed pre-test freeze artifacts")
    lock = json.loads((root / LOCK).read_text())
    spec = json.loads((root / SPECIFICATION).read_text())
    try:
        freeze_commit = git(root, "log", "-1", "--format=%H", "--", PROTOCOL).decode().strip()
        if not freeze_commit:
            raise ValueError("Protocol has not been committed")
        for path in [PROTOCOL, SPECIFICATION, LOCK]:
            if git(root, "show", f"{freeze_commit}:{path}") != (root / path).read_bytes():
                raise ValueError("Freeze artifacts must match their committed bytes")
        git(root, "merge-base", "--is-ancestor", lock["pretest_code_commit"], freeze_commit)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Final evaluation requires a valid committed freeze and code ancestry"
        ) from error
    verify_file(root / PROTOCOL, lock["protocol_sha256"])
    verify_file(root / SPECIFICATION, lock["specification_sha256"])
    verify_file(root / "reports/modeling/split_manifest.json", lock["split_manifest_sha256"])
    for path, expected in lock["implementation"].items():
        verify_file(root / path, expected)
        if git(root, "show", f"{lock['pretest_code_commit']}:{path}") != (root / path).read_bytes():
            raise ValueError("Evaluation implementation differs from its frozen code commit")
    if digest(lock["policy"]) != lock["policy_sha256"] or lock["test_evaluated_at_freeze"]:
        raise ValueError("Invalid freeze policy record")
    if spec["targeting"] != lock["policy"]["targeting"]:
        raise ValueError("Specification and frozen targeting policy differ")
    if require_models:
        verify_contract(root)
        for name, item in spec["models"].items():
            verify_file(root / item["path"], lock["model_sha256"][name])
        for name, version in spec["library_versions"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError("Frozen inference/evaluation library version changed")
    return lock, spec, freeze_commit


if __name__ == "__main__":
    write_freeze()
