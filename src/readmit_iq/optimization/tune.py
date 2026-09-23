"""Bounded Optuna searches and feature comparisons using training patients only."""

import json
import logging
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.modeling.metrics import classification_metrics
from readmit_iq.optimization.contract import (
    BLOCKED,
    digest,
    git_revision,
    load_partition,
    settings,
    training_fingerprint,
    verify_contract,
)
from readmit_iq.optimization.cv import grouped_folds
from readmit_iq.optimization.pipelines import DEFAULTS, build_pipeline
from readmit_iq.optimization.tracking import LocalTracker

LOGGER = logging.getLogger(__name__)
CV_METRICS = ["average_precision", "roc_auc", "brier", "recall", "precision", "f1"]


def cross_validate(
    family: str,
    configuration: str,
    parameters: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list,
    policy: dict,
) -> tuple[dict, list]:
    """Every fold fits its complete preprocessing pipeline using its fit subset only."""
    rows = []
    for fold, (fit, holdout) in enumerate(folds):
        model = build_pipeline(
            family,
            configuration,
            parameters,
            policy["random_seed"],
            policy["features"]["rare_min_count"],
        )
        start = time.perf_counter()
        with threadpool_limits(limits=policy["threads"]), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X.iloc[fit], y[fit])
            probability = model.predict_proba(X.iloc[holdout])[:, 1]
        encoder = model.named_steps["preprocess"].named_transformers_["categorical"]
        if encoder.n_fit_rows_ != len(fit):
            raise ValueError("CV preprocessing fit on an unexpected row count")
        rows.append(
            {
                "fold": fold,
                **classification_metrics(y[holdout], probability),
                "fit_encounters": len(fit),
                "holdout_encounters": len(holdout),
                "encoded_features": len(model.named_steps["preprocess"].get_feature_names_out()),
                "duration_seconds": time.perf_counter() - start,
            }
        )
    summary = {}
    for metric in CV_METRICS:
        values = [r[metric] for r in rows]
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_sd"] = float(np.std(values, ddof=1))
    summary.update(
        cv_ap_min=min(r["average_precision"] for r in rows),
        cv_ap_max=max(r["average_precision"] for r in rows),
    )
    return summary, rows


def choose_configuration(table: pd.DataFrame, policy: dict) -> str:
    """Sensitivity features cannot win, regardless of their apparent CV performance."""
    eligible = table.loc[table.configuration.isin(policy["features"]["allowed"])]
    cutoff = eligible.cv_average_precision_mean.max() - policy["features"]["minimum_ap_gain"]
    for configuration in policy["features"]["preference"]:
        row = eligible.loc[eligible.configuration.eq(configuration)].iloc[0]
        if row.cv_average_precision_mean >= cutoff:
            return configuration
    raise ValueError("No eligible feature configuration")


def suggest_parameters(trial, family: str, space: dict) -> dict:
    if family == "logistic":
        return {"C": trial.suggest_float("C", *space["C"], log=True)}
    return {
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        **{
            name: trial.suggest_categorical(name, space[name])
            for name in ["max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
        },
    }


class PlateauStopper:
    """Prespecified sequential stopping based only on completed training-CV objectives."""

    def __init__(self, minimum_trials: int, patience: int, improvement: float):
        self.minimum_trials, self.patience, self.improvement = minimum_trials, patience, improvement
        self.best, self.last_improvement = -np.inf, 0

    def __call__(self, study, trial):
        if trial.value > self.best + self.improvement:
            self.best, self.last_improvement = trial.value, trial.number
        if (
            len(study.trials) >= self.minimum_trials
            and trial.number - self.last_improvement >= self.patience
        ):
            study.set_user_attr("stop_reason", "prespecified training-CV plateau")
            study.stop()


def optimize(
    family: str,
    configuration: str,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list,
    policy: dict,
    tracker: LocalTracker,
) -> optuna.Study:
    space = policy["tuning"][family]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{family}-{configuration}",
    )
    names = (
        ["C"]
        if family == "logistic"
        else [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]
    )
    study.enqueue_trial({name: DEFAULTS[family][name] for name in names})

    def objective(trial):
        parameters = suggest_parameters(trial, family, space)
        summary, rows = cross_validate(family, configuration, parameters, X, y, folds, policy)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("folds", rows)
        run_id = tracker.log(
            f"tune-{family}-{trial.number}",
            {
                "family": family,
                "configuration": configuration,
                "calibration": "uncalibrated",
                "parameters": parameters,
                "partition": "train",
                "trial": trial.number,
            },
            {**summary, "duration_seconds": sum(r["duration_seconds"] for r in rows)},
        )
        trial.set_user_attr("mlflow_run_id", run_id)
        return summary["cv_average_precision_mean"]

    stopper = PlateauStopper(
        space["minimum_trials"],
        space["plateau_patience"],
        policy["tuning"]["plateau_minimum_improvement"],
    )
    study.optimize(objective, n_trials=space["trials"], n_jobs=1, callbacks=[stopper])
    return study


def run_tuning() -> dict:
    root, policy = settings()
    frozen = verify_contract(root)
    fingerprint = training_fingerprint(root, policy, frozen)
    token = digest(fingerprint)
    artifacts, out = root / policy["artifact_dir"], root / policy["report_dir"]
    artifacts.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    selection_path = artifacts / "training_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["fingerprint"] != fingerprint:
            raise ValueError("Completed training contract changed; refusing silent retuning")
        LOGGER.info("Reusing verified training-only selection %s", token[:12])
        return selection
    if (artifacts / "validation_scores.json").exists():
        raise ValueError("Validation already scored; retuning is forbidden")
    frame = load_partition(root, "train", "tuning")
    folds, fold_table, fold_hash = grouped_folds(
        frame, policy["cv"]["folds"], policy["random_seed"], policy["cv"]["prevalence_tolerance"]
    )
    fold_table.to_csv(out / "cv_folds.csv", index=False)
    write_json(
        out / "development_contract.json",
        {
            "split_manifest_sha256": fingerprint["split_manifest_sha256"],
            "frozen_files": frozen["files"],
            "patient_overlap": frozen["patient_overlap"],
            "fold_membership_sha256": fold_hash,
            "training_encounters": len(frame),
            "test_records_loaded": False,
            "test_evaluated": False,
            "scoring_time": "Confirmed discharge after destination is known",
        },
    )
    X, y = frame.drop(columns=BLOCKED), frame.readmitted_lt30.to_numpy()
    revision = git_revision(root)
    tracker = LocalTracker(root, policy, revision)
    comparisons, fold_rows = [], []
    for family in ["logistic", "boosting"]:
        for configuration in policy["features"]["allowed"] + policy["features"]["sensitivity_only"]:
            LOGGER.info("Training-only feature comparison: %s / %s", family, configuration)
            summary, rows = cross_validate(family, configuration, {}, X, y, folds, policy)
            run_id = tracker.log(
                f"features-{family}-{configuration}",
                {
                    "family": family,
                    "configuration": configuration,
                    "parameters": DEFAULTS[family],
                    "partition": "train",
                    "calibration": "uncalibrated",
                    "selection_eligible": configuration in policy["features"]["allowed"],
                },
                summary,
            )
            comparisons.append(
                dict(
                    family=family,
                    configuration=configuration,
                    **summary,
                    eligible=configuration in policy["features"]["allowed"],
                    mlflow_run_id=run_id,
                )
            )
            fold_rows.extend(
                dict(family=family, configuration=configuration, **row) for row in rows
            )
    table = pd.DataFrame(comparisons)
    table.to_csv(out / "feature_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "feature_fold_metrics.csv", index=False)
    selected, trial_rows, tuning_folds = {}, [], []
    for family in ["logistic", "boosting"]:
        configuration = choose_configuration(table.loc[table.family.eq(family)], policy)
        LOGGER.info("Optuna: %s uses eligible configuration %s", family, configuration)
        study = optimize(family, configuration, X, y, folds, policy, tracker)
        joblib.dump(study, artifacts / f"{family}_study.joblib")
        best = study.best_trial
        selected[family] = dict(
            configuration=configuration,
            parameters=best.params,
            **best.user_attrs["summary"],
            best_trial=best.number,
            trials=len(study.trials),
            stop_reason=study.user_attrs.get("stop_reason", "trial budget"),
        )
        for trial in study.trials:
            trial_rows.append(
                dict(
                    family=family,
                    configuration=configuration,
                    trial=trial.number,
                    state=trial.state.name,
                    parameters=json.dumps(trial.params, sort_keys=True),
                    **trial.user_attrs["summary"],
                    duration_seconds=trial.duration.total_seconds(),
                    mlflow_run_id=trial.user_attrs["mlflow_run_id"],
                )
            )
            tuning_folds.extend(
                dict(family=family, configuration=configuration, trial=trial.number, **row)
                for row in trial.user_attrs["folds"]
            )
    LOGGER.info("Training-only fixed random-forest reference")
    summary, rows = cross_validate("forest", "raw", {}, X, y, folds, policy)
    selected["forest"] = dict(
        configuration="raw",
        parameters={},
        **summary,
        trials=1,
        best_trial=0,
        stop_reason="fixed reference; no tuning",
    )
    tracker.log(
        "fixed-forest",
        {
            "family": "forest",
            "configuration": "raw",
            "parameters": DEFAULTS["forest"],
            "partition": "train",
            "calibration": "uncalibrated",
        },
        summary,
    )
    tuning_folds.extend(dict(family="forest", configuration="raw", trial=0, **row) for row in rows)
    pd.DataFrame(trial_rows).to_csv(out / "optuna_trials.csv", index=False)
    pd.DataFrame(tuning_folds).to_csv(out / "tuning_fold_metrics.csv", index=False)
    selection = dict(
        fingerprint=fingerprint,
        fingerprint_sha256=token,
        git_commit=revision,
        fold_membership_sha256=fold_hash,
        selected=selected,
        validation_opened=False,
        test_evaluated=False,
    )
    write_json(selection_path, selection)
    write_json(out / "training_selection.json", selection)
    return selection
