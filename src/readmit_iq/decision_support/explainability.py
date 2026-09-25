"""Validated log-odds explanations of unchanged development models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256
from readmit_iq.decision_support.ranking import target_mask
from readmit_iq.modeling.features import CORE_CATEGORICAL, UTILIZATION
from readmit_iq.optimization.contract import load_partition

NUMERIC = ["time_in_hospital", *UTILIZATION]
FEATURES = NUMERIC + CORE_CATEGORICAL


def dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def coefficient_table(pipeline, references):
    """Numeric effects per original unit; full one-hot category contrasts within each field."""
    coefficients = pipeline[-1].coef_[0]
    preprocess = pipeline.named_steps["preprocess"]
    scale = preprocess.named_transformers_["numeric"].scale_
    rows = []
    for index, name in enumerate(NUMERIC):
        beta = coefficients[index] / scale[index]
        rows.append(
            dict(
                feature=name,
                level="per original unit",
                reference="one fewer day/visit",
                coefficient=beta,
                standardized_coefficient=coefficients[index],
                odds_ratio=np.exp(beta),
                training_scale=scale[index],
                interpretation="Conditional predictive log-odds association per day/visit",
            )
        )
    encoder = preprocess.named_transformers_["categorical"]
    offset = len(NUMERIC)
    for feature, levels in zip(CORE_CATEGORICAL, encoder.categories_, strict=True):
        ref = references[feature]
        if ref not in levels:
            raise ValueError(f"Missing coefficient reference: {feature}/{ref}")
        reference_coef = coefficients[offset + levels.index(ref)]
        observed = {
            item["encoded_as"] for item in encoder.frequency_records_ if item["feature"] == feature
        }
        for index, level in enumerate(levels):
            if level not in observed:
                continue
            contrast = coefficients[offset + index] - reference_coef
            rows.append(
                dict(
                    feature=feature,
                    level=level,
                    reference=ref,
                    coefficient=contrast,
                    standardized_coefficient=np.nan,
                    odds_ratio=np.exp(contrast),
                    training_scale=np.nan,
                    interpretation="Within-field category contrast, not coefficient vs intercept",
                )
            )
        offset += len(levels)
    assert offset == len(coefficients)
    return pd.DataFrame(rows)


def group_linear_values(pipeline, values):
    result = np.zeros((len(values), len(FEATURES)))
    result[:, : len(NUMERIC)] = values[:, : len(NUMERIC)]
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    offset = len(NUMERIC)
    for index, levels in enumerate(encoder.categories_, start=len(NUMERIC)):
        result[:, index] = values[:, offset : offset + len(levels)].sum(axis=1)
        offset += len(levels)
    np.testing.assert_allclose(result.sum(axis=1), values.sum(axis=1), atol=1e-10)
    return result


def tree_adapter_check(model, values):
    """Never trust TreeSHAP's adapter-internal check for unsupported categorical routing."""
    try:
        explanation = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", model_output="raw"
        )(values)
        error = float(
            np.max(
                np.abs(
                    explanation.values.sum(axis=1)
                    + explanation.base_values
                    - model.decision_function(values)
                )
            )
        )
        return {
            "accepted": bool(error < 1e-7),
            "maximum_log_odds_error": error,
            "rows_checked": len(values),
        }
    except (ValueError, TypeError, NotImplementedError, AttributeError) as error:
        return {
            "accepted": False,
            "reason": f"{type(error).__name__}: {str(error)[:200]}",
            "rows_checked": len(values),
        }


def select_cases(frame, probability, selected):
    """Purposive validation case examples; no identities or source rows are exported."""
    p, y = np.asarray(probability), frame.y.to_numpy()
    masks = {
        "targeted_positive": selected & (y == 1),
        "targeted_negative": selected & (y == 0),
        "missed_positive_no_prior_inpatient": ~selected
        & (y == 1)
        & frame.number_inpatient.eq(0).to_numpy(),
    }
    result = {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"No validation example for {name}")
        # Closest to this category's median risk; original frozen row order breaks equal distances.
        result[name] = int(indices[np.argmin(np.abs(p[indices] - np.median(p[indices])))])
    return result


def explain(root, policy, frame, models, fitted):
    out = root / policy["report_dir"] / "validation"
    artifacts = root / policy["artifact_dir"]
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "explanation_manifest.json"
    expected = {
        "implementation_sha256": sha256(Path(__file__)),
        "validation_prediction_sha256": json.loads(
            (root / "models/development/phase4/validation_scores.json").read_text()
        )["prediction_sha256"],
        "settings": policy["explanations"],
        "models": {name: fitted["models"][name]["sha256"] for name in models},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["inputs"] != expected:
            raise ValueError("Explanation inputs changed; do not silently replace frozen analysis")
        return previous
    train = load_partition(root, "train", "calibration")
    rng = np.random.default_rng(policy["random_seed"])
    bg = np.sort(rng.choice(len(train), policy["explanations"]["background_rows"], replace=False))
    compare = np.sort(
        rng.choice(len(frame), policy["explanations"]["comparison_rows"], replace=False)
    )
    logistic, boosting = models[policy["primary"]], models[policy["challenger"]]
    x = dense(logistic[:-1].transform(frame))
    background = dense(logistic[:-1].transform(train.iloc[bg]))
    masker = shap.maskers.Independent(background, max_samples=len(background))
    linear = shap.LinearExplainer(logistic[-1], masker)
    values = linear(x)
    grouped = group_linear_values(logistic, values.values)
    with threadpool_limits(limits=policy["threads"]):
        margin = logistic[-1].decision_function(x)
    linear_error = float(np.max(np.abs(values.base_values + grouped.sum(axis=1) - margin)))
    np.testing.assert_allclose(values.base_values + grouped.sum(axis=1), margin, atol=1e-9)
    np.testing.assert_allclose(expit(margin), frame[policy["primary"]], rtol=0, atol=1e-8)

    bx = dense(boosting[:-1].transform(frame.iloc[compare]))
    bbg = dense(boosting[:-1].transform(train.iloc[bg]))
    with threadpool_limits(limits=policy["threads"]):
        tree_check = tree_adapter_check(boosting[-1], bx[:32])
        # Always use the prespecified independent-field permutation comparison. The adapter
        # audit explains why direct TreeSHAP is not used for this native-categorical model.
        permutation = shap.PermutationExplainer(
            boosting[-1].decision_function,
            shap.maskers.Independent(bbg, max_samples=len(bbg)),
            seed=policy["random_seed"],
            feature_names=FEATURES,
        )
        boost_values = permutation(
            bx,
            max_evals=(2 * len(FEATURES) + 1) * policy["explanations"]["permutation_repeats"],
            silent=True,
        )
        boost_margin = boosting[-1].decision_function(bx)
    boost_error = float(
        np.max(np.abs(boost_values.base_values + boost_values.values.sum(axis=1) - boost_margin))
    )
    np.testing.assert_allclose(
        boost_values.base_values + boost_values.values.sum(axis=1), boost_margin, atol=1e-7
    )
    np.testing.assert_allclose(
        expit(boost_margin), frame[policy["challenger"]].iloc[compare], rtol=0, atol=1e-8
    )
    global_values = pd.DataFrame(
        {
            "feature": FEATURES,
            "logistic_mean_absolute_log_odds": np.abs(grouped).mean(axis=0),
            "logistic_matched_mean_absolute_log_odds": np.abs(grouped[compare]).mean(axis=0),
            "boosting_matched_mean_absolute_log_odds": np.abs(boost_values.values).mean(axis=0),
        }
    )
    global_values.to_csv(out / "global_feature_influence.csv", index=False)
    coefficient_table(logistic, policy["explanations"]["categorical_references"]).to_csv(
        out / "coefficient_contrasts.csv", index=False
    )
    directions = []
    for index, name in enumerate(NUMERIC):
        raw = frame[name].iloc[compare].to_numpy()
        directions.append(
            dict(
                feature=name,
                logistic_coefficient_per_unit=logistic[-1].coef_[0, index]
                / logistic.named_steps["preprocess"].named_transformers_["numeric"].scale_[index],
                logistic_shap_rank_association=spearmanr(raw, grouped[compare, index]).statistic,
                boosting_shap_rank_association=spearmanr(
                    raw, boost_values.values[:, index]
                ).statistic,
            )
        )
    pd.DataFrame(directions).to_csv(out / "explanation_direction_checks.csv", index=False)
    selected = target_mask(
        frame[policy["primary"]], frame.encounter_id, policy["targeting"]["fraction"]
    )
    cases = select_cases(frame, frame[policy["primary"]], selected)
    case_rows = []
    for number, (name, index) in enumerate(cases.items(), 1):
        for j, feature in enumerate(FEATURES):
            case_rows.append(
                dict(
                    case=f"Example {number}",
                    case_type=name,
                    feature=feature,
                    contribution_log_odds=grouped[index, j],
                    base_log_odds=float(values.base_values[index]),
                    probability=float(frame[policy["primary"]].iloc[index]),
                    observed_readmission=int(frame.y.iloc[index]),
                )
            )
    pd.DataFrame(case_rows).to_csv(out / "individual_explanations.csv", index=False)
    # Individual arrays, indices and case-source mapping remain ignored local artifacts.
    np.savez_compressed(
        artifacts / "explanations.npz",
        logistic=grouped,
        boosting=boost_values.values,
        comparison_indices=compare,
    )
    record = dict(
        inputs=expected,
        logistic_maximum_additivity_error=linear_error,
        boosting_maximum_additivity_error=boost_error,
        tree_adapter_check=tree_check,
        linear_method="LinearExplainer, independent training background, grouped one-hot sums",
        boosting_method="PermutationExplainer of actual categorical decision_function",
        logistic_rows=len(frame),
        boosting_rows=len(compare),
        background_rows=len(bg),
        units="log odds",
        partition="validation",
        test_accessed=False,
    )
    write_json(manifest_path, record)
    write_json(out / "explanation_verification.json", record)
    return record
