"""Validation-only operational analysis and explanations; this module cannot open test."""

import json
import logging

from readmit_iq.analysis.reporting import write_json
from readmit_iq.decision_support.context import development_data, settings
from readmit_iq.decision_support.evaluation import evaluate_predictions
from readmit_iq.decision_support.explainability import explain


def run():
    root, policy = settings()
    frame, models, fitted, scoring, training = development_data()
    out = root / policy["report_dir"] / "validation"
    out.mkdir(parents=True, exist_ok=True)
    logging.info("Analyzing cached validation predictions at the 10%% portfolio capacity")
    summary = evaluate_predictions(
        frame, {name: frame[name].to_numpy() for name in models}, policy, out, "validation"
    )
    logging.info("Explaining the unchanged models using training background and validation cases")
    explanations = explain(root, policy, frame, models, fitted)
    record = dict(
        phase=5,
        partition="validation",
        test_records_loaded=False,
        source_validation_prediction_sha256=scoring["prediction_sha256"],
        training_git_commit=training["git_commit"],
        model_sha256={name: fitted["models"][name]["sha256"] for name in models},
        explanations=explanations,
        primary=summary["primary"],
        targeting=policy["targeting"],
    )
    write_json(out / "development_record.json", record)
    print(
        json.dumps(
            {
                "primary": summary["primary"],
                "policy": policy["targeting"]["fraction"],
                "validation_metrics": summary["metrics"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
