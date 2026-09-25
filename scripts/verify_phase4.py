"""Verify completed Phase 4 artifacts without fitting models or reopening partition tables."""

import argparse
import base64
import hashlib
import json
import platform
import xml.etree.ElementTree as ET

import nbformat
import numpy as np
import pandas as pd

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256
from readmit_iq.modeling.metrics import classification_metrics
from readmit_iq.optimization.contract import settings, verify_contract
from readmit_iq.optimization.evaluate import score_validation_once
from readmit_iq.optimization.tracking import LocalTracker


def verify(pytest_report: str) -> dict:
    root, policy = settings()
    artifacts, out = root / policy["artifact_dir"], root / policy["report_dir"]
    # Require completed caches before calling the accessor: verification cannot start a fit/score.
    for filename in ["training_selection.json", "fit_manifest.json", "validation_scores.json"]:
        if not (artifacts / filename).is_file():
            raise FileNotFoundError(f"Complete Phase 4 execution first: missing {filename}")
    frozen = verify_contract(root)
    scored, scoring, fitted = score_validation_once()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scoring_manifest"] == scoring
    assert scoring["score_passes"] == 1 and scoring["evaluated_partition"] == "validation"
    assert not scoring["test_evaluated"] and not summary["test_evaluated"]
    assert not summary["threshold_selected"] and not summary["production_model"]
    assert len(scored) == 13590 and scored.patient_nbr.nunique() == 9757
    assert all(item["fit_encounters"] == 63563 for item in fitted["models"].values())
    metrics = pd.read_csv(out / "validation_metrics.csv").set_index("model")
    for name in scoring["prediction_columns"]:
        actual = classification_metrics(scored.y.to_numpy(), scored[name].to_numpy())
        for metric, value in actual.items():
            np.testing.assert_allclose(metrics.loc[name, metric], value, rtol=0, atol=1e-12)

    # CV aggregate coverage is separate from the unit tests of group membership and ordering.
    folds = pd.read_csv(out / "cv_folds.csv")
    assert len(folds) == 10 and folds.patient_overlap.eq(0).all()
    assert folds.loc[folds.role.eq("holdout"), "encounters"].sum() == 63563
    tracker = LocalTracker(root, policy, summary["training_git_commit"])
    runs = tracker.client.search_runs([tracker.experiment_id], max_results=1000)
    validation_runs = summary["mlflow_validation_runs"]
    assert len(validation_runs) == len(fitted["models"]) == 7
    for name, run_id in validation_runs.items():
        run = tracker.client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["git_commit"] == summary["training_git_commit"]
        assert run.data.tags["seed"] == "42"
        assert run.data.params["model_sha256"] == fitted["models"][name]["sha256"]
        assert run.data.params["artifact_path"] == f"{policy['artifact_dir']}/{name}.joblib"
        for metric in ["average_precision", "roc_auc", "brier"]:
            np.testing.assert_allclose(
                run.data.metrics[f"validation_{metric}"],
                metrics.loc[name, metric],
                rtol=0,
                atol=1e-12,
            )

    notebook_path = root / "notebooks/04_model_optimization.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 13
    assert [cell.execution_count for cell in code_cells] == list(range(1, 14))
    outputs = [output for cell in code_cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    embedded = {
        hashlib.sha256(base64.b64decode(output.data["image/png"])).hexdigest()
        for output in outputs
        if "image/png" in output.get("data", {})
    }
    figures = sorted((root / policy["figure_dir"]).glob("*.png"))
    figure_hashes = {str(path.relative_to(root)): sha256(path) for path in figures}
    assert len(figures) == 6 and set(figure_hashes.values()) == embedded

    xml = ET.parse(root / pytest_report).getroot()
    suites = list(xml.iter("testsuite"))
    tests = sum(int(suite.attrib["tests"]) for suite in suites)
    assert tests >= 158, "The original suite and all Phase 4 protections must run"
    for status in ["failures", "errors", "skipped"]:
        assert sum(int(suite.attrib.get(status, 0)) for suite in suites) == 0
    record = dict(
        phase=4,
        scope="Development validation only; no test partition parsing or scoring",
        training_git_commit=summary["training_git_commit"],
        environment={
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        frozen_files=frozen["files"],
        frozen_manifest_sha256=sha256(root / "reports/modeling/split_manifest.json"),
        patient_overlap=frozen["patient_overlap"],
        training_fold_membership_sha256=fitted["fold_membership_sha256"],
        saved_models_verified=len(fitted["models"]),
        validation_prediction_sha256=scoring["prediction_sha256"],
        validation_score_passes=scoring["score_passes"],
        cached_metrics_reconciled=True,
        mlflow_runs=len(runs),
        mlflow_validation_runs_verified=len(validation_runs),
        tests_passed=tests,
        test_report_sha256=sha256(root / pytest_report),
        notebook=str(notebook_path.relative_to(root)),
        notebook_sha256=sha256(notebook_path),
        executed_code_cells=len(code_cells),
        notebook_error_outputs=0,
        notebook_images_match_saved_figures=True,
        figures=figure_hashes,
        test_evaluated=False,
        operational_threshold_selected=False,
        production_model=False,
    )
    write_json(out / "verification.json", record)
    print(
        f"Verified {tests} passing tests, 13 notebook cells, six embedded figures and seven models."
    )
    print("Frozen hashes unchanged; validation predictions reused; test unscored.")
    print(
        metrics.loc[
            [summary["primary"], summary["challenger"]], ["average_precision", "roc_auc", "brier"]
        ].to_string()
    )
    print("Primary:", summary["primary"], "Challenger:", summary["challenger"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-report", default=".cache/phase4-tests.xml")
    verify(parser.parse_args().pytest_report)
