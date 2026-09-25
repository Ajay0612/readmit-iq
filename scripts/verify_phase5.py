"""Reconcile Phase 5 evidence without opening test records or invoking real inference."""

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np

from readmit_iq.analysis.reporting import write_json
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.decision_support.final_evaluation import verify_published_results
from readmit_iq.decision_support.plots import read_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/modeling/phase5"
    development = json.loads((out / "validation/development_record.json").read_text())
    assert development["test_records_loaded"] is False
    val = read_table(root, "validation", "metrics", True).iloc[0]
    np.testing.assert_allclose(val.average_precision, 0.1967806996565404, atol=1e-12, rtol=0)
    assert (val.encounters, val.tp, val.fn, val.targeted_encounters) == (13590, 321, 1178, 1359)
    explanation = json.loads((out / "validation/explanation_verification.json").read_text())
    assert explanation["partition"] == "validation" and explanation["test_accessed"] is False
    assert explanation["logistic_maximum_additivity_error"] < 1e-8
    assert explanation["boosting_maximum_additivity_error"] < 1e-8
    partitions, notebooks = ["validation"], ["05_explainability_business_impact.ipynb"]
    if not args.development_only:
        final = verify_published_results(root)
        partitions += ["test"]
        notebooks += ["06_final_test_evaluation.ipynb"]
        binary = root / "models/final/readmit_iq_logistic.joblib"
        if binary.exists():
            metadata = json.loads((root / "reports/modeling/final_model_metadata.json").read_text())
            verify_file(binary, metadata["pipeline_sha256"])
        assert final["scoring"]["encounters"] == 13549
    for partition in partitions:
        metric = read_table(root, partition, "metrics", True).iloc[0]
        deciles = read_table(root, partition, "risk_deciles", True)
        assert deciles.encounters.sum() == metric.encounters
        assert deciles.positives.sum() == metric.positives
        assert deciles.iloc[0].positives == metric.tp
        assert metric.tp + metric.fn == metric.positives
        assert metric.tp + metric.fp == metric.targeted_encounters
        assert metric.targeted_patients <= metric.targeted_encounters
        groups = read_table(root, partition, "subgroups", True)
        for name in ["age", "gender", "race", "prior_inpatient"]:
            assert groups.loc[groups.group.eq(name), "encounters"].sum() == metric.encounters
        suppressed = groups.loc[groups.suppressed]
        assert suppressed.average_precision.isna().all()
        assert suppressed.recall.isna().all()
        intervals = read_table(root, partition, "metric_intervals")
        assert intervals.valid_replicates.eq(1000).all()
    execution = {}
    for name in notebooks:
        notebook = nbformat.read(root / "notebooks" / name, as_version=4)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert cells and all(cell.execution_count is not None for cell in cells)
        outputs = [output for cell in cells for output in cell.outputs]
        assert not any(output.output_type == "error" for output in outputs)
        assert not any("/Users/ajay/" in str(output) for output in outputs)
        images = sum("image/png" in output.get("data", {}) for output in outputs)
        assert images >= 8
        execution[name] = dict(code_cells=len(cells), embedded_images=images, errors=0)
    figures = sorted((root / "reports/figures/phase5").glob("*.png"))
    assert len(figures) == 8 and all(path.stat().st_size > 10000 for path in figures)
    record = dict(
        validation_reproduces_phase4=True,
        original_test_opened_by_verifier=False,
        model_predictions_by_verifier=0,
        archived_final_verified=not args.development_only,
        notebooks=execution,
        figure_sha256={path.name: sha256(path) for path in figures},
        checks="Frozen policy/hash provenance, clustered intervals, totals and suppression",
    )
    write_json(out / "verification.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
