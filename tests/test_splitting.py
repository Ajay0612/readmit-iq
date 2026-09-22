"""Patient isolation, deterministic allocation, balance and test-freeze regression checks."""

import pandas as pd
import pytest

from readmit_iq.data.cohort import TARGET
from readmit_iq.modeling.splitting import (
    freeze_partitions,
    load_development,
    split_patients,
    validate_split,
)

POLICY = dict(
    train_fraction=0.7, validation_fraction=0.15, test_fraction=0.15, prevalence_tolerance=0.03
)


@pytest.fixture
def population():
    rows = []
    for patient in range(1600):
        for encounter in range(1 + patient % 4):
            rows.append(
                dict(
                    patient_nbr=f"p{patient:04}",
                    encounter_id=f"e{patient:04}-{encounter}",
                    **{TARGET: int(patient % 10 == 0 and encounter == 0)},
                )
            )
    return pd.DataFrame(rows)


def test_patient_isolation_and_complete_coverage(population):
    assignments = split_patients(population, POLICY)
    table, overlap = validate_split(population, assignments, POLICY["prevalence_tolerance"])
    assert overlap == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert table.encounters.sum() == len(population)
    assert table.patients.sum() == population.patient_nbr.nunique()
    assert assignments.encounter_id.is_unique
    assert assignments.groupby("patient_nbr").partition.nunique().eq(1).all()
    assert (table.prevalence_pct / 100 - population[TARGET].mean()).abs().max() <= 0.03
    assert table.patient_pct.to_list() == pytest.approx([70, 15, 15], abs=0.15)


def test_deterministic_even_if_source_rows_reordered(population):
    first = split_patients(population, POLICY)
    reordered = split_patients(population.sample(frac=1, random_state=7), POLICY)
    pd.testing.assert_frame_equal(first, reordered)


def test_overlap_and_missing_encounter_are_rejected(population):
    a = split_patients(population, POLICY)
    with pytest.raises(ValueError, match="exactly once"):
        validate_split(population, a.iloc[1:], 1)
    patient = population.patient_nbr.value_counts().idxmax()
    ix = a.index[a.patient_nbr.eq(patient)][0]
    a.loc[ix, "partition"] = "test" if a.loc[ix, "partition"] != "test" else "train"
    with pytest.raises(ValueError, match="overlap"):
        validate_split(population, a, 1)


def test_unbalanced_allocation_fails_prespecified_tolerance(population):
    a = split_patients(population, POLICY)
    with pytest.raises(ValueError, match="Prevalence tolerance"):
        validate_split(population, a, 0)


def test_frozen_artifacts_cannot_be_overwritten(population, tmp_path):
    lock = freeze_partitions(population, tmp_path, POLICY, 42, {"source": "test"})
    assert lock["patient_overlap"]["train_test"] == 0
    assert len(lock["files"]) == 4
    with pytest.raises(FileExistsError, match="already"):
        freeze_partitions(population, tmp_path, POLICY, 43, {})


def test_compressed_artifacts_reproduce_exactly(population, tmp_path):
    a = freeze_partitions(population, tmp_path / "a", POLICY, 42, {})
    b = freeze_partitions(population, tmp_path / "b", POLICY, 42, {})
    assert a["files"] == b["files"]


def test_development_loader_refuses_test():
    with pytest.raises(ValueError, match="locked"):
        load_development("test")


def test_missing_patient_key_rejected(population):
    population.loc[0, "patient_nbr"] = pd.NA
    with pytest.raises(ValueError, match="patient keys"):
        split_patients(population, POLICY)


def test_full_cohort_eda_is_blocked_after_freeze(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from readmit_iq.analysis import run_phase2

    lock = tmp_path / "data/processed/phase3/lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text("{}")
    monkeypatch.setattr(
        run_phase2, "load_config", lambda: SimpleNamespace(raw_dir=tmp_path / "data/raw")
    )
    with pytest.raises(RuntimeError, match="test is frozen"):
        run_phase2.main()
