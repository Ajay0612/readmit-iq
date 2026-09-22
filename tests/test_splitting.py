"""Patient isolation, deterministic allocation, balance and test-freeze regression checks."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from readmit_iq.data.cohort import TARGET
from readmit_iq.data.download import sha256
from readmit_iq.modeling import splitting
from readmit_iq.modeling.splitting import (
    compressed_csv,
    freeze_partitions,
    load_development,
    prepare_frozen_partitions,
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


@pytest.mark.parametrize("host_os_byte", [3, 19, 255])
def test_compressed_csv_preserves_frozen_bytes_across_hosts(tmp_path, monkeypatch, host_os_byte):
    native_compress = gzip.compress

    def host_compress(*args, **kwargs):
        data = native_compress(*args, **kwargs)
        return data[:9] + bytes([host_os_byte]) + data[10:]

    monkeypatch.setattr(splitting.gzip, "compress", host_compress)
    frame = pd.DataFrame(
        {"category": ["café", "Unknown"], "value": pd.array([1, None], dtype="Int64")}
    )
    path = tmp_path / "portable.csv.gz"
    compressed_csv(frame, path)
    assert gzip.decompress(path.read_bytes()) == "category,value\ncafé,1\nUnknown,\n".encode()
    assert sha256(path) == "bf6535eb15a0b8a89d2e77bd96f7e5e7c9b9eb631f360ad80328d62fcbc99d5d"


@pytest.fixture
def fresh_checkout(population, tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    for folder in ["configs", "data/raw", "reports/eda", "reports/modeling"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "configs/phase2.yaml").write_text("cohort: {}\n")
    (root / "data/raw/diabetic_data.csv").write_text("verified synthetic source\n")
    source_hashes = {"diabetic_data.csv": sha256(root / "data/raw/diabetic_data.csv")}
    policy = {
        **POLICY,
        "directory": "data/processed/phase3",
        "lock_report": "reports/modeling/split_manifest.json",
    }
    (root / "reports/eda/summary.json").write_text(
        json.dumps(
            {
                "primary": {
                    "encounters": len(population),
                    "patients": population.patient_nbr.nunique(),
                    "positives": int(population[TARGET].sum()),
                }
            }
        )
    )
    prior = freeze_partitions(
        population,
        tmp_path / "original",
        policy,
        42,
        {
            "source_sha256": source_hashes,
            "cohort_policy_sha256": sha256(root / "configs/phase2.yaml"),
        },
    )
    report = root / policy["lock_report"]
    report.write_text(json.dumps(prior, indent=2) + "\n")
    monkeypatch.setattr(splitting, "settings", lambda: (root, {"split": policy, "random_seed": 42}))
    monkeypatch.setattr(
        splitting,
        "load_config",
        lambda: SimpleNamespace(
            raw_dir=root / "data/raw", dataset=SimpleNamespace(file_sha256=source_hashes)
        ),
    )
    monkeypatch.setattr(splitting, "load_raw", lambda _: population)
    monkeypatch.setattr(splitting, "load_phase2_config", lambda _: {"cohort": {}})
    monkeypatch.setattr(splitting, "build_cohort", lambda raw, _: (raw, None))
    return root, report, prior


def test_fresh_checkout_reproduces_original_contract_then_only_verifies(
    fresh_checkout, monkeypatch
):
    root, report, prior = fresh_checkout
    original_report = report.read_bytes()
    assert prepare_frozen_partitions() == prior
    directory = root / "data/processed/phase3"
    assert json.loads((directory / "lock.json").read_text()) == prior
    assert {name: sha256(directory / name) for name in prior["files"]} == prior["files"]
    assert report.read_bytes() == original_report

    def reject_reload(*_):
        pytest.fail("An existing freeze must be verified without loading source records")

    monkeypatch.setattr(splitting, "load_raw", reject_reload)
    assert prepare_frozen_partitions() == prior


@pytest.mark.parametrize(
    "field",
    [
        "files",
        "partitions",
        "provenance",
        "seed",
        "split_policy",
        "patient_overlap",
        "eligible_encounters",
        "eligible_patients",
        "method",
        "test_status",
    ],
)
def test_manifest_mismatch_cannot_publish_or_overwrite_a_freeze(fresh_checkout, field, caplog):
    root, report, prior = fresh_checkout
    prior[field] = "deliberately mismatched contract"
    report.write_text(json.dumps(prior))
    original_report = report.read_bytes()
    for _ in range(2):  # A failed first attempt must not poison the next verification.
        with pytest.raises(
            ValueError, match="Reproduction does not match committed split manifest"
        ):
            prepare_frozen_partitions()
        assert report.read_bytes() == original_report
        assert not (root / "data/processed/phase3").exists()
        assert not list((root / "data/processed").glob(".verify-split-*"))
    assert field in caplog.text


def test_existing_freeze_rejects_tampered_compressed_bytes(fresh_checkout):
    root, _, _ = fresh_checkout
    prepare_frozen_partitions()
    path = root / "data/processed/phase3/train.csv.gz"
    original = path.read_bytes()
    path.write_bytes(original[:9] + b"\x03" + original[10:])
    assert hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare_frozen_partitions()


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
