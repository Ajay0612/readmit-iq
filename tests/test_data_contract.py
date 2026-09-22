"""Guard silent changes to raw tokens, labels, source identity and patient keys."""

import hashlib
import zipfile

import pandas as pd
import pytest

from readmit_iq.config import load_config
from readmit_iq.data.download import extract_verified, verify_file
from readmit_iq.data.load_data import load_id_mapping, load_raw
from readmit_iq.data.validate import binary_target, validate_raw


def test_loader_preserves_not_measured_and_missing_markers(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text(
        "encounter_id,patient_nbr,diag_1,diag_2,diag_3,A1Cresult,max_glu_serum,race\n"
        "001,002,250.00,V45,E849,None,None,?\n"
    )
    frame = load_raw(source)
    assert frame.loc[0, "A1Cresult"] == "None"
    assert frame.loc[0, "race"] == "?"
    assert frame.loc[0, "encounter_id"] == "001"
    assert frame.loc[0, "diag_1"] == "250.00"
    assert frame.isna().sum().sum() == 0


def test_target_uses_less_than_30_only():
    assert binary_target(pd.Series(["NO", "<30", ">30"])).tolist() == [0, 1, 0]


@pytest.mark.parametrize("label", ["", "?", None, "30", "No", "unknown"])
def test_unknown_target_is_not_silently_negative(label):
    with pytest.raises(ValueError, match="undocumented"):
        binary_target(pd.Series(["<30", label]))


def test_checksum_rejects_changed_file(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("changed bytes")
    with pytest.raises(ValueError, match="Checksum"):
        verify_file(path, "0" * 64)


def test_archive_rejects_unexpected_paths(tmp_path):
    path = tmp_path / "test.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("../outside.csv", "bad")
    with pytest.raises(ValueError, match="contents"):
        extract_verified(path, tmp_path, {"expected.csv": "0" * 64})
    assert not (tmp_path.parent / "outside.csv").exists()


def test_archive_does_not_install_bad_member(tmp_path):
    path = tmp_path / "test.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("expected.csv", "bad")
    with pytest.raises(ValueError, match="Checksum"):
        extract_verified(path, tmp_path, {"expected.csv": hashlib.sha256(b"good").hexdigest()})
    assert not (tmp_path / "expected.csv").exists()


@pytest.fixture
def small_contract_frame():
    config = load_config()
    config.dataset.expected_rows = 2
    record = dict.fromkeys(config.dataset.expected_columns, "No")
    record.update({name: limits[0] for name, limits in config.numeric_bounds.items()})
    record.update(
        encounter_id="1",
        patient_nbr="5",
        readmitted="NO",
        age="[50-60)",
        admission_type_id=1,
        discharge_disposition_id=1,
        admission_source_id=1,
    )
    frame = pd.DataFrame([record, {**record, "encounter_id": "2", "readmitted": "<30"}])
    mapping = {
        name: {1: "Known"}
        for name in [
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
        ]
    }
    return frame, config, mapping


def test_repeated_patient_is_allowed_but_duplicate_encounter_is_not(small_contract_frame):
    frame, config, mapping = small_contract_frame
    assert validate_raw(frame, config, mapping)["passed"]
    frame.loc[1, "encounter_id"] = "1"
    report = validate_raw(frame, config, mapping)
    assert not report["passed"]
    assert report["duplicate_encounter_ids"] == 1


def test_negative_counts_fail_source_validation(small_contract_frame):
    frame, config, mapping = small_contract_frame
    frame.loc[0, "number_inpatient"] = -1
    report = validate_raw(frame, config, mapping)
    assert not report["passed"]
    assert report["invalid_numeric_counts"]["number_inpatient"] == 1


def test_mapping_retains_null_description_and_quoted_comma(tmp_path):
    path = tmp_path / "mapping.csv"
    path.write_text(
        "admission_type_id,description\n6,NULL\n,\n"
        'discharge_disposition_id,description\n19,"Expired, at home"\n,\n'
        "admission_source_id,description\n1, Physician Referral\n"
    )
    mapping = load_id_mapping(path)
    assert mapping["admission_type_id"][6] == "NULL"
    assert mapping["discharge_disposition_id"][19] == "Expired, at home"
