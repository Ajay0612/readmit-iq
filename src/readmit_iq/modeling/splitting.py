"""Deterministic patient allocation, immutable partition artifacts and development-only loading."""

import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from readmit_iq.analysis.reporting import markdown_table, write_json
from readmit_iq.config import load_config
from readmit_iq.data.cohort import TARGET, build_cohort, load_phase2_config
from readmit_iq.data.download import sha256, verify_file
from readmit_iq.data.load_data import load_raw
from readmit_iq.modeling.settings import settings

LOGGER = logging.getLogger(__name__)
PARTITIONS = ("train", "validation", "test")
STRING_COLUMNS = ["encounter_id", "patient_nbr", "diag_1", "diag_2", "diag_3"]


def split_patients(frame: pd.DataFrame, policy: dict, seed: int = 42) -> pd.DataFrame:
    """Allocate sorted unique patients using outcome × size strata, never encounter rows."""
    if frame.encounter_id.isna().any() or frame.encounter_id.duplicated().any():
        raise ValueError("Encounter keys must be complete and unique")
    if frame.patient_nbr.isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Complete patient keys and binary outcomes are required")
    fractions = [policy[f"{name}_fraction"] for name in PARTITIONS]
    if abs(sum(fractions) - 1) > 1e-10 or any(f <= 0 for f in fractions):
        raise ValueError("Three positive fractions must sum to one")
    patients = frame.groupby("patient_nbr", sort=True)[TARGET].agg(any_positive="max", n="size")
    strata = patients.any_positive.astype(str) + ":" + patients.n.clip(upper=4).astype(str)
    # Count-based fallback for small synthetic/new populations, not outcome-performance search.
    if strata.value_counts().min() < 10:
        strata = patients.any_positive.astype(str)
    if strata.value_counts().min() < 10:
        raise ValueError("Insufficient patients per outcome stratum for a reliable three-way split")
    development, test = train_test_split(
        patients.index, test_size=fractions[2], random_state=seed, stratify=strata
    )
    train, validation = train_test_split(
        development,
        test_size=fractions[1] / (1 - fractions[2]),
        random_state=seed,
        stratify=strata.loc[development],
    )
    allocation = pd.concat(
        [
            pd.Series(part, index=ids)
            for part, ids in zip(PARTITIONS, [train, validation, test], strict=True)
        ]
    )
    result = frame[["encounter_id", "patient_nbr"]].copy()
    result["partition"] = result.patient_nbr.map(allocation)
    return result.sort_values("encounter_id").reset_index(drop=True)


def validate_split(
    frame: pd.DataFrame, assignments: pd.DataFrame, tolerance: float
) -> tuple[pd.DataFrame, dict]:
    """Check complete coverage, group isolation and prespecified absolute prevalence tolerance."""
    if assignments.encounter_id.duplicated().any() or set(assignments.encounter_id) != set(
        frame.encounter_id
    ):
        raise ValueError("Every eligible encounter must occur exactly once")
    joined = frame[["encounter_id", "patient_nbr", TARGET]].merge(
        assignments, on=["encounter_id", "patient_nbr"], validate="one_to_one", how="left"
    )
    if joined.partition.isna().any() or set(joined.partition) != set(PARTITIONS):
        raise ValueError("Assignments must cover every patient with valid partitions")
    if joined.groupby("patient_nbr").partition.nunique().ne(1).any():
        raise ValueError("Patient overlap across partitions is forbidden")
    rows = []
    for part in PARTITIONS:
        data = joined.loc[joined.partition.eq(part)]
        positives = int(data[TARGET].sum())
        if not 0 < positives < len(data):
            raise ValueError("Both target classes must occur in every partition")
        prevalence = positives / len(data)
        if abs(prevalence - frame[TARGET].mean()) > tolerance:
            raise ValueError(
                "Prevalence tolerance exceeded; review allocation, never search model scores"
            )
        rows.append(
            dict(
                partition=part,
                encounters=len(data),
                patients=data.patient_nbr.nunique(),
                positives=positives,
                negatives=len(data) - positives,
                prevalence_pct=100 * prevalence,
                encounter_pct=100 * len(data) / len(frame),
                patient_pct=100 * data.patient_nbr.nunique() / frame.patient_nbr.nunique(),
            )
        )
    sets = {p: set(joined.loc[joined.partition.eq(p), "patient_nbr"]) for p in PARTITIONS}
    overlap = {
        f"{a}_{b}": len(sets[a] & sets[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    return pd.DataFrame(rows), overlap


def compressed_csv(frame: pd.DataFrame, path: Path) -> None:
    """Portable deterministic gzip CSV; no new Parquet dependency or duplicate Git data."""
    path.write_bytes(gzip.compress(frame.to_csv(index=False, na_rep="").encode(), mtime=0))


def freeze_partitions(
    frame: pd.DataFrame, directory: Path, policy: dict, seed: int, provenance: dict
) -> dict:
    """Create once. A nonempty destination is never silently overwritten or reallocated."""
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Frozen partition destination already contains artifacts")
    assignments = split_patients(frame, policy, seed)
    table, overlap = validate_split(frame, assignments, policy["prevalence_tolerance"])
    lookup = assignments.set_index("encounter_id").partition
    for part in PARTITIONS:
        compressed_csv(
            frame.loc[frame.encounter_id.map(lookup).eq(part)].sort_values("encounter_id"),
            directory / f"{part}.csv.gz",
        )
    compressed_csv(assignments, directory / "assignments.csv.gz")
    lock = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        seed=seed,
        method=(
            "Sorted unique patients; any-positive x capped encounter-count strata; "
            "two stratified splits"
        ),
        split_policy=policy,
        provenance=provenance,
        partitions=table.to_dict("records"),
        patient_overlap=overlap,
        eligible_encounters=len(frame),
        eligible_patients=frame.patient_nbr.nunique(),
        files={p.name: sha256(p) for p in sorted(directory.glob("*.csv.gz"))},
        test_status=(
            "LOCKED: aggregate split diagnostics only; no test predictions or development EDA"
        ),
    )
    write_json(directory / "lock.json", lock)
    return lock


def prepare_frozen_partitions() -> dict:
    """Reconcile Phase 2 once, or verify existing artifact bytes without opening test records."""
    root, policy = settings()
    source = load_config()
    directory = root / policy["split"]["directory"]
    lock_path = directory / "lock.json"
    report_path = root / policy["split"]["lock_report"]
    provenance = dict(
        source_sha256=source.dataset.file_sha256,
        cohort_policy_sha256=sha256(root / "configs/phase2.yaml"),
    )
    for name, expected in source.dataset.file_sha256.items():
        verify_file(source.raw_dir / name, expected)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if (
            lock["provenance"] != provenance
            or lock["split_policy"] != policy["split"]
            or lock["seed"] != policy["random_seed"]
        ):
            raise ValueError("Frozen split provenance/policy changed; refusing silent resplit")
        for name, expected in lock["files"].items():
            verify_file(directory / name, expected)
        if report_path.exists() and json.loads(report_path.read_text()) != lock:
            raise ValueError("Tracked split manifest disagrees with frozen lock")
        LOGGER.info("Verified frozen partition hashes; no test table opened")
        return lock
    raw = load_raw(source.raw_dir / "diabetic_data.csv")
    cohort, _ = build_cohort(raw, load_phase2_config(root / "configs/phase2.yaml")["cohort"])
    expected = json.loads((root / "reports/eda/summary.json").read_text())["primary"]
    actual = dict(
        encounters=len(cohort),
        patients=cohort.patient_nbr.nunique(),
        positives=int(cohort[TARGET].sum()),
    )
    if any(actual[k] != expected[k] for k in actual):
        raise ValueError("Phase 2 cohort does not reconcile")
    lock = freeze_partitions(cohort, directory, policy["split"], policy["random_seed"], provenance)
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if (
            prior["files"] != lock["files"]
            or prior["partitions"] != lock["partitions"]
            or prior["provenance"] != provenance
        ):
            raise ValueError("Reproduction does not match committed split manifest")
        # Preserve the original freeze date when reconstructing exactly the same partitions.
        lock = prior
        write_json(lock_path, lock)
    else:
        write_json(report_path, lock)
    report = "# Frozen patient-grouped partitions\n\n" + markdown_table(
        pd.DataFrame(lock["partitions"])
    )
    report += (
        "\n\nAll three pairwise patient overlaps are **zero**. Every eligible "
        "patient and encounter occurs in one partition. "
    )
    report += (
        "Seed 42; outcome × encounter-count patient strata; absolute "
        "prevalence tolerance 1 percentage point, fixed before allocation. "
    )
    report += "No seed search. Groups take precedence over exact row proportions.\n\n"
    report += (
        "The test is locked. Its labels were used only for allocation "
        "diagnostics; no test features, predictions or performance guide Phase "
        "3. "
    )
    report += (
        "Full-cohort Phase 2 EDA previously viewed outcomes indirectly; this "
        "is not a fully unseen confirmatory sample. "
    )
    report += "No timestamps support temporal validation.\n\n"
    report += (
        "Compressed CSV partitions and assignments live in ignored "
        "data/processed/phase3. SHA-256 hashes, source/cohort provenance and "
        "freeze time are in split_manifest.json. "
    )
    report += (
        "Reproduction must match those hashes. Development loaders reject test "
        "access. This is an application guard, not an operating-system access "
        "boundary.\n"
    )
    (root / "reports/modeling/data_split_report.md").write_text(report)
    return lock


def load_development(partition: str) -> pd.DataFrame:
    """Only train/validation are legal in the Phase 3 API; test loading needs a later phase."""
    if partition not in ("train", "validation"):
        raise ValueError("Test partition is locked; Phase 3 permits train and validation only")
    root, policy = settings()
    directory = root / policy["split"]["directory"]
    lock = json.loads((directory / "lock.json").read_text())
    path = directory / f"{partition}.csv.gz"
    verify_file(path, lock["files"][path.name])
    return pd.read_csv(
        path, keep_default_na=False, na_values=[""], dtype=dict.fromkeys(STRING_COLUMNS, "string")
    )


if __name__ == "__main__":
    prepare_frozen_partitions()
