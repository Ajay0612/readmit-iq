"""Load source data without cleaning, imputing, or discarding meaningful tokens."""

import csv
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
ID_COLUMNS = ["encounter_id", "patient_nbr"]
MAPPING_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw(path: Path) -> pd.DataFrame:
    """Keep '?', empty strings and literal 'None'; preserve IDs/diagnoses as strings."""
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        dtype=dict.fromkeys([*ID_COLUMNS, "diag_1", "diag_2", "diag_3"], "string"),
    )
    LOGGER.info("Loaded %s: %s rows, %s columns", path.name, *frame.shape)
    return frame


def load_id_mapping(path: Path) -> dict[str, dict[int, str]]:
    """Parse the three separate tables in UCI's original IDS_mapping.csv."""
    mappings: dict[str, dict[int, str]] = {}
    current = None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            if row[0] in MAPPING_COLUMNS:
                current = row[0]
                mappings[current] = {}
            elif current is not None:
                key = int(row[0])
                if key in mappings[current]:
                    raise ValueError(f"Duplicate mapping: {current}={key}")
                mappings[current][key] = row[1].strip()
            else:
                raise ValueError("Mapping data appeared before a section header")
    if set(mappings) != set(MAPPING_COLUMNS):
        raise ValueError("Mapping file is missing a section")
    return mappings
