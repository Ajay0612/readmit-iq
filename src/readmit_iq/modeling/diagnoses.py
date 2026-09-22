"""Strack et al. (2014), DOI 10.1155/2014/781670, Table 3 diagnosis groups.

The paper defines the primary-diagnosis groups. Applying them to secondary/tertiary
positions is an explicit project extension. Unknown and malformed codes stay separate.
"""

import re

import pandas as pd

MAPPING_VERSION = "strack2014-table3-v1"
RANGES = {
    "Circulatory": (390, 459),
    "Respiratory": (460, 519),
    "Digestive": (520, 579),
    "Genitourinary": (580, 629),
    "Injury": (800, 999),
    "Musculoskeletal": (710, 739),
    "Neoplasms": (140, 239),
}
SYMPTOMS = {785: "Circulatory", 786: "Respiratory", 787: "Digestive", 788: "Genitourinary"}


def diagnosis_group(value: object) -> str:
    """Map code families, including decimals; no integer truncation across boundaries."""
    if pd.isna(value) or str(value).strip() in ("", "?", "Unknown"):
        return "Unknown"
    code = str(value).strip().upper()
    if re.fullmatch(r"(V\d{2}|E\d{3})(\.\d{1,2})?", code):
        return "Other"  # The source paper's residual group includes supplementary codes.
    if not re.fullmatch(r"\d{1,3}(\.\d{1,2})?", code):
        return "Invalid"
    family = int(code.split(".")[0])
    if not 1 <= family <= 999:
        return "Invalid"
    if family == 250:
        return "Diabetes"
    if family in SYMPTOMS:
        return SYMPTOMS[family]
    for label, (low, high) in RANGES.items():
        if low <= family <= high:
            return label
    return "Other"
