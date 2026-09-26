"""Versioned transport constraints and provenance anchors, separate from modeling policy."""

import json
import os
from functools import lru_cache
from pathlib import Path

CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())
FEATURES = tuple(CONTRACT["features"])
SCHEMA_VERSION = CONTRACT["schema_version"]
MAX_BATCH = CONTRACT["maximum_batch_records"]
MAX_BODY_BYTES = CONTRACT["maximum_body_bytes"]
DISCLAIMER = (
    "Portfolio demonstration only. Built using historical 1999–2008 hospital data. "
    "Not validated for clinical decision-making or patient care. No claim of HIPAA compliance."
)
SINGLE_CONTEXT = (
    "Retrospective model estimate of recorded <30-day readmission, not a diagnosis. "
    "Outreach prioritization requires comparison within an operational batch."
)


def bundle_root():
    return Path(os.environ.get("READMITIQ_SERVING_ROOT", Path.cwd())).resolve()


@lru_cache(maxsize=8)
def read_bundle_json(relative, root=None):
    return json.loads(((root or bundle_root()) / relative).read_text())
