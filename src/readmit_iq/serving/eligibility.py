"""Reuse the original Phase 2 mapping before any model invocation."""

import pandas as pd

from readmit_iq.data.cohort import assign_cohort, load_phase2_config
from readmit_iq.serving.contract import bundle_root
from readmit_iq.serving.errors import ServingError


def rules(root=None):
    return load_phase2_config((root or bundle_root()) / "configs/phase2.yaml")["cohort"]


def require_eligible(features, cohort_rules):
    frame = pd.DataFrame(
        {
            "encounter_id": range(len(features)),
            "discharge_disposition_id": [f.discharge_disposition_id for f in features],
        }
    )
    reasons = assign_cohort(frame, cohort_rules)
    rejected = reasons.loc[reasons.ne("eligible")]
    if not rejected.empty:
        raise ServingError(
            "ineligible_encounter",
            "The entire request was rejected before scoring; discharge eligibility is required.",
            422,
            [
                dict(
                    location=["records", int(index)],
                    code=str(reason),
                    message="Outside the original confirmed-discharge outreach cohort.",
                )
                for index, reason in rejected.iloc[:20].items()
            ],
        )
