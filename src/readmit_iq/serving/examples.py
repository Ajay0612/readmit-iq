"""Hand-authored synthetic demonstration inputs; no source/data partition is read."""

from copy import deepcopy

LOW_HISTORY = dict(
    time_in_hospital=2,
    number_inpatient=0,
    number_emergency=0,
    number_outpatient=0,
    age="[40-50)",
    gender="Female",
    admission_type_id=3,
    admission_source_id=1,
    discharge_disposition_id=1,
    medical_specialty="InternalMedicine",
)
PRIOR_UTILIZATION = dict(
    time_in_hospital=7,
    number_inpatient=4,
    number_emergency=2,
    number_outpatient=1,
    age="[60-70)",
    gender="Male",
    admission_type_id=1,
    admission_source_id=7,
    discharge_disposition_id=6,
    medical_specialty="Cardiology",
)
DISCHARGE_CONTEXT = dict(
    time_in_hospital=5,
    number_inpatient=0,
    number_emergency=0,
    number_outpatient=0,
    age="[70-80)",
    gender="Female",
    admission_type_id=3,
    admission_source_id=7,
    discharge_disposition_id=22,
    medical_specialty="PhysicalMedicineandRehabilitation",
)
PATTERNS = {
    "Lower prior utilization": LOW_HISTORY,
    "Prior utilization": PRIOR_UTILIZATION,
    "Rehabilitation discharge context": DISCHARGE_CONTEXT,
}


def synthetic_batch(n=20):
    patterns = list(PATTERNS.values())
    return {
        "records": [
            dict(request_id=f"demo-{i + 1:04}", features=deepcopy(patterns[i % 3]))
            for i in range(n)
        ]
    }
