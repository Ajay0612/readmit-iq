"""Strict API v1 schema; outcome and demographic audit fields cannot enter inference."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from readmit_iq.serving.contract import CONTRACT, MAX_BATCH, SCHEMA_VERSION, read_bundle_json

# Integers are JSON numbers, never coerced strings/bools/floats. The upper count bound is
# a technical signed-int32 transport limit, not a medical plausibility criterion.
Count = Annotated[StrictInt, Field(ge=0, le=2**31 - 1)]
TrackingID = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
]
MissingCategory = Literal["Unknown", "?", ""] | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EncounterFeatures(StrictModel):
    """Exactly ten frozen predictors, required even when a known missing marker is supplied."""

    time_in_hospital: Annotated[StrictInt, Field(ge=1, le=14)]
    number_inpatient: Count
    number_emergency: Count
    number_outpatient: Count
    age: str = Field(description="Exact source age band, e.g. [50-60).")
    gender: str | None = Field(description="Female, Male, Unknown/Invalid or known missing marker.")
    admission_type_id: StrictInt | MissingCategory
    admission_source_id: StrictInt | MissingCategory
    discharge_disposition_id: StrictInt
    medical_specialty: Annotated[str, Field(max_length=100)] | None

    @field_validator("age")
    @classmethod
    def age_band(cls, value):
        if value not in CONTRACT["age_bands"]:
            raise ValueError("Use an exact documented ten-year age band")
        return value

    @field_validator("gender")
    @classmethod
    def gender_category(cls, value):
        if value not in [*CONTRACT["genders"], "?", "", None]:
            raise ValueError("Use a documented gender category or known missing marker")
        return value

    @field_validator("medical_specialty")
    @classmethod
    def specialty_category(cls, value):
        if value not in [*CONTRACT["medical_specialties"], "?", "", None]:
            raise ValueError("Use a documented specialty spelling, Other, or known missing marker")
        return value

    @field_validator("admission_type_id", "admission_source_id", "discharge_disposition_id")
    @classmethod
    def administrative_code(cls, value, info):
        if value in [None, "Unknown", "?", ""]:
            return value
        mapping = read_bundle_json("reports/data_quality/id_mapping.json")
        if str(value) not in mapping[info.field_name]:
            raise ValueError("Code is not in the repository's historical source mapping")
        return value


class BatchRecord(StrictModel):
    request_id: TrackingID = Field(description="Stable non-predictive encounter tracking/tie key.")
    features: EncounterFeatures


class BatchRequest(StrictModel):
    records: Annotated[list[BatchRecord], Field(min_length=1, max_length=MAX_BATCH)]

    @model_validator(mode="after")
    def unique_keys(self):
        if len({record.request_id for record in self.records}) != len(self.records):
            raise ValueError("Batch request_id values must be unique")
        return self


class PredictionResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    model_version: str
    readmission_probability: float
    risk_percent: float
    outreach_selected: None = None
    decision_context: str


class BatchPrediction(PredictionResponse):
    request_id: str


class BatchResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    model_version: str
    total_encounters: int
    predictions: list[BatchPrediction]


class RankedPrediction(StrictModel):
    request_id: str
    readmission_probability: float
    risk_percent: float
    rank: int
    selected_for_outreach: bool


class PrioritizationResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    model_version: str
    policy_version: str
    total_eligible_encounters: int
    number_selected: int
    capacity_fraction: float
    capacity_is_assumed: bool
    decision_context: str
    predictions: list[RankedPrediction]


class HealthResponse(StrictModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None


class ErrorDetail(StrictModel):
    location: list[str | int]
    code: str
    message: str


class ErrorResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    error: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ModelResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    model_name: str
    model_version: str
    model_family: str
    scoring_context: str
    feature_count: int
    feature_names: list[str]
    training_commit: str
    freeze_commit: str
    artifact_verified: bool
    artifact_sha256: str
    policy_version: str
    targeting_policy: dict
    maximum_batch_records: int
    disclaimer: str
