"""Only synthetic/new inference requests enter this service; no modeling loaders are used."""

import numpy as np
import pandas as pd

from readmit_iq.serving.contract import CONTRACT, DISCLAIMER, FEATURES, MAX_BATCH, SINGLE_CONTEXT
from readmit_iq.serving.eligibility import require_eligible
from readmit_iq.serving.errors import ServingError
from readmit_iq.serving.schemas import (
    BatchPrediction,
    BatchResponse,
    ModelResponse,
    PredictionResponse,
)


class Predictor:
    def __init__(self, model, cohort_rules):
        self.model = model
        self.cohort_rules = cohort_rules

    def probabilities(self, features):
        require_eligible(features, self.cohort_rules)
        # Tracking fields, targets and audit-only variables are never part of this frame.
        frame = pd.DataFrame([item.model_dump() for item in features], columns=FEATURES)
        try:
            output = np.asarray(self.model.predict_proba(frame))
            if (
                output.shape != (len(features), 2)
                or not np.isfinite(output).all()
                or (output < 0).any()
                or (output > 1).any()
                or not np.allclose(output.sum(axis=1), 1, atol=1e-12, rtol=0)
            ):
                raise ValueError("Invalid inference output")
        except Exception as error:
            raise ServingError(
                "inference_failed", "Inference failed; no prediction was returned.", 500
            ) from error
        return output[:, 1]

    def prediction(self, probability):
        return PredictionResponse(
            model_version=self.model.metadata["version"],
            readmission_probability=float(probability),
            risk_percent=float(probability * 100),
            decision_context=SINGLE_CONTEXT,
        )

    def single(self, features):
        return self.prediction(self.probabilities([features])[0])

    def batch(self, request):
        probabilities = self.probabilities([record.features for record in request.records])
        return BatchResponse(
            model_version=self.model.metadata["version"],
            total_encounters=len(request.records),
            predictions=[
                BatchPrediction(request_id=record.request_id, **self.prediction(p).model_dump())
                for record, p in zip(request.records, probabilities, strict=True)
            ],
        )

    def public_metadata(self):
        metadata = self.model.metadata
        return ModelResponse(
            model_name="ReadmitIQ",
            model_family="LogisticRegression",
            model_version=metadata["version"],
            scoring_context="Confirmed discharge once destination is known",
            feature_count=len(FEATURES),
            feature_names=list(FEATURES),
            training_commit=metadata["training_commit"],
            freeze_commit=metadata["freeze_commit"],
            artifact_verified=self.model.artifact_verified,
            artifact_sha256=metadata["pipeline_sha256"],
            policy_version=CONTRACT["policy_version"],
            targeting_policy={
                key: metadata["targeting"][key]
                for key in [
                    "type",
                    "fraction",
                    "capacity_is_assumed",
                    "unit",
                    "capacity_rounding",
                    "tie_break",
                ]
            },
            maximum_batch_records=MAX_BATCH,
            disclaimer=DISCLAIMER,
        )
