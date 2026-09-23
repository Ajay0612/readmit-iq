"""Phase 4 feature configurations reuse the reviewed Phase 3 transformations unchanged."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from readmit_iq.modeling.features import DERIVED_UTILIZATION, FeatureBuilder, feature_columns
from readmit_iq.modeling.preprocessing import CategoryEncoder

CONFIGURATIONS = {
    "raw": "raw_utilization",
    "indicators": "primary",
    "raw_engineered": "primary",
    "diagnosis_grouped": "diagnosis_groups",
    "diagnosis_raw": "diagnosis_raw",
}
DEFAULTS = {
    "logistic": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 0.0001},
    "boosting": {
        "max_iter": 150,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
    },
    "forest": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "n_jobs": 2,
    },
}


def schema(configuration: str) -> tuple[list[str], list[str]]:
    if configuration not in CONFIGURATIONS:
        raise ValueError("Unknown Phase 4 feature configuration")
    numeric, categorical = feature_columns(CONFIGURATIONS[configuration])
    if configuration == "indicators":
        numeric = ["time_in_hospital", *DERIVED_UTILIZATION[1:]]
    elif configuration.startswith("diagnosis_"):
        numeric = [c for c in numeric if c not in DERIVED_UTILIZATION]
    return numeric, categorical


class DevelopmentFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, configuration: str = "raw"):
        self.configuration = configuration

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_out_ = np.asarray(self.transform(X).columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric, categorical = schema(self.configuration)
        return FeatureBuilder(CONFIGURATIONS[self.configuration]).transform(X)[
            numeric + categorical
        ]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def build_pipeline(
    family: str,
    configuration: str,
    parameters: dict | None = None,
    seed: int = 42,
    rare_min_count: int = 100,
) -> Pipeline:
    if family not in DEFAULTS:
        raise ValueError("Unsupported Phase 4 model family")
    numeric, categorical = schema(configuration)
    params = deepcopy(DEFAULTS[family])
    params.update(parameters or {})
    if params.get("class_weight") is not None or params.get("early_stopping", False):
        raise ValueError("Unweighted models and group-safe explicit CV are required")
    if family == "logistic":
        model = LogisticRegression(**params, random_state=seed)
    elif family == "forest":
        model = RandomForestClassifier(**params, random_state=seed)
    else:
        model = HistGradientBoostingClassifier(
            **params,
            random_state=seed,
            categorical_features=list(range(len(numeric), len(numeric) + len(categorical))),
        )
    return Pipeline(
        [
            ("features", DevelopmentFeatures(configuration)),
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            StandardScaler() if family == "logistic" else "passthrough",
                            numeric,
                        ),
                        (
                            "categorical",
                            CategoryEncoder(
                                "ordinal" if family == "boosting" else "onehot", rare_min_count
                            ),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                    sparse_threshold=0 if family == "boosting" else 1.0,
                ),
            ),
            ("model", model),
        ]
    )
