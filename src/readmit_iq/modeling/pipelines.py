"""Comparable linear, bagged-tree and native-categorical boosting pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from readmit_iq.modeling.features import FeatureBuilder, feature_columns
from readmit_iq.modeling.preprocessing import CategoryEncoder


def build_pipeline(experiment: dict, policy: dict) -> Pipeline:
    """No estimator receives test data; early stopping is off to avoid an internal row split."""
    family = experiment["family"]
    variant = experiment["variant"]
    numeric, categorical = feature_columns(variant)
    settings = policy["modeling"]
    seed = policy["random_seed"]
    if family == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif family == "logistic":
        estimator = LogisticRegression(
            **settings["logistic"], random_state=seed, class_weight=experiment.get("class_weight")
        )
    elif family == "forest":
        estimator = RandomForestClassifier(**settings["forest"], random_state=seed)
    elif family == "boosting":
        estimator = HistGradientBoostingClassifier(
            **settings["boosting"],
            random_state=seed,
            categorical_features=list(range(len(numeric), len(numeric) + len(categorical))),
        )
    else:
        raise ValueError(f"Unsupported family: {family}")
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler() if family == "logistic" else "passthrough", numeric),
            (
                "categorical",
                CategoryEncoder(
                    "ordinal" if family == "boosting" else "onehot", settings["rare_min_count"]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0 if family == "boosting" else 1.0,
    )
    return Pipeline(
        [("features", FeatureBuilder(variant)), ("preprocess", preprocess), ("model", estimator)]
    )
