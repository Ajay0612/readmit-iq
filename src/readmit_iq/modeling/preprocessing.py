"""Train-only category vocabularies with reserved unknown/rare levels and safe inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class CategoryEncoder(TransformerMixin, BaseEstimator):
    """Learn frequent categories from training only; preserve Unknown and Not measured."""

    def __init__(self, mode: str = "onehot", min_count: int = 100):
        self.mode = mode
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None):
        if self.mode not in {"onehot", "ordinal"} or self.min_count < 1:
            raise ValueError("Invalid categorical encoder configuration")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.n_fit_rows_ = len(X)
        self.categories_ = []
        self.frequency_records_ = []
        protected = {"Unknown", "Not measured", "Other", "Invalid"}
        for column in X:
            counts = X[column].value_counts(dropna=False)
            levels = sorted(set(counts[counts.ge(self.min_count)].index) | protected)
            if column == "age":
                levels = sorted(set(levels) | {f"[{a}-{a + 10})" for a in range(0, 100, 10)})
            self.categories_.append(levels)
            self.frequency_records_.extend(
                dict(
                    feature=column,
                    category=str(level),
                    train_count=int(count),
                    encoded_as=str(level) if level in levels else "Other",
                )
                for level, count in counts.items()
            )
        if self.mode == "onehot":
            self.encoder_ = OneHotEncoder(
                categories=self.categories_, handle_unknown="ignore", sparse_output=True
            )
        else:
            self.encoder_ = OrdinalEncoder(
                categories=self.categories_, handle_unknown="use_encoded_value", unknown_value=-1
            )
        self.encoder_.fit(self._group(X))
        return self

    def _group(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if list(result.columns) != list(self.feature_names_in_):
            raise ValueError("Categorical schema changed")
        for column, levels in zip(result, self.categories_, strict=True):
            result[column] = result[column].where(result[column].isin(levels), "Other")
        return result

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "encoder_")
        return self.encoder_.transform(self._group(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return self.encoder_.get_feature_names_out(self.feature_names_in_)
