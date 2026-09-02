from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from .config import RANDOM_STATE
from .data import distance_band
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class LaneMedianBaseline(BaseEstimator, RegressorMixin):
    """Predicts rate as distance times the median rate per mile of comparable loads.

    Lookup order: lane and equipment, then equipment and distance band, then equipment.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LaneMedianBaseline":
        rpm = pd.Series(np.asarray(y, dtype=float) / X["distance"].to_numpy(), index=X.index)
        band = distance_band(X["distance"])
        self.lane_rpm_ = rpm.groupby([X["pickup"], X["delivery"], X["equipment"]]).median()
        self.band_rpm_ = rpm.groupby([X["equipment"], band]).median()
        self.equipment_rpm_ = rpm.groupby(X["equipment"]).median()
        self.global_rpm_ = float(rpm.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lane_key = pd.MultiIndex.from_frame(X[["pickup", "delivery", "equipment"]])
        band_key = pd.MultiIndex.from_arrays([X["equipment"], distance_band(X["distance"])])
        rpm = pd.Series(self.lane_rpm_.reindex(lane_key).to_numpy(), index=X.index)
        rpm = rpm.fillna(pd.Series(self.band_rpm_.reindex(band_key).to_numpy(), index=X.index))
        rpm = rpm.fillna(X["equipment"].map(self.equipment_rpm_)).fillna(self.global_rpm_)
        return (rpm * X["distance"]).to_numpy()


def _linear_preprocessor() -> ColumnTransformer:
    scaled = [column for column in NUMERIC_FEATURES if column != "log_distance"]
    return ColumnTransformer(
        [
            ("distance_spline", SplineTransformer(n_knots=8, degree=3), ["log_distance"]),
            ("numeric", StandardScaler(), scaled),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _tree_preprocessor() -> ColumnTransformer:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer(
        [("numeric", "passthrough", NUMERIC_FEATURES), ("categorical", encoder, CATEGORICAL_FEATURES)]
    )


def _as_category(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column.startswith("categorical__"):
            out[column] = out[column].astype(int).astype("category")
    return out


def _log_target(regressor: BaseEstimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def make_lane_median() -> LaneMedianBaseline:
    return LaneMedianBaseline()


def make_ridge(alpha: float = 1.0) -> TransformedTargetRegressor:
    return _log_target(Pipeline([("prep", _linear_preprocessor()), ("model", Ridge(alpha=alpha))]))


def make_elastic_net(alpha: float = 1e-4, l1_ratio: float = 0.5) -> TransformedTargetRegressor:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, random_state=RANDOM_STATE)
    return _log_target(Pipeline([("prep", _linear_preprocessor()), ("model", model)]))


def make_hist_gb(**overrides) -> TransformedTargetRegressor:
    categorical_index = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))
    params = dict(
        max_iter=800,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=categorical_index,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return _log_target(Pipeline([("prep", _tree_preprocessor()), ("model", HistGradientBoostingRegressor(**params))]))


def make_lightgbm(**overrides) -> TransformedTargetRegressor:
    params = dict(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    params.update(overrides)
    prep = _tree_preprocessor().set_output(transform="pandas")
    steps = [("prep", prep), ("cast", FunctionTransformer(_as_category)), ("model", LGBMRegressor(**params))]
    return _log_target(Pipeline(steps))


class LogBlend(BaseEstimator, RegressorMixin):
    """Weighted geometric mean of member predictions, equivalent to averaging in log space."""

    def __init__(self, members: list[tuple[str, BaseEstimator, float]]):
        self.members = members

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogBlend":
        self.fitted_ = [(name, estimator.fit(X, y), weight) for name, estimator, weight in self.members]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        total_weight = sum(weight for _, _, weight in self.fitted_)
        log_prediction = sum(weight * np.log(estimator.predict(X)) for _, estimator, weight in self.fitted_)
        return np.exp(log_prediction / total_weight)


def make_blend() -> LogBlend:
    return LogBlend([("ridge", make_ridge(), 0.5), ("lightgbm", make_lightgbm(), 0.5)])


MODEL_BUILDERS: dict[str, Callable[[], BaseEstimator]] = {
    "lane_median": make_lane_median,
    "ridge": make_ridge,
    "elastic_net": make_elastic_net,
    "hist_gb": make_hist_gb,
    "lightgbm": make_lightgbm,
    "blend": make_blend,
}

SIMPLE_MODELS = ("lane_median", "ridge", "elastic_net")
BOOSTED_MODELS = ("hist_gb", "lightgbm", "blend")


def make_model(name: str) -> BaseEstimator:
    try:
        return MODEL_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODEL_BUILDERS)}") from exc
