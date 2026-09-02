from __future__ import annotations

import numpy as np
import pandas as pd

from .data import haversine_miles

TREND_ORIGIN = pd.Timestamp("2025-01-01")

NUMERIC_FEATURES = [
    "log_distance",
    "distance",
    "haversine",
    "circuity",
    "weight",
    "weight_missing",
    "market_index",
    "log_market_index",
    "market_index_missing",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "day_of_week",
    "is_weekend",
    "days_since_origin",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"].astype(float)
    out["log_distance"] = np.log(out["distance"])
    out["haversine"] = haversine_miles(
        frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"]
    )
    out["circuity"] = out["distance"] / out["haversine"].clip(lower=1.0)
    out["weight"] = frame["weight"].astype(float)
    out["weight_missing"] = frame["weight_missing"].astype(int)
    out["market_index"] = frame["market_index"].astype(float)
    out["log_market_index"] = np.log(out["market_index"])
    out["market_index_missing"] = frame["market_index_missing"].astype(int)
    out["quote_signal"] = frame["quote_signal"].astype(float)
    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[column] = frame[column].astype(float)
    out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
    date = pd.to_datetime(frame["date"])
    out["day_of_week"] = date.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["days_since_origin"] = (date - TREND_ORIGIN).dt.days.astype(float)
    for column in CATEGORICAL_FEATURES:
        out[column] = frame[column].astype(str)
    return out[ALL_FEATURES]
