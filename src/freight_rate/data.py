from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGET

EARTH_RADIUS_MILES = 3958.8
DISTANCE_BANDS = (0, 200, 400, 800, 1500, 2500, np.inf)


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    lat1, lon1, lat2, lon2 = (np.radians(s.astype(float)) for s in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def distance_band(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, DISTANCE_BANDS, labels=False, right=True).astype(int)


@dataclass
class Cleaner:
    """Repairs known data defects using statistics learned from the training set.

    Weight sign flips are reversed with an absolute value. Missing weights are
    filled with the equipment median. Missing market index values are filled
    with the same-day mean of the frame being cleaned, since the index is a
    day-level series, falling back to the training median.
    """

    weight_by_equipment: dict[str, float] = field(default_factory=dict)
    weight_default: float = float("nan")
    market_index_default: float = float("nan")

    def fit(self, frame: pd.DataFrame) -> "Cleaner":
        weight = frame["weight"].abs()
        self.weight_by_equipment = weight.groupby(frame["equipment"]).median().to_dict()
        self.weight_default = float(weight.median())
        self.market_index_default = float(frame["market_index"].median())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["weight_missing"] = out["weight"].isna().astype(int)
        out["weight"] = out["weight"].abs()
        equipment_median = out["equipment"].map(self.weight_by_equipment)
        out["weight"] = out["weight"].fillna(equipment_median).fillna(self.weight_default)

        out["market_index_missing"] = out["market_index"].isna().astype(int)
        daily_mean = out.groupby("date")["market_index"].transform("mean")
        out["market_index"] = (
            out["market_index"].fillna(daily_mean).fillna(self.market_index_default)
        )
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def rate_outlier_mask(frame: pd.DataFrame, log_threshold: float = np.log(2.0)) -> pd.Series:
    """Flag loads whose rate per mile is implausibly far from comparable loads.

    The reference is the lane and equipment median rate per mile when the lane
    has at least three loads, otherwise the equipment and distance-band median.
    """
    rpm = frame[TARGET] / frame["distance"]
    lane_key = [frame["pickup"], frame["delivery"], frame["equipment"]]
    lane_median = rpm.groupby(lane_key).transform("median")
    lane_count = rpm.groupby(lane_key).transform("size")
    band_median = rpm.groupby([frame["equipment"], distance_band(frame["distance"])]).transform("median")
    reference = lane_median.where(lane_count >= 3, band_median)
    return (np.log(rpm / reference)).abs() > log_threshold


def prepare_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, Cleaner]:
    cleaner = Cleaner().fit(raw)
    frame = cleaner.transform(raw)
    outliers = rate_outlier_mask(frame)
    return frame.loc[~outliers].reset_index(drop=True), cleaner
