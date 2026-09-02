import numpy as np
import pandas as pd
import pytest

from freight_rate.data import Cleaner, haversine_miles, rate_outlier_mask


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup": ["A"] * 6,
            "delivery": ["B"] * 6,
            "equipment": ["Dry Van"] * 4 + ["Reefer"] * 2,
            "distance": [300.0] * 6,
            "weight": [30000.0, -28000.0, np.nan, 32000.0, 20000.0, np.nan],
            "market_index": [1.0, np.nan, 1.1, 1.05, 0.9, 0.9],
            "date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3),
            "posted_rate": [600.0, 610.0, 590.0, 1900.0, 700.0, 710.0],
        }
    )


def test_cleaner_reverses_sign_flips_and_imputes(frame):
    cleaned = Cleaner().fit_transform(frame)
    assert cleaned.loc[1, "weight"] == 28000.0
    assert cleaned["weight"].notna().all()
    assert cleaned.loc[2, "weight_missing"] == 1
    assert cleaned.loc[2, "weight"] == 30000.0
    assert cleaned.loc[5, "weight"] == 20000.0


def test_cleaner_fills_market_index_with_same_day_mean(frame):
    cleaned = Cleaner().fit_transform(frame)
    assert cleaned.loc[1, "market_index"] == pytest.approx((1.0 + 1.1) / 2)
    assert cleaned.loc[1, "market_index_missing"] == 1
    assert cleaned["market_index"].notna().all()


def test_outlier_mask_flags_only_the_implausible_rate(frame):
    mask = rate_outlier_mask(frame)
    assert mask.tolist() == [False, False, False, True, False, False]


def test_haversine_known_distance():
    miles = haversine_miles(pd.Series([40.7128]), pd.Series([-74.0060]), pd.Series([34.0522]), pd.Series([-118.2437]))
    assert miles.iloc[0] == pytest.approx(2445, rel=0.01)
