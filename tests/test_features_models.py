import numpy as np
import pandas as pd

from freight_rate.data import Cleaner
from freight_rate.features import ALL_FEATURES, build_features
from freight_rate.models import LaneMedianBaseline, make_model


def _frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = {"A": (35.0, -90.0), "B": (38.0, -85.0), "C": (33.0, -97.0)}
    pickup = rng.choice(list(cities), n)
    delivery = np.array([rng.choice([c for c in cities if c != p]) for p in pickup])
    distance = rng.uniform(100, 2000, n)
    frame = pd.DataFrame(
        {
            "pickup": pickup,
            "delivery": delivery,
            "pickup_lat": [cities[c][0] for c in pickup],
            "pickup_lon": [cities[c][1] for c in pickup],
            "delivery_lat": [cities[c][0] for c in delivery],
            "delivery_lon": [cities[c][1] for c in delivery],
            "distance": distance,
            "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"], n),
            "weight": rng.uniform(10000, 45000, n),
            "date": pd.to_datetime("2025-03-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
            "market_index": rng.uniform(0.8, 1.3, n),
            "quote_signal": rng.uniform(1.5, 2.5, n),
        }
    )
    frame["posted_rate"] = distance * rng.uniform(1.8, 2.6, n)
    return frame


def test_build_features_has_expected_columns_and_no_nans():
    features = build_features(Cleaner().fit_transform(_frame()))
    assert list(features.columns) == ALL_FEATURES
    assert not features.isna().any().any()


def test_lane_median_falls_back_for_unseen_lane():
    frame = Cleaner().fit_transform(_frame())
    X = build_features(frame)
    model = LaneMedianBaseline().fit(X, frame["posted_rate"].to_numpy())
    unseen = X.iloc[[0]].assign(pickup="Z", delivery="Q")
    assert np.isfinite(model.predict(unseen)).all()
    assert model.predict(unseen)[0] > 0


def test_every_model_fits_and_predicts_positive_rates():
    frame = Cleaner().fit_transform(_frame(120))
    X = build_features(frame)
    y = frame["posted_rate"].to_numpy()
    for name in ("lane_median", "ridge", "elastic_net", "hist_gb", "lightgbm", "blend"):
        model = make_model(name).fit(X, y)
        pred = model.predict(X.assign(pickup=np.where(X.index < 5, "NewCity", X["pickup"])))
        assert pred.shape == (len(X),)
        assert (pred > 0).all(), name
