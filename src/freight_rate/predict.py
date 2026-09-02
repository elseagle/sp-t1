from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DECEMBER_PATH, MODELS_DIR, PREDICTIONS_PATH, TEMPLATE_PATH, TRAIN_PATH, VALIDATION_PATH, ID_COLUMN
from .data import load_raw
from .features import build_features

DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.Series:
    cleaned = bundle["cleaner"].transform(frame)
    prediction = bundle["model"].predict(build_features(cleaned))
    return pd.Series(prediction, index=frame.index, name="predicted_rate")


def predict_validation(bundle: dict, validation: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    prediction = predict_frame(bundle, validation)
    lookup = pd.Series(prediction.to_numpy(), index=validation[ID_COLUMN])
    output = template[[ID_COLUMN]].copy()
    output["predicted_rate"] = output[ID_COLUMN].map(lookup).round(2)
    if output["predicted_rate"].isna().any():
        raise ValueError("template contains load_id values missing from the validation set")
    return output


def build_december_frame(december: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Complete the fixed-lane December rows with the context features the model expects.

    Coordinates come from the city reference in the training data. The market index
    is the same-day mean observed across all December loads in the validation set, and
    the quote signal is the training median for the equipment type.
    """
    frame = december.drop(columns=["predicted_rate"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    cities = pd.concat(
        [
            train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates("city").set_index("city")
    frame["pickup_lat"] = frame["pickup"].map(cities["lat"])
    frame["pickup_lon"] = frame["pickup"].map(cities["lon"])
    frame["delivery_lat"] = frame["delivery"].map(cities["lat"])
    frame["delivery_lon"] = frame["delivery"].map(cities["lon"])
    daily_index = validation.groupby("date")["market_index"].mean()
    frame["market_index"] = frame["date"].map(daily_index)
    quote_by_equipment = train.groupby("equipment")["quote_signal"].median()
    frame["quote_signal"] = frame["equipment"].map(quote_by_equipment)
    missing = frame.columns[frame.isna().any()].tolist()
    if missing:
        raise ValueError(f"could not complete December inputs, missing values in {missing}")
    return frame


def predict_december(bundle: dict, december: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    frame = build_december_frame(december, train, validation)
    output = december[DECEMBER_COLUMNS[:-1]].copy()
    output["predicted_rate"] = predict_frame(bundle, frame).round(2).to_numpy()
    return output[DECEMBER_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation and fixed-lane December predictions from a saved model.")
    parser.add_argument("--model", default="blend", help="name of the saved model bundle in models/")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--december-output", type=Path, default=DECEMBER_PATH)
    args = parser.parse_args()

    bundle = load_bundle(MODELS_DIR / f"{args.model}.joblib")
    train = load_raw(TRAIN_PATH)
    validation = load_raw(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    december = pd.read_csv(DECEMBER_PATH)

    predictions = predict_validation(bundle, validation, template)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {args.output}")

    december_out = predict_december(bundle, december, train, validation)
    december_out.to_csv(args.december_output, index=False)
    print(f"Wrote {len(december_out)} December predictions to {args.december_output}")


if __name__ == "__main__":
    main()
