from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .config import HOLDOUT_CUTOFF, MODELS_DIR, REPORTS_DIR, TARGET, TRAIN_PATH
from .data import Cleaner, load_raw, rate_outlier_mask
from .evaluation import expanding_window_folds, regression_metrics, time_holdout_split
from .features import ALL_FEATURES, build_features
from .models import MODEL_BUILDERS, SIMPLE_MODELS, make_model

METRIC_COLUMNS = ["mae", "rmse", "mape", "median_ape", "r2"]


def evaluate_fold(name: str, raw: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, outlier: pd.Series) -> dict:
    train_rows = raw.iloc[train_idx]
    train_rows = train_rows.loc[~outlier.iloc[train_idx].to_numpy()]
    cleaner = Cleaner().fit(train_rows)
    train_clean = cleaner.transform(train_rows)
    valid_clean = cleaner.transform(raw.iloc[valid_idx])

    started = time.perf_counter()
    model = make_model(name).fit(build_features(train_clean), train_clean[TARGET].to_numpy())
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(build_features(valid_clean))

    y_valid = valid_clean[TARGET].to_numpy()
    keep = ~outlier.iloc[valid_idx].to_numpy()
    result = {"model": name, "n_train": len(train_clean), "n_valid": len(valid_clean), "fit_seconds": round(fit_seconds, 1)}
    result.update({f"{k}_clean": v for k, v in regression_metrics(y_valid[keep], prediction[keep]).items()})
    result.update({f"{k}_all": v for k, v in regression_metrics(y_valid, prediction).items()})
    return result


def compare_models(names: list[str], raw: pd.DataFrame, use_folds: bool) -> pd.DataFrame:
    outlier = rate_outlier_mask(raw)
    if use_folds:
        folds = expanding_window_folds(raw)
    else:
        train_idx, valid_idx = time_holdout_split(raw, HOLDOUT_CUTOFF)
        folds = [("holdout", train_idx, valid_idx)]
    rows = []
    for name in names:
        for fold_name, train_idx, valid_idx in folds:
            row = evaluate_fold(name, raw, train_idx, valid_idx, outlier)
            row["fold"] = fold_name
            rows.append(row)
            print(f"{name:>12} {fold_name:>8}  MAE {row['mae_clean']:8.2f}  MAPE {row['mape_clean']:5.2f}%  R2 {row['r2_clean']:.4f}  ({row['fit_seconds']}s)")
    return pd.DataFrame(rows)


def fit_final(name: str, raw: pd.DataFrame) -> dict:
    outlier = rate_outlier_mask(raw)
    kept = raw.loc[~outlier]
    cleaner = Cleaner().fit(kept)
    frame = cleaner.transform(kept)
    model = make_model(name).fit(build_features(frame), frame[TARGET].to_numpy())
    return {
        "model_name": name,
        "model": model,
        "cleaner": cleaner,
        "features": ALL_FEATURES,
        "n_train": len(frame),
        "n_outliers_removed": int(outlier.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate models on time-based validation and fit the final model.")
    parser.add_argument("--models", nargs="+", default=list(SIMPLE_MODELS), choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--folds", action="store_true", help="use three forward-chaining folds instead of the single holdout")
    parser.add_argument("--final-model", default=None, help="model to refit on all data and save; omit to skip")
    parser.add_argument("--comparison-name", default="model_comparison")
    args = parser.parse_args()

    raw = load_raw(TRAIN_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = compare_models(args.models, raw, args.folds)
    output = REPORTS_DIR / f"{args.comparison_name}.csv"
    results.to_csv(output, index=False)
    print(f"\nSaved comparison to {output}")

    if args.final_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        bundle = fit_final(args.final_model, raw)
        path = MODELS_DIR / f"{args.final_model}.joblib"
        joblib.dump(bundle, path)
        print(f"Saved final {args.final_model} trained on {bundle['n_train']:,} loads to {path}")


if __name__ == "__main__":
    main()
