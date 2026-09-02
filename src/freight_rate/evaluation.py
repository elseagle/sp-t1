from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HOLDOUT_CUTOFF


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error / y_true)) * 100),
        "median_ape": float(np.median(np.abs(error / y_true)) * 100),
        "r2": float(1 - np.sum(error**2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def time_holdout_split(frame: pd.DataFrame, cutoff: str = HOLDOUT_CUTOFF) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(frame["date"])
    is_valid = (dates >= pd.Timestamp(cutoff)).to_numpy()
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def expanding_window_folds(
    frame: pd.DataFrame, valid_starts: tuple[str, ...] = ("2025-05-01", "2025-07-01", "2025-09-01"), horizon_months: int = 2
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Forward-chaining folds where each validation window spans the next two months."""
    dates = pd.to_datetime(frame["date"])
    folds = []
    for start in valid_starts:
        start_ts = pd.Timestamp(start)
        end_ts = start_ts + pd.DateOffset(months=horizon_months)
        train_idx = np.flatnonzero((dates < start_ts).to_numpy())
        valid_idx = np.flatnonzero(((dates >= start_ts) & (dates < end_ts)).to_numpy())
        folds.append((f"{start_ts:%b}-{(end_ts - pd.Timedelta(days=1)):%b}", train_idx, valid_idx))
    return folds
