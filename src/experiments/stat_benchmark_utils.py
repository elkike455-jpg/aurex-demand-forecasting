from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_loaders.load_m5 import load_m5_single_series
from src.experiments.common_protocol import OFFICIAL_BENCHMARK_PROTOCOL, split_series_protocol
from src.metrics.behavioral_metrics import behavioral_metrics


BENCHMARK_PRODUCTS = [
    ("FOODS_3_228_CA_1_validation", "high_demand_stable"),
    ("FOODS_2_044_CA_3_validation", "intermittent"),
    ("HOBBIES_1_133_CA_4_validation", "low_volume"),
]


def load_benchmark_series(series_id: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    df = load_m5_single_series(
        base_path=str(root / "data" / "raw" / "m5"),
        random_pick=False,
        series_id=series_id,
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.iloc[-OFFICIAL_BENCHMARK_PROTOCOL.max_days :].copy().reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Adapted from the existing SARIMAX/HURDLE benchmark scripts, but with
    lag-safe rolling features to avoid same-day target leakage.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    out["price"] = out["price"].ffill().bfill().fillna(0.0)

    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    sales_lagged = out["sales"].shift(1)
    out["lag_1"] = out["sales"].shift(1)
    out["lag_7"] = out["sales"].shift(7)
    out["lag_14"] = out["sales"].shift(14)
    out["lag_28"] = out["sales"].shift(28)

    out["ma_7"] = sales_lagged.rolling(7, min_periods=1).mean()
    out["ma_14"] = sales_lagged.rolling(14, min_periods=1).mean()
    out["ma_30"] = sales_lagged.rolling(30, min_periods=1).mean()
    out["std_7"] = sales_lagged.rolling(7, min_periods=1).std().fillna(0.0)

    zero_indicator = (out["sales"] == 0).astype(int)
    out["days_since_sale"] = zero_indicator.groupby((out["sales"] > 0).cumsum()).cumsum().shift(1)
    out["sold_yesterday"] = (out["lag_1"] > 0).astype(int)
    out["trend_7"] = (out["ma_7"] - out["ma_7"].shift(7)).fillna(0.0)

    feature_cols = [
        "price",
        "dow",
        "month",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "ma_7",
        "ma_14",
        "ma_30",
        "std_7",
        "days_since_sale",
        "sold_yesterday",
        "trend_7",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def prepare_protocol_data(series_id: str) -> Dict[str, object]:
    df = load_benchmark_series(series_id)
    feat_df, feature_cols = build_features(df)

    y = feat_df["sales"].to_numpy(dtype=float)
    y_train, y_val, y_test = split_series_protocol(
        y,
        val_days=OFFICIAL_BENCHMARK_PROTOCOL.val_days,
        test_days=OFFICIAL_BENCHMARK_PROTOCOL.test_days,
    )

    train_days = len(y_train)
    fit_days = len(y_train) + len(y_val)

    train_feat = feat_df.iloc[:train_days].copy().reset_index(drop=True)
    val_feat = feat_df.iloc[train_days:fit_days].copy().reset_index(drop=True)
    fit_feat = feat_df.iloc[:fit_days].copy().reset_index(drop=True)
    test_feat = feat_df.iloc[fit_days:].copy().reset_index(drop=True)

    X_train = train_feat[feature_cols].to_numpy(dtype=float)
    X_val = val_feat[feature_cols].to_numpy(dtype=float)
    X_fit = fit_feat[feature_cols].to_numpy(dtype=float)
    X_test = test_feat[feature_cols].to_numpy(dtype=float)

    return {
        "df": df,
        "feature_df": feat_df,
        "feature_cols": feature_cols,
        "train_feat": train_feat,
        "val_feat": val_feat,
        "fit_feat": fit_feat,
        "test_feat": test_feat,
        "y_train": y_train,
        "y_val": y_val,
        "y_fit": np.concatenate([y_train, y_val]),
        "y_test": y_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_fit": X_fit,
        "X_test": X_test,
    }


def flat_nonflat_label(variance_ratio: float, threshold: float = 0.10) -> str:
    return "flat" if float(variance_ratio) < float(threshold) else "non-flat"


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    beh = behavioral_metrics(y_true, y_pred)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "pred_std": float(np.std(y_pred)),
        "real_std": float(np.std(y_true)),
        "variance_ratio": float(beh["variance_ratio"]),
        "trend_correlation": float(beh["trend_correlation"]),
        "direction_accuracy": float(beh["direction_accuracy"]),
        "shape_similarity": float(beh["shape_similarity"]),
        "peak_detection_rate": float(beh["peak_detection_rate"]),
        "n_peaks_real": int(beh["n_peaks_real"]),
        "n_peaks_detected": int(beh["n_peaks_detected"]),
    }
