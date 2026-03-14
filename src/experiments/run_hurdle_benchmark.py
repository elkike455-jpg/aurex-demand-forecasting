import os
import numpy as np
import pandas as pd

from src.data_loaders.load_aurex import load_aurex
from src.data_loaders.load_m5 import load_m5_single_series
from src.data_loaders.load_favorita import load_favorita_series
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.hurdle_model import HurdleModel


def train_test_split_ts(df, test_days=365):
    """Time-aware split: keep final block as test window."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) <= test_days + 30:
        test_days = max(30, int(len(df) * 0.2))
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """MAPE over strictly positive actuals."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def bias(y_true, y_pred):
    """Average signed prediction error."""
    return float(np.mean(y_pred - y_true))


def build_features(df):
    """Build consistent temporal features used by the Hurdle baseline."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    out["price"] = out["price"].ffill().bfill().fillna(0.0)

    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    out["lag_1"] = out["sales"].shift(1)
    out["lag_7"] = out["sales"].shift(7)
    out["lag_14"] = out["sales"].shift(14)
    out["lag_28"] = out["sales"].shift(28)

    out["ma_7"] = out["sales"].rolling(7, min_periods=1).mean()
    out["ma_14"] = out["sales"].rolling(14, min_periods=1).mean()
    out["ma_30"] = out["sales"].rolling(30, min_periods=1).mean()
    out["std_7"] = out["sales"].rolling(7, min_periods=1).std().fillna(0)

    out["days_since_sale"] = (out["sales"] == 0).astype(int).groupby((out["sales"] > 0).cumsum()).cumsum()
    out["sold_yesterday"] = (out["lag_1"] > 0).astype(int)
    out["trend_7"] = (out["ma_7"] - out["ma_7"].shift(7)).fillna(0)

    feature_cols = [
        "price", "dow", "month", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "ma_7", "ma_14", "ma_30", "std_7",
        "days_since_sale", "sold_yesterday", "trend_7",
    ]

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def evaluate_dataset(name, df, meta=None):
    """Run Hurdle baseline on one dataset and return one result row."""
    feat_df, feature_cols = build_features(df)
    train_feat, test_feat = train_test_split_ts(feat_df, test_days=365)

    y_train = train_feat["sales"].values.astype(float)
    y_test = test_feat["sales"].values.astype(float)
    X_train = train_feat[feature_cols].values.astype(float)
    X_test = test_feat[feature_cols].values.astype(float)

    model = HurdleModel().fit(X_train, y_train)
    y_pred, _, _ = model.forecast(X_test, X_train, y_train)
    beh = behavioral_metrics(y_test, y_pred)

    row = {
        "dataset": name,
        "model": "Hurdle-Only",
        "n_days_total": len(df),
        "train_days": len(train_feat),
        "test_days": len(test_feat),
        "start_date": str(feat_df["date"].min().date()),
        "end_date": str(feat_df["date"].max().date()),
        "zero_rate_train": float((y_train == 0).mean()),
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mape": mape(y_test, y_pred),
        "bias": bias(y_test, y_pred),
    }
    row.update(beh)

    if meta:
        row.update(meta)

    return row


def main():
    """Run Hurdle-only baseline across the three independent datasets."""
    os.makedirs("reports", exist_ok=True)
    results = []

    df_aurex = load_aurex(product_id=6)
    results.append(evaluate_dataset("AUREX_DB", df_aurex, meta={"series": "product_id=6"}))

    df_m5 = load_m5_single_series(base_path="data/raw/m5", random_pick=True, seed=42)
    results.append(evaluate_dataset("M5_WALMART", df_m5, meta={"series": "random_series"}))

    df_fav = load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family="GROCERY I")
    results.append(evaluate_dataset("FAVORITA", df_fav, meta={"series": "store=1,family=GROCERY I"}))

    out = pd.DataFrame(results)
    out_path = os.path.join("reports", "hurdle_benchmark_results.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
