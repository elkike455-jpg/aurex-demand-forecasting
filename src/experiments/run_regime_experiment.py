import numpy as np
import pandas as pd

from src.data_loaders.load_favorita import load_favorita_series
from src.models.regime_forecast_engine import RegimeForecastEngine


def build_features(df):
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
    out["std_7"] = out["sales"].rolling(7, min_periods=1).std().fillna(0.0)

    out["days_since_sale"] = (out["sales"] == 0).astype(int).groupby((out["sales"] > 0).cumsum()).cumsum()
    out["sold_yesterday"] = (out["lag_1"] > 0).astype(int)
    out["trend_7"] = (out["ma_7"] - out["ma_7"].shift(7)).fillna(0.0)

    feature_cols = [
        "price", "dow", "month", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "ma_7", "ma_14", "ma_30", "std_7",
        "days_since_sale", "sold_yesterday", "trend_7",
    ]

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def main():
    df = load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family="GROCERY I")

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    train_feat, feature_cols = build_features(train_df)
    test_feat, _ = build_features(test_df)

    y_train = train_feat["sales"].values.astype(float)
    y_test = test_feat["sales"].values.astype(float)
    X_train = train_feat[feature_cols].values.astype(float)
    X_test = test_feat[feature_cols].values.astype(float)

    engine = RegimeForecastEngine()
    y_pred, _, _, model_name = engine.run(y_train, X_train, y_test, X_test)

    mae = float(np.mean(np.abs(y_test - y_pred)))
    print(f"Model: {model_name}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
