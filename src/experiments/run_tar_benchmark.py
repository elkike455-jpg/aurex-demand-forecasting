import os
import pandas as pd

from src.data_loaders.load_aurex import load_aurex
from src.data_loaders.load_m5 import load_m5_single_series
from src.data_loaders.load_favorita import load_favorita_series
from src.experiments.run_sarimax_benchmark import (
    bias,
    build_features,
    mae,
    mape,
    rmse,
    train_test_split_ts,
)
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.tar_model import TARModel


def evaluate_dataset(name, df, meta=None):
    """Run TAR baseline on one dataset and return one result row."""
    feat_df, feature_cols = build_features(df)
    train_feat, test_feat = train_test_split_ts(feat_df, test_days=365)

    y_train = train_feat["sales"].values.astype(float)
    y_test = test_feat["sales"].values.astype(float)
    X_train = train_feat[feature_cols].values.astype(float)
    X_test = test_feat[feature_cols].values.astype(float)

    model = TARModel().fit(y_train, X_train)
    y_pred, _, _ = model.forecast(len(y_test), X_test)
    beh = behavioral_metrics(y_test, y_pred)

    row = {
        "dataset": name,
        "model": "TAR-Only",
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
    """Run TAR-only baseline across the three independent datasets."""
    os.makedirs("reports", exist_ok=True)
    results = []

    df_aurex = load_aurex(product_id=6)
    results.append(evaluate_dataset("AUREX_DB", df_aurex, meta={"series": "product_id=6"}))

    df_m5 = load_m5_single_series(base_path="data/raw/m5", random_pick=True, seed=42)
    results.append(evaluate_dataset("M5_WALMART", df_m5, meta={"series": "random_series"}))

    df_fav = load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family="GROCERY I")
    results.append(evaluate_dataset("FAVORITA", df_fav, meta={"series": "store=1,family=GROCERY I"}))

    out = pd.DataFrame(results)
    out_path = os.path.join("reports", "tar_benchmark_results.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
