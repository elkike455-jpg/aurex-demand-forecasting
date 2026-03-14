import argparse
import os

import numpy as np
import pandas as pd

from src.experiments.run_cross_market_peak_comparison import (
    _load_amazon_selected,
    _load_favorita_selected,
    _load_m5_selected,
)
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.tsb_model import TSBModel


def train_test_split_ts(df, test_days=365):
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) <= test_days + 30:
        test_days = max(30, int(len(df) * 0.2))
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def evaluate_series(df):
    train_df, test_df = train_test_split_ts(df, test_days=365)
    y_train = np.asarray(train_df["sales"], dtype=float)
    y_test = np.asarray(test_df["sales"], dtype=float)

    model = TSBModel().fit(y_train)
    y_pred, _, _ = model.forecast(len(y_test))
    beh = behavioral_metrics(y_test, y_pred)

    row = {
        "n_days_total": len(df),
        "train_days": len(train_df),
        "test_days": len(test_df),
        "start_date": str(pd.to_datetime(df["date"]).min().date()),
        "end_date": str(pd.to_datetime(df["date"]).max().date()),
        "zero_rate_train": float((y_train == 0).mean()),
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mape": mape(y_test, y_pred),
        "bias": bias(y_test, y_pred),
        "tsb_alpha": model.alpha,
        "tsb_beta": model.beta,
    }
    row.update(beh)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run TSB experiment on 3 cross-market inferred product triplets.")
    parser.add_argument(
        "--triplets-csv",
        type=str,
        default="reports/cross_market_peak_comparison/inferred_triplets.csv",
        help="CSV with inferred product triplets (M5_WALMART, FAVORITA, AMAZON_2023).",
    )
    parser.add_argument("--amazon-file", type=str, default="Health_and_Household.jsonl.gz")
    parser.add_argument("--amazon-max-rows", type=int, default=100000)
    parser.add_argument("--out-dir", type=str, default="reports/tsb_triplet_experiment")
    args = parser.parse_args()

    if not os.path.exists(args.triplets_csv):
        raise FileNotFoundError(f"Missing triplets file: {args.triplets_csv}")

    triplets = pd.read_csv(args.triplets_csv)
    required_cols = {"inferred_product", "M5_WALMART", "FAVORITA", "AMAZON_2023"}
    if not required_cols.issubset(triplets.columns):
        raise ValueError(f"Triplets CSV must contain columns: {sorted(required_cols)}")

    os.makedirs(args.out_dir, exist_ok=True)

    m5_ids = triplets["M5_WALMART"].astype(str).tolist()
    fav_ids = triplets["FAVORITA"].astype(str).tolist()
    amz_ids = triplets["AMAZON_2023"].astype(str).tolist()

    series_map_by_ds = {
        "M5_WALMART": _load_m5_selected("data/raw/m5", m5_ids),
        "FAVORITA": _load_favorita_selected("data/raw/favorita", fav_ids, store_nbr=1),
        "AMAZON_2023": _load_amazon_selected(
            "data/raw/amazon_2023/review_categories",
            args.amazon_file,
            amz_ids,
            max_rows=args.amazon_max_rows,
        ),
    }

    rows = []
    for _, trip in triplets.iterrows():
        inferred_product = trip["inferred_product"]
        for ds in ["M5_WALMART", "FAVORITA", "AMAZON_2023"]:
            sid = str(trip[ds])
            if sid not in series_map_by_ds[ds]:
                raise RuntimeError(f"Missing series for {ds}: {sid}")

            df = series_map_by_ds[ds][sid].copy()
            metrics = evaluate_series(df)
            metrics["inferred_product"] = inferred_product
            metrics["dataset"] = ds
            metrics["series_id"] = sid
            rows.append(metrics)

    out = pd.DataFrame(rows).sort_values(["inferred_product", "dataset"]).reset_index(drop=True)
    out_path = os.path.join(args.out_dir, "tsb_triplet_results.csv")
    out.to_csv(out_path, index=False)

    summary_cols = [
        "inferred_product",
        "dataset",
        "series_id",
        "mae",
        "rmse",
        "mape",
        "bias",
        "peak_detection_rate",
        "trend_correlation",
        "direction_accuracy",
        "shape_similarity",
        "variance_ratio",
        "tsb_alpha",
        "tsb_beta",
    ]
    summary = out[summary_cols].copy()
    summary_path = os.path.join(args.out_dir, "tsb_triplet_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
