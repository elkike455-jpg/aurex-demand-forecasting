import argparse
import os
import time

import numpy as np
import pandas as pd

from src.data_loaders.load_amazon import load_amazon_series
from src.data_loaders.load_aurex import load_aurex
from src.data_loaders.load_favorita import load_favorita_series
from src.data_loaders.load_m5 import load_m5_single_series
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


def evaluate_dataset(name, df, meta=None):
    train_df, test_df = train_test_split_ts(df, test_days=365)

    y_train = np.asarray(train_df["sales"], dtype=float)
    y_test = np.asarray(test_df["sales"], dtype=float)

    model = TSBModel().fit(y_train)
    y_pred, _, _ = model.forecast(len(y_test))
    beh = behavioral_metrics(y_test, y_pred)

    row = {
        "dataset": name,
        "model": "TSB-Only",
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

    if meta:
        row.update(meta)

    return row


def append_dataset_result(results, name, loader_fn, meta=None):
    start = time.perf_counter()
    try:
        df = loader_fn()
        if df is None or df.empty:
            print(f"[WARN] Skipped {name}: empty dataframe.")
            return
        results.append(evaluate_dataset(name, df, meta=meta))
        elapsed = time.perf_counter() - start
        print(f"[OK] {name} processed in {elapsed:.1f}s.")
    except Exception as e:
        print(f"[WARN] Skipped {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run TSB benchmark.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="m5,amazon",
        help="Comma-separated subset: aurex,m5,favorita,amazon (default: m5,amazon)",
    )
    parser.add_argument(
        "--amazon-max-rows",
        type=int,
        default=300000,
        help="Max rows to stream from Amazon jsonl.gz (default: 300000).",
    )
    args = parser.parse_args()
    selected = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}

    os.makedirs("reports", exist_ok=True)
    results = []

    if "aurex" in selected:
        append_dataset_result(
            results,
            "AUREX_DB",
            lambda: load_aurex(product_id=6),
            meta={"series": "product_id=6"},
        )

    if "m5" in selected:
        append_dataset_result(
            results,
            "M5_WALMART",
            lambda: load_m5_single_series(base_path="data/raw/m5", random_pick=True, seed=42),
            meta={"series": "random_series"},
        )

    fav_family = "CLEANING"
    if "favorita" in selected:
        append_dataset_result(
            results,
            "FAVORITA",
            lambda: load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family=fav_family),
            meta={"series": f"store=1,family={fav_family}"},
        )

    amazon_candidates = [
        "Health_and_Household.jsonl.gz",
        "Home_and_Kitchen.jsonl.gz",
        "Cell_Phones_and_Accessories.jsonl.gz",
    ]
    amazon_base = "data/raw/amazon_2023/review_categories"
    amazon_file = next((f for f in amazon_candidates if os.path.exists(os.path.join(amazon_base, f))), None)

    if "amazon" in selected and amazon_file is not None:
        append_dataset_result(
            results,
            "AMAZON_2023",
            lambda: load_amazon_series(
                base_path=amazon_base,
                filename=amazon_file,
                top_rank=1,
                max_rows=args.amazon_max_rows,
            ),
            meta={
                "series": (
                    f"category={amazon_file.replace('.jsonl.gz', '')},"
                    f"top_rank=1,max_rows={args.amazon_max_rows}"
                )
            },
        )

    if not results:
        raise RuntimeError("No dataset could be processed. Check local data files and DB connectivity.")

    out = pd.DataFrame(results)
    out_path = os.path.join("reports", "tsb_benchmark_results.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
