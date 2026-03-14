import os
import argparse
import time
import numpy as np
import pandas as pd

from src.data_loaders.load_aurex import load_aurex
from src.data_loaders.load_amazon import load_amazon_series
from src.data_loaders.load_m5 import load_m5_single_series
from src.data_loaders.load_favorita import load_favorita_series
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.regime_forecast_engine import RegimeForecastEngine


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
    """Build consistent temporal features used by all compared models."""
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
    """Run full regime benchmark on one dataset and return one result row."""
    feat_df, feature_cols = build_features(df)
    train_feat, test_feat = train_test_split_ts(feat_df, test_days=365)

    y_train = train_feat["sales"].values.astype(float)
    y_test = test_feat["sales"].values.astype(float)
    X_train = train_feat[feature_cols].values.astype(float)
    X_test = test_feat[feature_cols].values.astype(float)

    engine = RegimeForecastEngine()
    y_pred, _, _, model_name = engine.run(y_train, X_train, y_test, X_test)
    beh = behavioral_metrics(y_test, y_pred)

    row = {
        "dataset": name,
        "model": model_name,
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
        # Selection diagnostics help explain regime decisions in reports.
        "selection_mode": engine.selection_info.get("regime_mode"),
        "selection_reason": engine.selection_info.get("reason"),
        "intermittent_share": engine.selection_info.get("intermittent_share"),
        "val_mae_hurdle": engine.selection_info.get("val_mae_hurdle"),
        "val_mae_sarimax": engine.selection_info.get("val_mae_sarimax"),
    }
    row.update(beh)

    if meta:
        row.update(meta)

    return row


def append_dataset_result(results, name, loader_fn, meta=None):
    """Append dataset result row; keep run alive if one source fails."""
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
    """Run regime benchmark across independent datasets with comparable categories."""
    parser = argparse.ArgumentParser(description="Run regime benchmark.")
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

    # Keep Favorita aligned with M5 household-like demand.
    fav_family = "CLEANING"
    if "favorita" in selected:
        append_dataset_result(
            results,
            "FAVORITA",
            lambda: load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family=fav_family),
            meta={"series": f"store=1,family={fav_family}"},
        )

    # Optional section: Amazon 2023 review-based demand proxy.
    # Prioritize household-like category to stay close to M5 HOUSEHOLD.
    amazon_candidates = [
        "Health_and_Household.jsonl.gz",
        "Home_and_Kitchen.jsonl.gz",
        "Cell_Phones_and_Accessories.jsonl.gz",  # fallback if only this file is available
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
            meta={"series": f"category={amazon_file.replace('.jsonl.gz', '')},top_rank=1,max_rows={args.amazon_max_rows}"},
        )

    if not results:
        raise RuntimeError("No dataset could be processed. Check local data files and DB connectivity.")

    out = pd.DataFrame(results)
    out_path = os.path.join("reports", "regime_benchmark_results.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
