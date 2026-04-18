from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.data_loaders.load_amazon import load_amazon_series
from src.data_loaders.load_favorita import load_favorita_series
from src.data_loaders.load_m5 import load_m5_single_series
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.deep_challenger_models import (
    GRUForecastModel,
    DeepARForecastModel,
    mae,
    rmse,
    smape,
    train_test_split_series,
)


def _resolve_amazon_candidate(base_path: str, candidates: List[str]) -> str | None:
    for filename in candidates:
        direct = os.path.join(base_path, filename)
        if os.path.exists(direct):
            return filename
        if filename.endswith(".jsonl.gz"):
            alt = filename[:-3]
            if os.path.exists(os.path.join(base_path, alt)):
                return alt
        elif filename.endswith(".jsonl"):
            alt = filename + ".gz"
            if os.path.exists(os.path.join(base_path, alt)):
                return alt
    return None


def evaluate_model(
    dataset_name: str,
    df: pd.DataFrame,
    model_name: str,
    model_factory: Callable[[], object],
    meta: Dict[str, str] | None = None,
) -> Dict[str, object]:
    df = df.sort_values("date").reset_index(drop=True)
    y = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_train, y_test = train_test_split_series(y, test_days=365)

    model = model_factory()
    model.fit(y_train)
    y_pred, _, _ = model.forecast(len(y_test), y_train)
    beh = behavioral_metrics(y_test, y_pred)

    row: Dict[str, object] = {
        "dataset": dataset_name,
        "model": model_name,
        "n_days_total": int(len(y)),
        "train_days": int(len(y_train)),
        "test_days": int(len(y_test)),
        "start_date": str(pd.to_datetime(df["date"]).min().date()),
        "end_date": str(pd.to_datetime(df["date"]).max().date()),
        "zero_rate_train": float((y_train == 0).mean()),
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "smape": smape(y_test, y_pred),
        "final_loss": getattr(model, "training_history_", {}).get("final_loss"),
        "context_length": getattr(model, "config", None).context_length,
        "hidden_size": getattr(model, "config", None).hidden_size,
        "epochs": getattr(model, "config", None).epochs,
    }
    row.update(beh)
    if meta:
        row.update(meta)
    return row


def append_dataset_results(
    results: List[Dict[str, object]],
    dataset_name: str,
    loader_fn: Callable[[], pd.DataFrame],
    model_factories: Dict[str, Callable[[], object]],
    meta: Dict[str, str] | None = None,
) -> None:
    try:
        df = loader_fn()
        if df is None or df.empty:
            print(f"[WARN] Skipped {dataset_name}: empty dataframe.")
            return
        for model_name, factory in model_factories.items():
            print(f"[RUN] {dataset_name} - {model_name}")
            results.append(evaluate_model(dataset_name, df, model_name, factory, meta=meta))
    except Exception as exc:
        print(f"[WARN] Skipped {dataset_name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep challenger benchmark for missing deep models.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="m5,favorita,amazon",
        help="Comma-separated subset: m5,favorita,amazon",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--amazon-max-rows", type=int, default=100000)
    args = parser.parse_args()

    selected = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}
    os.makedirs("reports", exist_ok=True)

    model_factories: Dict[str, Callable[[], object]] = {
        "GRU": lambda: GRUForecastModel(
            context_length=args.context_length,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
        ),
        "DeepAR": lambda: DeepARForecastModel(
            context_length=args.context_length,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
        ),
    }

    results: List[Dict[str, object]] = []

    if "m5" in selected:
        append_dataset_results(
            results,
            "M5_WALMART",
            lambda: load_m5_single_series(base_path="data/raw/m5", random_pick=True, seed=42),
            model_factories,
            meta={"series": "random_series"},
        )

    if "favorita" in selected:
        append_dataset_results(
            results,
            "FAVORITA",
            lambda: load_favorita_series(base_path="data/raw/favorita", store_nbr=1, family="CLEANING"),
            model_factories,
            meta={"series": "store=1,family=CLEANING"},
        )

    amazon_base = "data/raw/amazon_2023/review_categories"
    amazon_file = _resolve_amazon_candidate(
        amazon_base,
        [
            "Health_and_Household.jsonl.gz",
            "Health_and_Household.jsonl",
            "Home_and_Kitchen.jsonl.gz",
            "Home_and_Kitchen.jsonl",
        ],
    )
    if "amazon" in selected and amazon_file is not None:
        append_dataset_results(
            results,
            "AMAZON_2023",
            lambda: load_amazon_series(
                base_path=amazon_base,
                filename=amazon_file,
                top_rank=1,
                max_rows=args.amazon_max_rows,
            ),
            model_factories,
            meta={"series": f"category={amazon_file},top_rank=1,max_rows={args.amazon_max_rows}"},
        )

    if not results:
        raise RuntimeError("No dataset could be processed. Check local data files.")

    out = pd.DataFrame(results)
    out_path = os.path.join("reports", "deep_challenger_benchmark_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
