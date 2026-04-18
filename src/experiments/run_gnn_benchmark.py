from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_loaders.load_m5_panel import load_m5_panel_subset
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.deep_challenger_models import GRUForecastModel, mae, rmse, smape
from src.models.graph_forecast_model import GraphDemandForecastModel, build_m5_product_graph


def _panel_split(panel: np.ndarray, val_steps: int, test_steps: int):
    total_steps = panel.shape[1]
    if total_steps <= val_steps + test_steps + 30:
        raise ValueError(
            f"Panel length {total_steps} is too short for val_steps={val_steps} and test_steps={test_steps}."
        )
    train = panel[:, : total_steps - val_steps - test_steps]
    val = panel[:, total_steps - val_steps - test_steps : total_steps - test_steps]
    test = panel[:, total_steps - test_steps :]
    return train, val, test


def _aggregate_behavioral(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for idx in range(y_true.shape[0]):
        rows.append(behavioral_metrics(y_true[idx], y_pred[idx]))
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in df.columns}


def evaluate_graph_model(
    model: GraphDemandForecastModel,
    train_panel: np.ndarray,
    test_panel: np.ndarray,
    adj: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    y_pred, _, _ = model.forecast(test_panel.shape[1], train_panel, adj=adj)
    rows = []
    for idx, meta in metadata.reset_index(drop=True).iterrows():
        beh = behavioral_metrics(test_panel[idx], y_pred[idx])
        rows.append(
            {
                "model": "GNN_HURDLE" if model.config.hurdle_mode else "GNN_MSE",
                "series_id": meta["id"],
                "item_id": meta["item_id"],
                "dept_id": meta["dept_id"],
                "cat_id": meta["cat_id"],
                "store_id": meta["store_id"],
                "state_id": meta["state_id"],
                "zero_rate_train": float(np.mean(train_panel[idx] == 0)),
                "mae": mae(test_panel[idx], y_pred[idx]),
                "rmse": rmse(test_panel[idx], y_pred[idx]),
                "smape": smape(test_panel[idx], y_pred[idx]),
                **beh,
            }
        )
    return pd.DataFrame(rows)


def evaluate_univariate_gru(
    train_panel: np.ndarray,
    test_panel: np.ndarray,
    metadata: pd.DataFrame,
    context_length: int,
    epochs: int,
    hidden_size: int,
) -> pd.DataFrame:
    rows = []
    for idx, meta in metadata.reset_index(drop=True).iterrows():
        model = GRUForecastModel(
            context_length=context_length,
            epochs=epochs,
            hidden_size=hidden_size,
        )
        model.fit(train_panel[idx])
        y_pred, _, _ = model.forecast(test_panel.shape[1], train_panel[idx])
        beh = behavioral_metrics(test_panel[idx], y_pred)
        rows.append(
            {
                "model": "GRU_UNIVARIATE",
                "series_id": meta["id"],
                "item_id": meta["item_id"],
                "dept_id": meta["dept_id"],
                "cat_id": meta["cat_id"],
                "store_id": meta["store_id"],
                "state_id": meta["state_id"],
                "zero_rate_train": float(np.mean(train_panel[idx] == 0)),
                "mae": mae(test_panel[idx], y_pred),
                "rmse": rmse(test_panel[idx], y_pred),
                "smape": smape(test_panel[idx], y_pred),
                **beh,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a first graph-based demand forecasting prototype on an M5 panel.")
    parser.add_argument("--num-products", type=int, default=24)
    parser.add_argument("--state-id", type=str, default="CA")
    parser.add_argument("--store-id", type=str, default="")
    parser.add_argument("--cat-id", type=str, default="")
    parser.add_argument("--dept-id", type=str, default="")
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-steps", type=int, default=28)
    parser.add_argument("--test-steps", type=int, default=28)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corr-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=365)
    parser.add_argument("--min-nonzero-days", type=int, default=28)
    parser.add_argument("--hurdle-head", action="store_true")
    parser.add_argument("--skip-gru-baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    panel = load_m5_panel_subset(
        base_path="data/raw/m5",
        num_products=args.num_products,
        seed=args.seed,
        state_id=args.state_id or None,
        store_id=args.store_id or None,
        cat_id=args.cat_id or None,
        dept_id=args.dept_id or None,
        min_nonzero_days=args.min_nonzero_days,
        max_days=args.max_days,
    )

    sales = np.asarray(panel["sales"], dtype=np.float32)
    metadata = panel["metadata"].copy().reset_index(drop=True)
    train_panel, val_panel, test_panel = _panel_split(sales, val_steps=args.val_steps, test_steps=args.test_steps)
    train_plus_val = np.concatenate([train_panel, val_panel], axis=1)
    adj = build_m5_product_graph(metadata, train_plus_val, top_k=args.top_k, corr_weight=args.corr_weight)

    nonzero_offdiag = int(np.count_nonzero(adj) - len(adj))
    density = float(nonzero_offdiag / max(len(adj) * max(len(adj) - 1, 1), 1))
    print(
        f"[INFO] panel shape={sales.shape}, train={train_panel.shape}, val={val_panel.shape}, test={test_panel.shape}"
    )
    print(f"[INFO] adjacency shape={adj.shape}, nonzero_offdiag={nonzero_offdiag}, density={density:.4f}")

    model = GraphDemandForecastModel(
        context_length=args.context_length,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        hurdle_mode=args.hurdle_head,
    )
    model.fit(train_plus_val, adj)
    gnn_df = evaluate_graph_model(model, train_plus_val, test_panel, adj, metadata)

    result_frames = [gnn_df]
    if not args.skip_gru_baseline:
        gru_df = evaluate_univariate_gru(
            train_plus_val,
            test_panel,
            metadata,
            context_length=args.context_length,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
        )
        result_frames.append(gru_df)

    results_df = pd.concat(result_frames, ignore_index=True)
    summary_df = (
        results_df.groupby("model", as_index=False)[
            [
                "mae",
                "rmse",
                "smape",
                "peak_detection_rate",
                "variance_ratio",
                "trend_correlation",
                "shape_similarity",
                "direction_accuracy",
            ]
        ]
        .mean()
        .sort_values("mae")
        .reset_index(drop=True)
    )

    out_dir = os.path.join("reports", "gnn_benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "m5_gnn_product_results.csv")
    summary_path = os.path.join(out_dir, "m5_gnn_summary.csv")
    results_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved: {detail_path}")
    print(f"[OK] Saved: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
