from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.common_protocol import OFFICIAL_BENCHMARK_PROTOCOL
from src.experiments.stat_benchmark_utils import (
    BENCHMARK_PRODUCTS,
    flat_nonflat_label,
    prepare_protocol_data,
    summarize_predictions,
)
from src.models.hurdle_model import HurdleModel


def _evaluate_one(series_id: str, benchmark_label: str, out_dir: Path) -> Dict[str, object]:
    data = prepare_protocol_data(series_id)
    model = HurdleModel().fit(data["X_fit"], data["y_fit"])
    y_pred, conf_low, conf_up = model.forecast(data["X_test"], data["X_fit"], data["y_fit"])
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)

    metrics = summarize_predictions(data["y_test"], y_pred)
    variance_ratio = metrics["variance_ratio"]
    metrics_row = {
        "series_id": series_id,
        "benchmark_label": benchmark_label,
        "model": "HURDLE",
        "max_days": OFFICIAL_BENCHMARK_PROTOCOL.max_days,
        "train_days": int(len(data["y_train"])),
        "val_days": int(len(data["y_val"])),
        "fit_days": int(len(data["y_fit"])),
        "test_days": int(len(data["y_test"])),
        "context_length": OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        "final_loss": float("nan"),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "pred_std": metrics["pred_std"],
        "real_std": metrics["real_std"],
        "variance_ratio": variance_ratio,
        "trend_correlation": metrics["trend_correlation"],
        "direction_accuracy": metrics["direction_accuracy"],
        "shape_similarity": metrics["shape_similarity"],
        "peak_detection_rate": metrics["peak_detection_rate"],
        "n_peaks_real": metrics["n_peaks_real"],
        "n_peaks_detected": metrics["n_peaks_detected"],
        "flat_nonflat": flat_nonflat_label(variance_ratio),
    }
    training_row = {
        "series_id": series_id,
        "benchmark_label": benchmark_label,
        "model": "HURDLE",
        "occurrence_mode": "constant_probability" if model.occurrence_model is None else "logistic_regression",
        "quantity_mode": "mean_positive_size" if model.quantity_model is None else "ridge_log_size",
        "final_loss": float("nan"),
        "context_length": OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        "train_days": int(len(data["y_train"])),
        "fit_days": int(len(data["y_fit"])),
    }

    pred_df = data["test_feat"][["date", "sales"]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["conf_low"] = conf_low
    pred_df["conf_up"] = conf_up
    pred_df["series_id"] = series_id
    pred_df["benchmark_label"] = benchmark_label
    pred_df.to_csv(out_dir / "per_series" / f"{benchmark_label}_{series_id}_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(pred_df["date"], pred_df["sales"], label="Real", linewidth=2)
    ax.plot(pred_df["date"], pred_df["y_pred"], label="Predicted", linewidth=2)
    ax.fill_between(pred_df["date"], pred_df["conf_low"], pred_df["conf_up"], alpha=0.15, label="Approx. 95% interval")
    ax.set_title(f"HURDLE real vs predicted: {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"{benchmark_label}_{series_id}_real_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[RUN] {series_id} | occurrence={training_row['occurrence_mode']} | quantity={training_row['quantity_mode']} | "
        f"mae={metrics_row['mae']:.6f} | vr={metrics_row['variance_ratio']:.6f} | flat={metrics_row['flat_nonflat']}"
    )
    return {"metrics": metrics_row, "training": training_row, "predictions": pred_df}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "reports" / "gnn_benchmarks" / "hurdle_benchmark_products"
    os.makedirs(out_dir / "figures", exist_ok=True)
    os.makedirs(out_dir / "per_series", exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []
    training_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    for series_id, label in BENCHMARK_PRODUCTS:
        out = _evaluate_one(series_id, label, out_dir)
        metrics_rows.append(out["metrics"])
        training_rows.append(out["training"])
        pred_frames.append(out["predictions"])

    metrics_df = pd.DataFrame(metrics_rows)
    training_summary_df = pd.DataFrame(training_rows)
    predictions_df = pd.concat(pred_frames, ignore_index=True)
    summary_df = metrics_df[
        [
            "series_id",
            "benchmark_label",
            "final_loss",
            "mae",
            "rmse",
            "pred_std",
            "real_std",
            "variance_ratio",
            "trend_correlation",
            "direction_accuracy",
            "shape_similarity",
            "peak_detection_rate",
            "flat_nonflat",
        ]
    ].copy()

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    training_summary_df.to_csv(out_dir / "training_summary.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"[OK] Saved metrics -> {out_dir / 'metrics.csv'}")
    print(f"[OK] Saved training summary -> {out_dir / 'training_summary.csv'}")
    print(f"[OK] Saved summary -> {out_dir / 'summary.csv'}")
    print(f"[OK] Saved predictions -> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
