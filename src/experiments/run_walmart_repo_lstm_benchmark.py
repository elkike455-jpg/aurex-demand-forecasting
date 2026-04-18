from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loaders.load_m5 import load_m5_single_series
from src.experiments.common_protocol import OFFICIAL_BENCHMARK_PROTOCOL, split_series_protocol
from src.metrics.behavioral_metrics import behavioral_metrics
from src.models.walmart_repo_bilstm import WalmartRepoBiLSTMForecastModel


BENCHMARK_PRODUCTS = [
    ("FOODS_3_228_CA_1_validation", "high_demand_stable"),
    ("FOODS_2_044_CA_3_validation", "intermittent"),
    ("HOBBIES_1_133_CA_4_validation", "low_volume"),
]


def _flat_nonflat_label(variance_ratio: float, threshold: float = 0.10) -> str:
    return "flat" if float(variance_ratio) < float(threshold) else "non-flat"


def _evaluate_one(series_id: str, benchmark_label: str, out_dir: Path) -> Dict[str, object]:
    df = load_m5_single_series(
        base_path="data/raw/m5",
        random_pick=False,
        series_id=series_id,
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.iloc[-OFFICIAL_BENCHMARK_PROTOCOL.max_days :].copy().reset_index(drop=True)

    y = df["sales"].to_numpy(dtype=float)
    y_train, y_val, y_test = split_series_protocol(
        y,
        val_days=OFFICIAL_BENCHMARK_PROTOCOL.val_days,
        test_days=OFFICIAL_BENCHMARK_PROTOCOL.test_days,
    )
    y_fit = np.concatenate([y_train, y_val])

    model = WalmartRepoBiLSTMForecastModel(
        context_length=OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        units=50,
        dropout=0.2,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        seed=OFFICIAL_BENCHMARK_PROTOCOL.seed,
        verbose=0,
    )
    model.fit(y_fit, scaling_reference=y_train)
    y_pred, conf_low, conf_up = model.forecast(len(y_test), y_fit)

    val_end = len(y_train) + len(y_val)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    pred_df = test_df[["date", "sales"]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["conf_low"] = conf_low
    pred_df["conf_up"] = conf_up
    pred_df["series_id"] = series_id
    pred_df["benchmark_label"] = benchmark_label

    beh = behavioral_metrics(y_test, y_pred)
    variance_ratio = float(beh["variance_ratio"])
    row = {
        "series_id": series_id,
        "benchmark_label": benchmark_label,
        "model": "WalmartRepoBiLSTM",
        "max_days": OFFICIAL_BENCHMARK_PROTOCOL.max_days,
        "train_days": int(len(y_train)),
        "val_days": int(len(y_val)),
        "fit_days": int(len(y_fit)),
        "test_days": int(len(y_test)),
        "context_length": OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        "final_loss": float(model.training_history_.get("final_loss", np.nan)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "pred_std": float(np.std(y_pred)),
        "real_std": float(np.std(y_test)),
        "variance_ratio": variance_ratio,
        "trend_correlation": float(beh["trend_correlation"]),
        "direction_accuracy": float(beh["direction_accuracy"]),
        "shape_similarity": float(beh["shape_similarity"]),
        "peak_detection_rate": float(beh["peak_detection_rate"]),
        "n_peaks_real": int(beh["n_peaks_real"]),
        "n_peaks_detected": int(beh["n_peaks_detected"]),
        "flat_nonflat": _flat_nonflat_label(variance_ratio),
    }

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(pred_df["date"], pred_df["sales"], label="Real", linewidth=2)
    ax.plot(pred_df["date"], pred_df["y_pred"], label="Predicted", linewidth=2)
    ax.fill_between(pred_df["date"], pred_df["conf_low"], pred_df["conf_up"], alpha=0.15, label="Approx. 95% interval")
    ax.set_title(f"Walmart repo BiLSTM real vs predicted: {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = out_dir / "figures" / f"{benchmark_label}_{series_id}_real_vs_predicted.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pred_path = out_dir / "per_series" / f"{benchmark_label}_{series_id}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(
        f"[RUN] {series_id} | loss={row['final_loss']:.6f} | mae={row['mae']:.6f} | "
        f"vr={row['variance_ratio']:.6f} | flat={row['flat_nonflat']}"
    )
    return {"metrics": row, "predictions": pred_df}


def main() -> None:
    out_dir = Path("reports") / "gnn_benchmarks" / "walmart_repo_lstm_impl2"
    os.makedirs(out_dir / "figures", exist_ok=True)
    os.makedirs(out_dir / "per_series", exist_ok=True)

    rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for series_id, label in BENCHMARK_PRODUCTS:
        out = _evaluate_one(series_id, label, out_dir)
        rows.append(out["metrics"])
        pred_frames.append(out["predictions"])

    metrics_df = pd.DataFrame(rows)
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
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"[OK] Saved metrics -> {out_dir / 'metrics.csv'}")
    print(f"[OK] Saved summary -> {out_dir / 'summary.csv'}")
    print(f"[OK] Saved predictions -> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
