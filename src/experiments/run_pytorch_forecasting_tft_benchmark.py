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
from src.models.pytorch_forecasting_tft import (
    TFTBenchmarkConfig,
    build_single_series_frame,
    fit_tft_model,
    forecast_with_tft,
    make_predict_dataset,
    make_tft_datasets,
)


BENCHMARK_PRODUCTS = [
    ("FOODS_3_228_CA_1_validation", "high_demand_stable"),
    ("FOODS_2_044_CA_3_validation", "intermittent"),
    ("HOBBIES_1_133_CA_4_validation", "low_volume"),
]


def _flat_nonflat_label(variance_ratio: float, threshold: float = 0.10) -> str:
    return "flat" if float(variance_ratio) < float(threshold) else "non-flat"


def _evaluate_one(series_id: str, benchmark_label: str, out_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    df = load_m5_single_series(
        base_path=str(root / "data" / "raw" / "m5"),
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
    train_days = len(y_train)
    fit_days = len(y_train) + len(y_val)

    full_series_df = build_single_series_frame(y, series_id, dates=df["date"])
    train_df = full_series_df.iloc[:train_days].copy().reset_index(drop=True)
    fit_df = full_series_df.iloc[:fit_days].copy().reset_index(drop=True)

    config = TFTBenchmarkConfig(
        context_length=OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        prediction_length=OFFICIAL_BENCHMARK_PROTOCOL.test_days,
        max_epochs=25,
        learning_rate=0.03,
        hidden_size=8,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,
        batch_size=16,
        patience=5,
        seed=OFFICIAL_BENCHMARK_PROTOCOL.seed,
    )

    train_dataset, val_source_dataset = make_tft_datasets(train_df, fit_df, config)
    val_predict_dataset = make_predict_dataset(train_dataset, fit_df)
    _, phase1 = fit_tft_model(train_dataset, val_predict_dataset, config, max_epochs=config.max_epochs)

    fit_dataset = type(train_dataset).from_dataset(train_dataset, fit_df, stop_randomization=True)
    refit_epochs = max(1, phase1["epochs_ran"])
    final_model, phase2 = fit_tft_model(fit_dataset, None, config, max_epochs=refit_epochs)

    test_predict_dataset = make_predict_dataset(train_dataset, full_series_df)
    y_pred = forecast_with_tft(final_model, test_predict_dataset, batch_size=1)
    y_pred = np.maximum(y_pred[: len(y_test)], 0.0)

    fit_predict_dataset = make_predict_dataset(train_dataset, fit_df)
    fit_pred = forecast_with_tft(final_model, fit_predict_dataset, batch_size=1)
    fit_target = y_val[-len(fit_pred) :] if len(fit_pred) <= len(y_val) else np.concatenate([y_train, y_val])[-len(fit_pred) :]
    resid_std = float(np.std(fit_target[: len(fit_pred)] - fit_pred[: len(fit_target)])) if len(fit_pred) else 0.0
    spread = 1.959963984540054 * max(resid_std, 1e-6)
    conf_low = np.maximum(y_pred - spread, 0.0)
    conf_up = y_pred + spread

    test_df = df.iloc[fit_days:].copy().reset_index(drop=True)
    pred_df = test_df[["date", "sales"]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["conf_low"] = conf_low
    pred_df["conf_up"] = conf_up
    pred_df["series_id"] = series_id
    pred_df["benchmark_label"] = benchmark_label

    beh = behavioral_metrics(y_test, y_pred)
    variance_ratio = float(beh["variance_ratio"])
    metrics_row = {
        "series_id": series_id,
        "benchmark_label": benchmark_label,
        "model": "PyTorchForecastingTFT",
        "max_days": OFFICIAL_BENCHMARK_PROTOCOL.max_days,
        "train_days": int(len(y_train)),
        "val_days": int(len(y_val)),
        "fit_days": int(fit_days),
        "test_days": int(len(y_test)),
        "context_length": OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        "phase1_val_loss": float(phase1.get("val_loss", np.nan)),
        "final_loss": float(phase2.get("final_loss", np.nan)),
        "refit_epochs": int(phase2.get("epochs_ran", 0)),
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
    training_row = {
        "series_id": series_id,
        "benchmark_label": benchmark_label,
        "model": "PyTorchForecastingTFT",
        "phase1_epochs": int(phase1.get("epochs_ran", 0)),
        "phase1_val_loss": float(phase1.get("val_loss", np.nan)),
        "refit_epochs": int(phase2.get("epochs_ran", 0)),
        "final_loss": float(phase2.get("final_loss", np.nan)),
        "context_length": OFFICIAL_BENCHMARK_PROTOCOL.context_length,
        "train_days": int(len(y_train)),
        "fit_days": int(fit_days),
    }

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(pred_df["date"], pred_df["sales"], label="Real", linewidth=2)
    ax.plot(pred_df["date"], pred_df["y_pred"], label="Predicted", linewidth=2)
    ax.fill_between(pred_df["date"], pred_df["conf_low"], pred_df["conf_up"], alpha=0.15, label="Approx. 95% interval")
    ax.set_title(f"PyTorch Forecasting TFT real vs predicted: {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = out_dir / "figures" / f"{benchmark_label}_{series_id}_real_vs_predicted.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pred_df.to_csv(out_dir / "per_series" / f"{benchmark_label}_{series_id}_predictions.csv", index=False)
    print(
        f"[RUN] {series_id} | final_loss={metrics_row['final_loss']:.6f} | mae={metrics_row['mae']:.6f} | "
        f"vr={metrics_row['variance_ratio']:.6f} | flat={metrics_row['flat_nonflat']}"
    )
    return {
        "metrics": metrics_row,
        "training": training_row,
        "predictions": pred_df,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "reports" / "gnn_benchmarks" / "pytorch_forecasting_tft_impl1"
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
