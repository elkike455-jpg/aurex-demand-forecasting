from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


def _ensure_local_venv_site_packages_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / ".venv" / "Lib" / "site-packages",
        root / ".venv" / "lib" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _ensure_local_pytorch_forecasting_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    repo_root = root / "pytorch-forecasting-main" / "pytorch-forecasting-main"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_local_venv_site_packages_on_path()
_ensure_local_pytorch_forecasting_on_path()

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE


@dataclass
class TFTBenchmarkConfig:
    context_length: int = 28
    prediction_length: int = 28
    max_epochs: int = 25
    refit_epochs: int | None = None
    learning_rate: float = 0.03
    hidden_size: int = 8
    attention_head_size: int = 1
    dropout: float = 0.1
    hidden_continuous_size: int = 4
    batch_size: int = 32
    patience: int = 5
    seed: int = 42


def build_single_series_frame(y: np.ndarray, series_id: str, dates: pd.Series | None = None) -> pd.DataFrame:
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    if dates is None:
        dates = pd.date_range("2000-01-01", periods=n, freq="D")
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    df = pd.DataFrame(
        {
            "time_idx": np.arange(n, dtype=int),
            "date": dates,
            "sales": y.astype(float),
            "series": series_id,
        }
    )
    df["day_of_week"] = df["date"].dt.dayofweek.astype(str).astype("category")
    df["month"] = df["date"].dt.month.astype(str).astype("category")
    return df


def make_tft_datasets(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    config: TFTBenchmarkConfig,
):
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="sales",
        group_ids=["series"],
        min_encoder_length=config.context_length,
        max_encoder_length=config.context_length,
        min_prediction_length=config.prediction_length,
        max_prediction_length=config.prediction_length,
        static_categoricals=["series"],
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(groups=["series"], center=False, transformation="softplus"),
        categorical_encoders={
            "series": NaNLabelEncoder(add_nan=True),
            "day_of_week": NaNLabelEncoder(add_nan=True),
            "month": NaNLabelEncoder(add_nan=True),
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
        predict_mode=False,
    )
    dataset = TimeSeriesDataSet.from_dataset(training, full_df, stop_randomization=True)
    return training, dataset


def make_predict_dataset(base_training: TimeSeriesDataSet, full_df: pd.DataFrame):
    return TimeSeriesDataSet.from_dataset(
        base_training,
        full_df,
        predict=True,
        stop_randomization=True,
    )


def create_tft_model(training: TimeSeriesDataSet, config: TFTBenchmarkConfig) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        loss=RMSE(),
        log_interval=-1,
        log_val_interval=-1,
        reduce_on_plateau_patience=3,
    )


def fit_tft_model(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet | None,
    config: TFTBenchmarkConfig,
    max_epochs: int | None = None,
) -> Tuple[TemporalFusionTransformer, Dict[str, float]]:
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    train_loader = training.to_dataloader(train=True, batch_size=config.batch_size, num_workers=0)
    callbacks = []
    val_loader = None
    if validation is not None:
        val_loader = validation.to_dataloader(train=False, batch_size=config.batch_size, num_workers=0)
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=config.patience,
                verbose=False,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs or config.max_epochs,
        accelerator="cpu",
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        deterministic=True,
    )
    model = create_tft_model(training, config)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    metrics = trainer.callback_metrics
    final_loss = float(metrics.get("train_loss_epoch", metrics.get("train_loss_step", torch.tensor(float("nan")))))
    val_loss = float(metrics.get("val_loss", torch.tensor(float("nan"))))
    epochs_ran = int(trainer.current_epoch + 1)
    return model, {"final_loss": final_loss, "val_loss": val_loss, "epochs_ran": epochs_ran}


def forecast_with_tft(
    model: TemporalFusionTransformer,
    predict_dataset: TimeSeriesDataSet,
    batch_size: int,
) -> np.ndarray:
    pred_loader = predict_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    preds = model.predict(pred_loader, trainer_kwargs={"accelerator": "cpu"})
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=float)
    return preds.reshape(-1)
