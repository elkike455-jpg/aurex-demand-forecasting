from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
M5_BASE = REPO_ROOT / "data" / "raw" / "m5"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "biweekly_bilstm_attention"

SERIES_CONFIG = {
    "high_demand": {
        "series_id": "FOODS_3_228_CA_1_validation",
        "model_name": "BiLSTM-HighDemand",
        "loss_mode": "huber",
        "early_stop_metric": "val_loss",
        "smooth_predictions": True,
    },
    "intermittent": {
        "series_id": "FOODS_2_044_CA_3_validation",
        "model_name": "BiLSTM-Intermittent",
        "loss_mode": "weighted_huber",
        "early_stop_metric": "val_pdr",
        "smooth_predictions": False,
    },
}


@dataclass(frozen=True)
class BiweeklyBiLSTMConfig:
    seed: int = 42
    lookback: int = 28
    horizon: int = 14
    train_days: int = 365 * 3
    validation_cycles: int = 26
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    peak_multiplier: float = 1.5
    peak_weight: float = 3.0
    min_delta: float = 1e-4

    @property
    def validation_days(self) -> int:
        # 26 prediction windows need 28 warmup days plus 26 * 14 target days.
        return self.lookback + self.validation_cycles * self.horizon


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sales_files() -> List[Path]:
    paths = []
    for name in ("sales_train_evaluation.csv", "sales_train_validation.csv"):
        path = M5_BASE / name
        if path.exists():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"Missing M5 sales file under {M5_BASE}")
    return paths


def load_m5_series_with_calendar(series_id: str) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    row_df = pd.DataFrame()
    d_cols: List[str] = []
    for sales_path in sales_files():
        sales_header = pd.read_csv(sales_path, nrows=0).columns.tolist()
        d_cols = [col for col in sales_header if col.startswith("d_")]
        candidate = pd.read_csv(sales_path, usecols=id_cols + d_cols)
        candidate = candidate.loc[candidate["id"] == series_id].copy()
        if not candidate.empty:
            row_df = candidate
            break
    if row_df.empty:
        searched = ", ".join(path.name for path in sales_files())
        raise ValueError(f"Series not found in M5 sales files ({searched}): {series_id}")
    row = row_df.iloc[0]

    calendar = pd.read_csv(M5_BASE / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        f"snap_{row['state_id']}",
    ]
    existing_calendar_cols = [col for col in calendar_cols if col in calendar.columns]
    prices = pd.read_csv(M5_BASE / "sell_prices.csv")

    df = pd.DataFrame({"d": d_cols, "sales": row[d_cols].astype(float).to_numpy()})
    df = df.merge(calendar[existing_calendar_cols], on="d", how="left")

    price_sub = prices.loc[
        (prices["store_id"] == row["store_id"]) & (prices["item_id"] == row["item_id"]),
        ["wm_yr_wk", "sell_price"],
    ].drop_duplicates("wm_yr_wk")
    df = df.merge(price_sub, on="wm_yr_wk", how="left")
    df["price"] = df["sell_price"].ffill().bfill().fillna(0.0)
    df = df.drop(columns=["sell_price"], errors="ignore")

    for col in id_cols:
        df[col] = row[col]
    df["series_id"] = series_id
    return df.sort_values("date").reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    out["sales"] = out["sales"].astype(float).fillna(0.0)
    out["price"] = out["price"].astype(float).ffill().bfill().fillna(0.0)

    # Lags and rolling statistics are shifted by one day so the current target does not leak.
    lagged_sales = out["sales"].shift(1)
    for lag in (7, 14, 21, 28):
        out[f"lag_{lag}"] = out["sales"].shift(lag)
    for window in (7, 14):
        out[f"rolling_mean_{window}"] = lagged_sales.rolling(window, min_periods=1).mean()
        out[f"rolling_std_{window}"] = lagged_sales.rolling(window, min_periods=2).std()
    out["rolling_mean_28"] = lagged_sales.rolling(28, min_periods=1).mean()

    out["day_of_week"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(float)

    holiday_calendar = USFederalHolidayCalendar()
    holidays = holiday_calendar.holidays(out["date"].min(), out["date"].max())
    out["is_us_federal_holiday"] = out["date"].isin(holidays).astype(float)

    event_cols = [col for col in ["event_name_1", "event_name_2"] if col in out.columns]
    out["has_m5_event"] = 0.0
    for col in event_cols:
        out["has_m5_event"] = np.maximum(out["has_m5_event"], out[col].notna().astype(float))

    snap_cols = [col for col in out.columns if col.startswith("snap_")]
    out["custom_promo_flag"] = out[snap_cols].max(axis=1).astype(float) if snap_cols else 0.0

    # Price offer proxy: current price below the trailing 28-day max price.
    trailing_price_max = out["price"].shift(1).rolling(28, min_periods=1).max().fillna(out["price"])
    out["discount_pct"] = ((trailing_price_max - out["price"]) / trailing_price_max.replace(0.0, np.nan)).clip(0.0, 1.0)
    out["discount_pct"] = out["discount_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # The biweekly cycle is encoded as sin/cos over the 26 two-week periods in a year.
    day_of_year_zero = out["date"].dt.dayofyear - 1
    out["biweek_id"] = ((day_of_year_zero // 14) % 26).astype(int)
    out["biweek_sin"] = np.sin(2.0 * np.pi * out["biweek_id"] / 26.0)
    out["biweek_cos"] = np.cos(2.0 * np.pi * out["biweek_id"] / 26.0)

    feature_cols = [
        "sales",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        "rolling_mean_28",
        "day_of_week",
        "week_of_year",
        "month",
        "is_weekend",
        "is_us_federal_holiday",
        "has_m5_event",
        "custom_promo_flag",
        "price",
        "discount_pct",
        "biweek_sin",
        "biweek_cos",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def trim_to_protocol(feat_df: pd.DataFrame, config: BiweeklyBiLSTMConfig) -> pd.DataFrame:
    total_needed = config.train_days + config.validation_days
    if len(feat_df) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} days for train_days={config.train_days} and "
            f"validation_days={config.validation_days}; got {len(feat_df)}."
        )
    return feat_df.iloc[-total_needed:].copy().reset_index(drop=True)


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols].to_numpy(dtype=float))
    target_scaler.fit(train_df[["sales"]].to_numpy(dtype=float))
    return feature_scaler, target_scaler


def transform_features(df: pd.DataFrame, feature_cols: List[str], scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[feature_cols].to_numpy(dtype=float)).astype(np.float32)


def transform_target(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel().astype(np.float32)


def inverse_target(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    arr = scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()
    return np.maximum(arr, 0.0)


def actual_peak_flags(y: np.ndarray, rolling_mean_28: np.ndarray, multiplier: float) -> np.ndarray:
    threshold = np.asarray(rolling_mean_28, dtype=float) * float(multiplier)
    return np.asarray(y, dtype=float) > threshold


def make_sequences(
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    raw_target: np.ndarray,
    rolling_mean_28: np.ndarray,
    lookback: int,
    horizon: int,
    peak_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    peak_rows: List[np.ndarray] = []
    max_start = len(scaled_target) - lookback - horizon + 1
    for start in range(max_start):
        mid = start + lookback
        end = mid + horizon
        x_rows.append(scaled_features[start:mid])
        y_rows.append(scaled_target[mid:end])
        peak_rows.append(actual_peak_flags(raw_target[mid:end], rolling_mean_28[mid:end], peak_multiplier))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(peak_rows, dtype=np.float32),
    )


class BiLSTMAttentionNet(nn.Module):
    def __init__(self, n_features: int, horizon: int):
        super().__init__()
        # First bidirectional LSTM keeps all timesteps so attention can learn important days in the 28-day lookback.
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.attention_score = nn.Linear(256, 1)
        # Second bidirectional LSTM compresses the attended sequence into one context representation.
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(32, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm1(x)
        seq = self.dropout(seq)
        weights = torch.softmax(self.attention_score(seq), dim=1)
        attended_seq = seq * weights
        _, (hidden, _) = self.lstm2(attended_seq)
        context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden_out = self.dense(context)
        return self.output(hidden_out)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    peak_mask: torch.Tensor,
    peak_weight: float,
    delta: float = 1.0,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    base = torch.where(abs_diff <= delta, 0.5 * diff.pow(2), delta * (abs_diff - 0.5 * delta))
    under_pred_peak = (target > pred) & (peak_mask > 0.5)
    weights = torch.ones_like(base)
    weights = torch.where(under_pred_peak, torch.full_like(weights, float(peak_weight)), weights)
    return (base * weights).mean()


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rolling_mean_28: np.ndarray,
    peak_multiplier: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual_peaks = actual_peak_flags(y_true, rolling_mean_28, peak_multiplier)
    predicted_peaks = actual_peak_flags(y_pred, rolling_mean_28, peak_multiplier)
    detected = int(np.logical_and(actual_peaks, predicted_peaks).sum())
    total_actual = int(actual_peaks.sum())
    pdr = float(detected / total_actual) if total_actual > 0 else 1.0
    actual_var = float(np.var(y_true))
    pred_var = float(np.var(y_pred))
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "pdr": pdr,
        "actual_peak_days": total_actual,
        "detected_peak_days": detected,
        "predicted_peak_days": int(predicted_peaks.sum()),
        "variance_ratio": float(pred_var / actual_var) if actual_var > 1e-8 else (1.0 if pred_var <= 1e-8 else np.inf),
        "missed_all_peaks": bool(total_actual > 0 and detected == 0),
    }


def evaluate_biweekly(
    model: BiLSTMAttentionNet,
    val_df: pd.DataFrame,
    scaled_val_features: np.ndarray,
    target_scaler: MinMaxScaler,
    config: BiweeklyBiLSTMConfig,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for biweek in range(config.validation_cycles):
            start = biweek * config.horizon
            input_start = start
            input_end = start + config.lookback
            target_start = input_end
            target_end = target_start + config.horizon

            x_input = torch.from_numpy(scaled_val_features[input_start:input_end][None, :, :]).to(device)
            y_pred_scaled = model(x_input).detach().cpu().numpy().reshape(-1)
            y_pred = inverse_target(y_pred_scaled, target_scaler)
            y_real = val_df["sales"].iloc[target_start:target_end].to_numpy(dtype=float)
            rolling_mean = val_df["rolling_mean_28"].iloc[target_start:target_end].to_numpy(dtype=float)
            metrics = compute_window_metrics(y_real, y_pred, rolling_mean, config.peak_multiplier)
            rows.append(
                {
                    "biweek": int(biweek + 1),
                    "window_start": str(val_df["date"].iloc[target_start].date()),
                    "window_end": str(val_df["date"].iloc[target_end - 1].date()),
                    **metrics,
                }
            )
            pred_peaks = actual_peak_flags(y_pred, rolling_mean, config.peak_multiplier)
            actual_peaks = actual_peak_flags(y_real, rolling_mean, config.peak_multiplier)
            for offset, (date, actual, pred, threshold, pred_peak, actual_peak) in enumerate(
                zip(
                    val_df["date"].iloc[target_start:target_end],
                    y_real,
                    y_pred,
                    rolling_mean * config.peak_multiplier,
                    pred_peaks,
                    actual_peaks,
                )
            ):
                prediction_rows.append(
                    {
                        "biweek": int(biweek + 1),
                        "horizon_day": int(offset + 1),
                        "date": str(pd.Timestamp(date).date()),
                        "sales": float(actual),
                        "y_pred": float(pred),
                        "peak_threshold": float(threshold),
                        "is_actual_peak": bool(actual_peak),
                        "is_predicted_peak": bool(pred_peak),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def plot_loss_curves(history: pd.DataFrame, figure_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss", linewidth=2)
    ax1.plot(history["epoch"], history["val_loss"], label="validation loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    if "val_pdr" in history.columns:
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["val_pdr"], label="validation PDR", color="#2ca02c", linewidth=2)
        ax2.set_ylabel("PDR")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_biweekly_predictions(pred_df: pd.DataFrame, figure_path: Path, title: str) -> None:
    fig, axes = plt.subplots(13, 2, figsize=(16, 34), sharey=False)
    axes = axes.ravel()
    for idx, biweek in enumerate(sorted(pred_df["biweek"].unique())):
        ax = axes[idx]
        sub = pred_df.loc[pred_df["biweek"] == biweek].copy()
        dates = pd.to_datetime(sub["date"])
        ax.plot(dates, sub["sales"], label="Real", linewidth=1.8)
        ax.plot(dates, sub["y_pred"], label="Predicted", linewidth=1.8)
        peak_sub = sub.loc[sub["is_predicted_peak"]]
        if not peak_sub.empty:
            ax.scatter(pd.to_datetime(peak_sub["date"]), peak_sub["y_pred"], color="red", s=18, label="Predicted peak")
        ax.set_title(f"Biweek {int(biweek)}")
        ax.tick_params(axis="x", labelrotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pdr_heatmap(metrics_df: pd.DataFrame, figure_path: Path) -> None:
    pivot = metrics_df.pivot(index="series_label", columns="biweek", values="pdr").sort_index()
    fig, ax = plt.subplots(figsize=(15, 3.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("PDR per Biweek x Series")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=90)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="PDR")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_one_series(
    series_label: str,
    series_cfg: Dict[str, object],
    config: BiweeklyBiLSTMConfig,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_df = load_m5_series_with_calendar(str(series_cfg["series_id"]))
    feat_df, feature_cols = add_engineered_features(raw_df)
    feat_df = trim_to_protocol(feat_df, config)

    train_df = feat_df.iloc[: config.train_days].copy().reset_index(drop=True)
    val_df = feat_df.iloc[config.train_days : config.train_days + config.validation_days].copy().reset_index(drop=True)

    feature_scaler, target_scaler = fit_scalers(train_df, feature_cols)
    train_features = transform_features(train_df, feature_cols, feature_scaler)
    val_features = transform_features(val_df, feature_cols, feature_scaler)
    train_target = transform_target(train_df["sales"].to_numpy(dtype=float), target_scaler)

    x_all, y_all, peak_all = make_sequences(
        scaled_features=train_features,
        scaled_target=train_target,
        raw_target=train_df["sales"].to_numpy(dtype=float),
        rolling_mean_28=train_df["rolling_mean_28"].to_numpy(dtype=float),
        lookback=config.lookback,
        horizon=config.horizon,
        peak_multiplier=config.peak_multiplier,
    )
    split_idx = max(1, int(len(x_all) * 0.85))
    x_train, y_train, peak_train = x_all[:split_idx], y_all[:split_idx], peak_all[:split_idx]
    x_stop, y_stop, peak_stop = x_all[split_idx:], y_all[split_idx:], peak_all[split_idx:]

    model = BiLSTMAttentionNet(n_features=len(feature_cols), horizon=config.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(peak_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    stop_tensors = (
        torch.from_numpy(x_stop).to(device),
        torch.from_numpy(y_stop).to(device),
        torch.from_numpy(peak_stop).to(device),
    )

    best_score = math.inf if series_cfg["early_stop_metric"] == "val_loss" else -math.inf
    best_state = None
    wait = 0
    history_rows: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses = []
        for bx, by, bpeak in loader:
            bx, by, bpeak = bx.to(device), by.to(device), bpeak.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            if series_cfg["loss_mode"] == "weighted_huber":
                loss = weighted_huber_loss(pred, by, bpeak, peak_weight=config.peak_weight)
            else:
                loss = nn.functional.smooth_l1_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            sx, sy, speak = stop_tensors
            stop_pred = model(sx)
            val_loss = float(weighted_huber_loss(stop_pred, sy, speak, config.peak_weight).item())
        val_metrics_df, _ = evaluate_biweekly(model, val_df, val_features, target_scaler, config, device)
        val_pdr = float(val_metrics_df["pdr"].mean())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_pdr": val_pdr})

        if series_cfg["early_stop_metric"] == "val_pdr":
            score = val_pdr
            improved = score > best_score + config.min_delta
        else:
            score = val_loss
            improved = score < best_score - config.min_delta
        if improved:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_df, pred_df = evaluate_biweekly(model, val_df, val_features, target_scaler, config, device)
    metrics_df.insert(0, "series_label", series_label)
    metrics_df.insert(1, "series_id", str(series_cfg["series_id"]))
    metrics_df.insert(2, "model_name", str(series_cfg["model_name"]))
    pred_df.insert(0, "series_label", series_label)
    pred_df.insert(1, "series_id", str(series_cfg["series_id"]))
    pred_df.insert(2, "model_name", str(series_cfg["model_name"]))
    history_df = pd.DataFrame(history_rows)
    history_df.insert(0, "series_label", series_label)
    history_df.insert(1, "series_id", str(series_cfg["series_id"]))
    history_df.insert(2, "model_name", str(series_cfg["model_name"]))

    artifacts = {
        "config": asdict(config),
        "series_label": series_label,
        "series_config": series_cfg,
        "feature_cols": feature_cols,
        "feature_scaler_min": feature_scaler.data_min_.tolist(),
        "feature_scaler_max": feature_scaler.data_max_.tolist(),
        "target_scaler_min": target_scaler.data_min_.tolist(),
        "target_scaler_max": target_scaler.data_max_.tolist(),
    }
    model_path = out_dir / "models" / f"{series_label}_bilstm_attention.pt"
    torch.save({"model_state_dict": model.state_dict(), "artifacts": artifacts}, model_path)

    train_df.to_csv(out_dir / "data" / f"{series_label}_train_features.csv", index=False)
    val_df.to_csv(out_dir / "data" / f"{series_label}_validation_features.csv", index=False)
    history_df.to_csv(out_dir / "history" / f"{series_label}_training_history.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics" / f"{series_label}_biweekly_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "predictions" / f"{series_label}_biweekly_predictions.csv", index=False)
    (out_dir / "models" / f"{series_label}_model_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    plot_loss_curves(
        history_df,
        out_dir / "figures" / f"{series_label}_training_loss_curve.png",
        f"{series_cfg['model_name']} Training Curves",
    )
    plot_biweekly_predictions(
        pred_df,
        out_dir / "figures" / f"{series_label}_biweekly_real_vs_predicted.png",
        f"{series_cfg['model_name']}: 26 Biweekly Forecast Windows",
    )
    return metrics_df, pred_df, history_df


def aggregate_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["series_label", "series_id", "model_name"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_pdr=("pdr", "mean"),
            mean_variance_ratio=("variance_ratio", "mean"),
            pct_windows_pdr_gt_0_5=("pdr", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.5))),
            missed_peak_windows=("missed_all_peaks", "sum"),
            total_windows=("biweek", "count"),
        )
        .reset_index(drop=True)
    )


def ensure_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models", "figures", "metrics", "predictions", "history", "data"):
        (out_dir / name).mkdir(exist_ok=True)


def run_pipeline(out_dir: Path = DEFAULT_OUT_DIR, config: BiweeklyBiLSTMConfig = BiweeklyBiLSTMConfig()) -> Dict[str, str]:
    set_seed(config.seed)
    ensure_dirs(out_dir)
    metrics_frames = []
    prediction_frames = []
    history_frames = []
    for series_label, series_cfg in SERIES_CONFIG.items():
        metrics_df, pred_df, history_df = train_one_series(series_label, series_cfg, config, out_dir)
        metrics_frames.append(metrics_df)
        prediction_frames.append(pred_df)
        history_frames.append(history_df)

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_history = pd.concat(history_frames, ignore_index=True)
    summary = aggregate_summary(all_metrics)

    all_metrics.to_csv(out_dir / "biweekly_metrics_all.csv", index=False)
    all_predictions.to_csv(out_dir / "biweekly_predictions_all.csv", index=False)
    all_history.to_csv(out_dir / "training_history_all.csv", index=False)
    summary.to_csv(out_dir / "aggregate_metrics_summary.csv", index=False)
    plot_pdr_heatmap(all_metrics, out_dir / "figures" / "pdr_heatmap_biweek_by_series.png")

    readme_lines = [
        "# Biweekly BiLSTM Attention Forecast",
        "",
        "Pipeline: 28-day lookback, 14-day horizon, 26 rolling validation windows.",
        "Models: BiLSTM-HighDemand and BiLSTM-Intermittent.",
        "Architecture: BiLSTM(128, return sequences) + dropout + temporal attention + BiLSTM(64) + Dense(32 ReLU) + Dense(14).",
        "Peak definition: predicted or actual sales > rolling_mean_28d * 1.5.",
        "Intermittent model: weighted Huber loss with 3x under-predicted peak penalty and early stopping on validation PDR.",
        "",
        "Outputs:",
        "- biweekly_metrics_all.csv",
        "- biweekly_predictions_all.csv",
        "- aggregate_metrics_summary.csv",
        "- figures/*",
        "- models/*_bilstm_attention.pt",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "metrics": str(out_dir / "biweekly_metrics_all.csv"),
        "predictions": str(out_dir / "biweekly_predictions_all.csv"),
        "summary": str(out_dir / "aggregate_metrics_summary.csv"),
        "pdr_heatmap": str(out_dir / "figures" / "pdr_heatmap_biweek_by_series.png"),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run biweekly BiLSTM-attention M5 peak forecasting pipeline.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=BiweeklyBiLSTMConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BiweeklyBiLSTMConfig.batch_size)
    parser.add_argument("--patience", type=int, default=BiweeklyBiLSTMConfig.patience)
    parser.add_argument("--train-days", type=int, default=BiweeklyBiLSTMConfig.train_days)
    parser.add_argument("--seed", type=int, default=BiweeklyBiLSTMConfig.seed)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = BiweeklyBiLSTMConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        train_days=args.train_days,
    )
    outputs = run_pipeline(args.out_dir, config)
    print(json.dumps(outputs, indent=2))
    summary = pd.read_csv(outputs["summary"])
    print("\nFinal aggregate metrics:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
