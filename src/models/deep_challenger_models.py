from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _choose_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_float_array(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _series_scale(y: np.ndarray) -> float:
    y = _as_float_array(y)
    scale = float(np.mean(np.abs(y[y != 0]))) if np.any(y != 0) else 1.0
    return max(scale, 1.0)


def _build_windows(y: np.ndarray, context_length: int) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_float_array(y)
    if len(y) <= context_length:
        raise ValueError(
            f"Series length {len(y)} must be greater than context_length {context_length}."
        )

    X, target = [], []
    for end in range(context_length, len(y)):
        X.append(y[end - context_length : end])
        target.append(y[end])
    return np.asarray(X, dtype=np.float32), np.asarray(target, dtype=np.float32)


def _normal_quantile_975() -> float:
    return 1.959963984540054


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.mean(200.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def train_test_split_series(y: np.ndarray, test_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_float_array(y)
    if len(y) <= test_days + 30:
        test_days = max(30, int(len(y) * 0.2))
    return y[:-test_days], y[-test_days:]


class _GRUNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class _DeepARNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        mu = self.mu_head(last_hidden).squeeze(-1)
        sigma = self.softplus(self.sigma_head(last_hidden)).squeeze(-1) + 1e-4
        return mu, sigma


@dataclass
class DeepTrainingConfig:
    context_length: int = 28
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str | None = None
    seed: int = 42


class GRUForecastModel:
    """Simple recursive GRU forecaster for challenger-model benchmarking."""

    def __init__(self, **kwargs):
        self.config = DeepTrainingConfig(**kwargs)
        self.device = _choose_device(self.config.device)
        self.model = _GRUNet(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.scale_: float = 1.0
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def fit(self, y_train: np.ndarray) -> "GRUForecastModel":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        y = _as_float_array(y_train)
        self.scale_ = _series_scale(y)
        y_scaled = (y / self.scale_).astype(np.float32)

        X, target = _build_windows(y_scaled, self.config.context_length)
        ds = TensorDataset(
            torch.from_numpy(X).unsqueeze(-1),
            torch.from_numpy(target),
        )
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.model.train()
        final_loss = None
        for _ in range(self.config.epochs):
            epoch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            final_loss = float(np.mean(epoch_losses)) if epoch_losses else None

        fitted = self._predict_in_sample_scaled(X) * self.scale_
        resid = y[self.config.context_length :] - fitted
        self.train_resid_std_ = float(np.std(resid)) if len(resid) else 0.0
        self.training_history_ = {"final_loss": final_loss if final_loss is not None else float("nan")}
        return self

    def _predict_in_sample_scaled(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).unsqueeze(-1).to(self.device)
            pred = self.model(tensor_x).detach().cpu().numpy()
        return pred.astype(float)

    def forecast(self, h: int, y_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = (_as_float_array(y_history) / self.scale_).astype(np.float32)
        context = history[-self.config.context_length :]
        if len(context) < self.config.context_length:
            context = np.pad(context, (self.config.context_length - len(context), 0))

        preds_scaled = np.zeros(int(h), dtype=np.float32)
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device).view(
            1, self.config.context_length, 1
        )

        self.model.eval()
        with torch.inference_mode():
            for step in range(int(h)):
                next_pred = float(self.model(context_tensor).item())
                preds_scaled[step] = next_pred
                context_tensor = torch.roll(context_tensor, shifts=-1, dims=1)
                context_tensor[0, -1, 0] = next_pred

        y_pred = np.maximum(preds_scaled.astype(float) * self.scale_, 0.0)
        spread = _normal_quantile_975() * max(self.train_resid_std_, 1e-6)
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up


class DeepARForecastModel:
    """Minimal Gaussian DeepAR-style challenger using autoregressive LSTM likelihood."""

    def __init__(self, **kwargs):
        self.config = DeepTrainingConfig(**kwargs)
        self.device = _choose_device(self.config.device)
        self.model = _DeepARNet(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.scale_: float = 1.0
        self.avg_sigma_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def fit(self, y_train: np.ndarray) -> "DeepARForecastModel":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        y = _as_float_array(y_train)
        self.scale_ = _series_scale(y)
        y_scaled = (y / self.scale_).astype(np.float32)

        X, target = _build_windows(y_scaled, self.config.context_length)
        ds = TensorDataset(
            torch.from_numpy(X).unsqueeze(-1),
            torch.from_numpy(target),
        )
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.model.train()
        final_loss = None
        sigma_values = []
        for _ in range(self.config.epochs):
            epoch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                mu, sigma = self.model(batch_x)
                dist = torch.distributions.Normal(mu, sigma)
                loss = -dist.log_prob(batch_y).mean()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                sigma_values.append(float(sigma.mean().item()))
            final_loss = float(np.mean(epoch_losses)) if epoch_losses else None

        self.avg_sigma_ = float(np.mean(sigma_values)) * self.scale_ if sigma_values else 0.0
        self.training_history_ = {"final_loss": final_loss if final_loss is not None else float("nan")}
        return self

    def forecast(self, h: int, y_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = (_as_float_array(y_history) / self.scale_).astype(np.float32)
        context = history[-self.config.context_length :]
        if len(context) < self.config.context_length:
            context = np.pad(context, (self.config.context_length - len(context), 0))

        preds_scaled = np.zeros(int(h), dtype=np.float32)
        sigma_scaled = np.zeros(int(h), dtype=np.float32)
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device).view(
            1, self.config.context_length, 1
        )

        self.model.eval()
        with torch.inference_mode():
            for step in range(int(h)):
                mu, sigma = self.model(context_tensor)
                next_mu = float(mu.item())
                next_sigma = float(sigma.item())
                preds_scaled[step] = next_mu
                sigma_scaled[step] = next_sigma
                context_tensor = torch.roll(context_tensor, shifts=-1, dims=1)
                context_tensor[0, -1, 0] = next_mu

        y_pred = np.maximum(preds_scaled.astype(float) * self.scale_, 0.0)
        sigma = sigma_scaled.astype(float) * self.scale_
        conf_low = np.maximum(y_pred - _normal_quantile_975() * sigma, 0.0)
        conf_up = y_pred + _normal_quantile_975() * sigma
        return y_pred, conf_low, conf_up
