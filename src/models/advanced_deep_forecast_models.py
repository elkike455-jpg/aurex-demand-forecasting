from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.deep_challenger_models import (
    DeepTrainingConfig,
    _as_float_array,
    _build_windows,
    _choose_device,
    _normal_quantile_975,
    _series_scale,
)


class _RecursivePointForecaster:
    """Shared fit / forecast loop for one-step recursive neural forecasters."""

    def __init__(self, **kwargs):
        self.config = DeepTrainingConfig(**kwargs)
        self.device = _choose_device(self.config.device)
        self.model = self._build_model().to(self.device)
        self.scale_: float = 1.0
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _loss_fn(self) -> nn.Module:
        return nn.MSELoss()

    def fit(self, y_train: np.ndarray):
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
        loss_fn = self._loss_fn()

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _LSTMNet(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class _TCNNet(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        channels = [hidden_size, hidden_size]
        blocks = []
        in_channels = 1
        for level, out_channels in enumerate(channels):
            blocks.append(
                _TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        last_hidden = out[:, :, -1]
        return self.head(last_hidden).squeeze(-1)


class _NBEATSBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size + 1),
        )
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.fc(x)
        backcast = theta[:, : self.input_size]
        forecast = theta[:, self.input_size :]
        return backcast, forecast.squeeze(-1)


class _NBEATSNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_NBEATSBlock(input_size, hidden_size) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.squeeze(-1)
        forecast = torch.zeros(residual.size(0), device=residual.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class _InformerStyleNet(nn.Module):
    """Lightweight Informer-inspired encoder for long-context forecasting."""

    def __init__(self, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        nhead = 4 if d_model >= 32 else 2
        self.input_proj = nn.Linear(1, d_model)
        self.positional = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional(h)
        h = self.encoder(h)
        h = self.distill(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1, :]).squeeze(-1)


class _GRN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        gate = torch.sigmoid(self.gate(residual))
        return self.norm(gate * x + (1.0 - gate) * residual)


class _TFTStyleNet(nn.Module):
    """Lightweight TFT-style forecaster for single-series comparisons."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.input_proj = nn.Linear(1, hidden_size)
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4 if hidden_size >= 32 else 2, dropout=dropout, batch_first=True)
        self.post_grn = _GRN(hidden_size, hidden_size, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h * self.selector(h)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, need_weights=False)
        fused = self.post_grn(attn_out[:, -1, :])
        return self.head(fused).squeeze(-1)


class _LSTMTransformerNet(nn.Module):
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
        self.positional = _PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4 if hidden_size >= 32 else 2,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.positional(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class LSTMForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _LSTMNet(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )


class TCNForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _TCNNet(
            hidden_size=self.config.hidden_size,
            dropout=max(self.config.dropout, 0.05),
        )


class NBEATSForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _NBEATSNet(
            input_size=self.config.context_length,
            hidden_size=max(self.config.hidden_size, 64),
            num_blocks=max(3, self.config.num_layers + 2),
        )


class InformerForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _InformerStyleNet(
            d_model=max(self.config.hidden_size, 32),
            num_layers=max(1, self.config.num_layers),
            dropout=max(self.config.dropout, 0.05),
        )


class TFTForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _TFTStyleNet(
            hidden_size=max(self.config.hidden_size, 32),
            num_layers=max(1, self.config.num_layers),
            dropout=max(self.config.dropout, 0.05),
        )


class LSTMTransformerForecastModel(_RecursivePointForecaster):
    def _build_model(self) -> nn.Module:
        return _LSTMTransformerNet(
            hidden_size=max(self.config.hidden_size, 32),
            num_layers=max(1, self.config.num_layers),
            dropout=max(self.config.dropout, 0.05),
        )


__all__ = [
    "InformerForecastModel",
    "LSTMForecastModel",
    "LSTMTransformerForecastModel",
    "NBEATSForecastModel",
    "TCNForecastModel",
    "TFTForecastModel",
]
