from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SpdinRepoLSTMConfig:
    context_length: int = 28
    num_classes: int = 1
    input_size: int = 1
    hidden_size: int = 2
    num_layers: int = 1
    epochs: int = 2000
    lr: float = 0.01
    seed: int = 42


def _as_float_array(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def sliding_windows(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for i in range(len(data) - seq_length):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


class RepoLSTM(nn.Module):
    """
    Close adaptation of the original class from
    `spdin/time-series-prediction-lstm-pytorch`.
    """

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


class SpdinRepoLSTMForecastModel:
    def __init__(self, **kwargs):
        self.config = SpdinRepoLSTMConfig(**kwargs)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.scaler_ = MinMaxScaler()
        self.model = RepoLSTM(
            self.config.num_classes,
            self.config.input_size,
            self.config.hidden_size,
            self.config.num_layers,
        )
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def fit(self, y_fit: np.ndarray, scaling_reference: np.ndarray | None = None) -> "SpdinRepoLSTMForecastModel":
        y_fit = _as_float_array(y_fit).reshape(-1)
        scale_ref = y_fit if scaling_reference is None else _as_float_array(scaling_reference).reshape(-1)
        self.scaler_.fit(scale_ref.reshape(-1, 1))

        fit_scaled = self.scaler_.transform(y_fit.reshape(-1, 1))
        x, y = sliding_windows(fit_scaled, self.config.context_length)
        trainX = torch.tensor(np.array(x), dtype=torch.float32)
        trainY = torch.tensor(np.array(y), dtype=torch.float32)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.model.train()
        final_loss = None
        for _ in range(self.config.epochs):
            outputs = self.model(trainX)
            optimizer.zero_grad()
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        self.model.eval()
        with torch.no_grad():
            fitted_scaled = self.model(trainX).detach().cpu().numpy()
        fitted = self.scaler_.inverse_transform(fitted_scaled).reshape(-1)
        resid = y_fit[self.config.context_length :] - fitted
        self.train_resid_std_ = float(np.std(resid)) if len(resid) else 0.0
        self.training_history_ = {"final_loss": final_loss if final_loss is not None else float("nan")}
        return self

    def forecast(self, h: int, y_history: np.ndarray):
        history = _as_float_array(y_history).reshape(-1)
        history_scaled = self.scaler_.transform(history.reshape(-1, 1)).reshape(-1)
        context = history_scaled[-self.config.context_length :]
        if len(context) < self.config.context_length:
            context = np.pad(context, (self.config.context_length - len(context), 0))

        preds_scaled = np.zeros(int(h), dtype=np.float32)
        work = context.astype(np.float32).copy()

        self.model.eval()
        with torch.no_grad():
            for step in range(int(h)):
                x = torch.tensor(work.reshape(1, self.config.context_length, 1), dtype=torch.float32)
                next_pred = float(self.model(x).detach().cpu().numpy().reshape(-1)[0])
                preds_scaled[step] = next_pred
                work = np.roll(work, -1)
                work[-1] = next_pred

        y_pred = self.scaler_.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        y_pred = np.maximum(y_pred, 0.0)
        spread = 1.959963984540054 * max(self.train_resid_std_, 1e-6)
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up
