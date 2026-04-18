from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class ZhangxuRepoGRUConfig:
    context_length: int = 28
    hidden_num: int = 64
    layer_num: int = 1
    lr: float = 1e-4
    epochs: int = 80
    batch_size: int = 32
    seed: int = 42
    use_cuda: bool = False


def _as_float_array(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def create_multi_ahead_samples(ts: np.ndarray, look_back: int, look_ahead: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Close adaptation of `src/util.py:create_multi_ahead_samples` from
    `zhangxu0307/time_series_forecasting_pytorch`.
    """
    ts = np.asarray(ts, dtype=np.float32).reshape(-1, 1)
    data_x, data_y = [], []
    for i in range(len(ts) - look_back - look_ahead + 1):
        history_seq = ts[i : i + look_back]
        future_seq = ts[i + look_back : i + look_back + look_ahead]
        data_x.append(history_seq)
        data_y.append(future_seq)
    return np.asarray(data_x, dtype=np.float32), np.asarray(data_y, dtype=np.float32)


class TimeSeriesDataset(Dataset):
    """
    Close adaptation of `src/ts_loader.py:Time_Series_Data`.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray):
        self.x = train_x
        self.y = train_y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class GRUModel(nn.Module):
    """
    Close adaptation of `src/model.py:GRUModel`.
    """

    def __init__(self, input_dim: int, hidden_num: int, output_dim: int, layer_num: int):
        super().__init__()
        self.hidden_num = hidden_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.cell = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_num,
            num_layers=self.layer_num,
            dropout=0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_num, self.output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_num, batch_size, self.hidden_num, dtype=x.dtype, device=x.device)
        _, hn = self.cell(x, h0)
        hn = hn.view(batch_size, self.hidden_num)
        return self.fc(hn)


def train_repo_style(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: ZhangxuRepoGRUConfig,
) -> Tuple[GRUModel, Dict[str, float]]:
    """
    Close adaptation of `src/NN_train.py:train`.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = TimeSeriesDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    model = GRUModel(input_dim=1, hidden_num=config.hidden_num, output_dim=1, layer_num=config.layer_num).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=config.lr, momentum=0.9)
    criterion = nn.MSELoss()

    final_loss = float("nan")
    epoch_losses = []
    model.train()
    for _ in range(config.epochs):
        batch_losses = []
        for batch_x, batch_y in dataloader:
            x = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            y = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        if batch_losses:
            final_loss = float(np.mean(batch_losses))
            epoch_losses.append(final_loss)

    return model, {"final_loss": final_loss, "last_epoch_loss": final_loss, "epochs": float(config.epochs)}


def predict_iteration_repo_style(
    model: GRUModel,
    test_x: np.ndarray,
    look_ahead: int,
    use_cuda: bool = False,
) -> np.ndarray:
    """
    Close adaptation of `src/NN_train.py:predict_iteration`.
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_x = np.asarray(test_x, dtype=np.float32)
    test_batch_size = test_x.shape[0]
    ans = []

    with torch.no_grad():
        for _ in range(int(look_ahead)):
            test_x_torch = torch.as_tensor(test_x, dtype=torch.float32, device=device)
            pred = model(test_x_torch).detach().cpu().numpy().reshape(test_batch_size)
            ans.append(pred)

            test_x = test_x[:, 1:]
            pred = pred.reshape((test_batch_size, 1, 1))
            test_x = np.append(test_x, pred, axis=1)

    ans = np.asarray(ans)
    return ans.transpose([1, 0])


class ZhangxuRepoGRUForecastModel:
    def __init__(self, **kwargs):
        self.config = ZhangxuRepoGRUConfig(**kwargs)
        self.scaler_ = MinMaxScaler(feature_range=(0, 1))
        self.model_: GRUModel | None = None
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def fit(self, y_fit: np.ndarray, scaling_reference: np.ndarray | None = None) -> "ZhangxuRepoGRUForecastModel":
        y_fit = _as_float_array(y_fit)
        scale_ref = y_fit if scaling_reference is None else _as_float_array(scaling_reference)
        self.scaler_.fit(scale_ref.reshape(-1, 1))

        fit_scaled = self.scaler_.transform(y_fit.reshape(-1, 1)).reshape(-1)
        train_x, train_y = create_multi_ahead_samples(
            fit_scaled,
            look_back=self.config.context_length,
            look_ahead=1,
        )
        train_y = np.squeeze(train_y, axis=1)

        self.model_, self.training_history_ = train_repo_style(train_x, train_y, self.config)

        with torch.no_grad():
            device = torch.device(
                "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
            )
            x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
            fitted_scaled = self.model_.to(device)(x_tensor).detach().cpu().numpy().reshape(-1, 1)
        fitted = self.scaler_.inverse_transform(fitted_scaled).reshape(-1)
        target = y_fit[self.config.context_length :]
        resid = target - fitted[: len(target)]
        self.train_resid_std_ = float(np.std(resid)) if len(resid) else 0.0
        return self

    def forecast(self, h: int, y_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Call fit() before forecast().")

        history = _as_float_array(y_history)
        history_scaled = self.scaler_.transform(history.reshape(-1, 1)).reshape(-1)
        context = history_scaled[-self.config.context_length :]
        if len(context) < self.config.context_length:
            context = np.pad(context, (self.config.context_length - len(context), 0))

        test_x = context.reshape(1, self.config.context_length, 1).astype(np.float32)
        preds_scaled = predict_iteration_repo_style(
            self.model_,
            test_x,
            look_ahead=int(h),
            use_cuda=self.config.use_cuda,
        ).reshape(-1)

        y_pred = self.scaler_.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        y_pred = np.maximum(y_pred, 0.0)
        spread = 1.959963984540054 * max(self.train_resid_std_, 1e-6)
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up
