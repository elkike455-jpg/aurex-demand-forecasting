from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class WalmartRepoBiLSTMConfig:
    context_length: int = 28
    units: int = 50
    dropout: float = 0.2
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    verbose: int = 0


def _as_float_array(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _build_windows(y_scaled: np.ndarray, context_length: int) -> Tuple[np.ndarray, np.ndarray]:
    y_scaled = _as_float_array(y_scaled).reshape(-1)
    if len(y_scaled) <= context_length:
        raise ValueError(
            f"Series length {len(y_scaled)} must be greater than context_length {context_length}."
        )

    X, target = [], []
    for end in range(context_length, len(y_scaled)):
        X.append(y_scaled[end - context_length : end])
        target.append(y_scaled[end])
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, context_length, 1)
    y_arr = np.asarray(target, dtype=np.float32)
    return X_arr, y_arr


class WalmartRepoBiLSTMForecastModel:
    """
    Minimal adaptation of the external repository `egemenozen1/Walmart-LSTM-Sales-Forecasting`.

    Preserved from the external repo:
      - Keras Sequential API
      - 2-layer Bidirectional LSTM
      - Dropout after each recurrent block
      - MSE loss with Adam optimizer

    Adapted for this project:
      - single-series recursive forecasting
      - chronological windows instead of random tabular train/test split
      - scaler can be fit on a separate reference segment to respect benchmark protocol
    """

    def __init__(self, **kwargs):
        self.config = WalmartRepoBiLSTMConfig(**kwargs)
        self.scaler_ = MinMaxScaler()
        self.model = self._build_model()
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(self.config.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(units=self.config.units, return_sequences=True),
                input_shape=(self.config.context_length, 1),
            )
        )
        model.add(Dropout(self.config.dropout))
        model.add(Bidirectional(LSTM(units=self.config.units, return_sequences=False)))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=self.config.lr), loss="mean_squared_error")
        return model

    def fit(self, y_fit: np.ndarray, scaling_reference: np.ndarray | None = None) -> "WalmartRepoBiLSTMForecastModel":
        y_fit = _as_float_array(y_fit).reshape(-1)
        scale_ref = y_fit if scaling_reference is None else _as_float_array(scaling_reference).reshape(-1)
        self.scaler_.fit(scale_ref.reshape(-1, 1))

        fit_scaled = self.scaler_.transform(y_fit.reshape(-1, 1)).reshape(-1)
        X_fit, y_target = _build_windows(fit_scaled, self.config.context_length)

        history = self.model.fit(
            X_fit,
            y_target,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            shuffle=False,
        )

        fitted_scaled = self.model.predict(X_fit, verbose=0).reshape(-1, 1)
        fitted = self.scaler_.inverse_transform(fitted_scaled).reshape(-1)
        resid = y_fit[self.config.context_length :] - fitted
        self.train_resid_std_ = float(np.std(resid)) if len(resid) else 0.0

        losses = history.history.get("loss", [])
        self.training_history_ = {
            "final_loss": float(losses[-1]) if losses else float("nan"),
        }
        return self

    def forecast(self, h: int, y_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = _as_float_array(y_history).reshape(-1)
        history_scaled = self.scaler_.transform(history.reshape(-1, 1)).reshape(-1)
        context = history_scaled[-self.config.context_length :]
        if len(context) < self.config.context_length:
            context = np.pad(context, (self.config.context_length - len(context), 0))

        preds_scaled = np.zeros(int(h), dtype=np.float32)
        work = context.astype(np.float32).copy()

        for step in range(int(h)):
            x = work.reshape(1, self.config.context_length, 1)
            next_pred = float(self.model.predict(x, verbose=0).reshape(-1)[0])
            preds_scaled[step] = next_pred
            work = np.roll(work, -1)
            work[-1] = next_pred

        y_pred = self.scaler_.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        y_pred = np.maximum(y_pred, 0.0)
        spread = 1.959963984540054 * max(self.train_resid_std_, 1e-6)
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up
