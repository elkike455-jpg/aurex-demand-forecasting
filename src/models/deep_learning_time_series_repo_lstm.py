from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DeepLearningTimeSeriesRepoLSTMConfig:
    context_length: int = 28
    units_first: int = 50
    units_second: int = 256
    dropout: float = 0.5
    epochs: int = 10
    batch_size: int = 70
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


class DeepLearningTimeSeriesRepoLSTMForecastModel:
    """
    Minimal adaptation of `deep-learning-time-series-master`.

    Reused from the repo notebooks:
      - Keras Sequential API
      - stacked LSTM architecture
      - dropout after each LSTM block
      - linear output head
      - MSE loss with Adam

    Adapted for this project:
      - chronological split instead of 80/20 split
      - context_length=28 instead of sine-wave window=50
      - recursive multi-step forecasting on benchmark products
    """

    def __init__(self, **kwargs):
        self.config = DeepLearningTimeSeriesRepoLSTMConfig(**kwargs)
        self.scaler_ = MinMaxScaler()
        self.model = self._build_model()
        self.train_resid_std_: float = 0.0
        self.training_history_: Dict[str, float] = {}

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(self.config.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        model = Sequential()
        model.add(
            LSTM(
                units=self.config.units_first,
                input_shape=(self.config.context_length, 1),
                return_sequences=True,
            )
        )
        model.add(Dropout(self.config.dropout))
        model.add(LSTM(self.config.units_second))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.config.lr))
        return model

    def fit(self, y_fit: np.ndarray, scaling_reference: np.ndarray | None = None) -> "DeepLearningTimeSeriesRepoLSTMForecastModel":
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
            validation_split=0.0,
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

    def forecast(self, h: int, y_history: np.ndarray):
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
