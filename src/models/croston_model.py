from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def aggregate_weekly(
    dates: pd.Series,
    sales: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Aggregate a daily demand series to weekly totals.

    This is useful for extremely sparse daily series where intermittent-demand
    models benefit from a less noisy signal.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "sales": sales})
    df = df.set_index("date").resample("W")["sales"].sum().reset_index()
    return df["date"], df["sales"].values


class CrostonSBAModel:
    """
    Croston model with Syntetos-Boylan Approximation bias correction.

    The model updates:
    - demand size on non-zero periods
    - inter-demand interval on non-zero periods

    The SBA correction multiplies the Croston forecast by (1 - alpha / 2).
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        use_weekly: bool = False,
    ):
        self.alpha = alpha
        self.use_weekly = use_weekly

        self._fitted_alpha: Optional[float] = None
        self._level: float = 0.0
        self._interval: float = 1.0
        self._is_weekly: bool = False
        self._weekly_scale: int = 7

    @staticmethod
    def _nonzero_positions(y: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.asarray(y, dtype=float) > 0)

    @classmethod
    def _fit_path(
        cls,
        y: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute one Croston-SBA pass and return fitted path plus terminal state.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        fitted = np.zeros(n, dtype=float)

        nz_pos = cls._nonzero_positions(y)
        if len(nz_pos) == 0:
            return fitted, 0.0, 1.0

        first_idx = int(nz_pos[0])
        level = float(y[first_idx])
        interval = float(first_idx + 1)
        last_demand_idx = first_idx

        point = (1.0 - alpha / 2.0) * (level / max(interval, 1e-9))
        fitted[: first_idx + 1] = point

        for t in range(first_idx + 1, n):
            fitted[t] = point
            if y[t] > 0:
                observed_interval = float(t - last_demand_idx)
                level = alpha * float(y[t]) + (1.0 - alpha) * level
                interval = alpha * observed_interval + (1.0 - alpha) * interval
                point = (1.0 - alpha / 2.0) * (level / max(interval, 1e-9))
                last_demand_idx = t

        return fitted, level, interval

    @classmethod
    def _optimize_alpha(cls, y: np.ndarray) -> float:
        """Select alpha by in-sample MAE over a small, stable grid."""
        best_alpha = 0.1
        best_mae = np.inf

        for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            fitted, _, _ = cls._fit_path(y, alpha)
            mae = float(np.mean(np.abs(np.asarray(y, dtype=float) - fitted)))
            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        return best_alpha

    def fit(self, y_train: np.ndarray) -> "CrostonSBAModel":
        y_train = np.asarray(y_train, dtype=float)
        y_fit = y_train

        self._is_weekly = False
        if self.use_weekly and len(y_train) >= 14:
            dates = pd.date_range(start="2020-01-01", periods=len(y_train), freq="D")
            _, y_weekly = aggregate_weekly(dates, y_train)
            y_fit = y_weekly
            self._is_weekly = True

        alpha = float(self.alpha) if self.alpha is not None else self._optimize_alpha(y_fit)
        self._fitted_alpha = alpha

        _, self._level, self._interval = self._fit_path(y_fit, alpha)
        return self

    def forecast(self, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fitted_alpha is None:
            raise RuntimeError("CrostonSBAModel must be fitted before forecasting.")

        point = (1.0 - self._fitted_alpha / 2.0) * (self._level / max(self._interval, 1e-9))
        y_pred = np.full(int(h), point, dtype=float)

        if self._is_weekly:
            n_weeks = max(1, int(np.ceil(h / self._weekly_scale)))
            y_pred_weekly = np.full(n_weeks, point / self._weekly_scale, dtype=float)
            y_pred = np.repeat(y_pred_weekly, self._weekly_scale)[:h]

        y_pred = np.maximum(y_pred, 0.0)
        spread = float(np.mean(np.abs(y_pred))) * 0.5
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up

    @property
    def params(self) -> dict:
        return {
            "alpha": self._fitted_alpha,
            "is_weekly": self._is_weekly,
        }
