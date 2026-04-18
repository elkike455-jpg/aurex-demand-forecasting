from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class ConformalForecaster:
    r"""
    Add split-conformal prediction intervals around a point forecaster.

    Given calibration targets ``y`` and point predictions ``\hat y``, the
    nonconformity scores are:

    .. math::
        s_i = |y_i - \hat y_i|

    The conformal radius is the empirical upper quantile:

    .. math::
        \hat q = Q_{1-\alpha}(s_1, \ldots, s_n)

    where ``coverage = 1 - alpha``.
    """

    def __init__(self, base_model: object, coverage: float = 0.90, regime: str = "stable") -> None:
        self.base_model = base_model
        self.coverage = float(coverage)
        self.regime = str(regime)
        self.alpha = 1.0 - self.coverage
        self.q_hat: float | None = None
        self.scores: np.ndarray | None = None
        self.coverage_achieved: float | None = None

    def calibrate(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> "ConformalForecaster":
        r"""
        Fit the split-conformal residual quantile.

        Parameters
        ----------
        y_calib : np.ndarray
            Observed calibration targets.
        y_pred_calib : np.ndarray
            Point predictions on the calibration window.

        Returns
        -------
        ConformalForecaster
            The fitted wrapper.

        Example
        -------
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> p = np.array([1.1, 1.9, 2.8])
        >>> cf = ConformalForecaster(base_model=None).calibrate(y, p)
        >>> cf.q_hat >= 0.0
        True
        """
        y_true = np.asarray(y_calib, dtype=float)
        y_pred = np.asarray(y_pred_calib, dtype=float)
        if len(y_true) != len(y_pred):
            raise ValueError("Calibration targets and predictions must have the same length.")
        if len(y_true) == 0:
            raise ValueError("Calibration data cannot be empty.")

        scores = np.abs(y_true - y_pred)
        level = np.ceil((len(scores) + 1) * (1.0 - self.alpha)) / len(scores)
        level = float(np.clip(level, 0.0, 1.0))
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.scores = scores
        self.coverage_achieved = float(np.mean(scores <= self.q_hat))
        LOGGER.info(
            "Conformal calibration complete for regime=%s with q_hat=%.4f and empirical coverage=%.3f",
            self.regime,
            self.q_hat,
            self.coverage_achieved,
        )
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build conformal prediction intervals around point predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Point forecasts.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper interval bounds.
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalForecaster must be calibrated before predicting intervals.")

        point = np.asarray(y_pred, dtype=float)
        if self.regime == "intermittent":
            lower = np.maximum(point - self.q_hat * 0.3, 0.0)
            upper = point + self.q_hat * 1.7
        else:
            lower = np.maximum(point - self.q_hat, 0.0)
            upper = point + self.q_hat
        return lower, upper

    def safety_stock(self, service_level: float = 0.95) -> float:
        """
        Convert the conformal radius into a safety-stock quantity.

        Parameters
        ----------
        service_level : float, default=0.95
            Requested service level. Supported: 0.90, 0.95, 0.99.

        Returns
        -------
        float
            Safety stock quantity.
        """
        if self.q_hat is None:
            raise RuntimeError("ConformalForecaster must be calibrated before computing safety stock.")

        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        if service_level not in z_scores:
            raise ValueError("service_level must be one of 0.90, 0.95, or 0.99.")
        return float(max(self.q_hat, 0.0) * z_scores[service_level])

    def reorder_point(self, y_pred: np.ndarray, lead_time_days: int) -> float:
        """
        Combine lead-time demand with safety stock.

        Parameters
        ----------
        y_pred : np.ndarray
            Point forecast horizon.
        lead_time_days : int
            Number of days to cover.

        Returns
        -------
        float
            Reorder point for inventory planning.
        """
        horizon = np.asarray(y_pred, dtype=float)
        lead_time = max(int(lead_time_days), 1)
        lead_demand = float(np.sum(np.maximum(horizon[:lead_time], 0.0)))
        return float(max(lead_demand + self.safety_stock(), 0.0))

    def coverage_report(self) -> Dict[str, float | int | str]:
        """Return calibration diagnostics for reporting and notebooks."""
        if self.q_hat is None or self.coverage_achieved is None or self.scores is None:
            raise RuntimeError("ConformalForecaster must be calibrated before requesting a report.")

        return {
            "q_hat": float(self.q_hat),
            "coverage_achieved": float(self.coverage_achieved),
            "n_calibration_samples": int(len(self.scores)),
            "regime": self.regime,
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = np.maximum(rng.normal(10.0, 2.0, size=120), 0.0)
    y_pred = np.maximum(y_true + rng.normal(0.0, 1.0, size=120), 0.0)

    wrapper = ConformalForecaster(base_model=None, coverage=0.90, regime="stable")
    wrapper.calibrate(y_true, y_pred)
    lower, upper = wrapper.predict_interval(y_pred[:10])

    print("Calibration:", "PASS" if wrapper.coverage_report()["coverage_achieved"] >= 0.85 else "FAIL")
    print("Interval shape:", "PASS" if np.all(lower <= upper) else "FAIL")
