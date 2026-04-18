from __future__ import annotations

import numpy as np

from src.detection.change_point_detector import BayesianOnlineChangepointDetector
from src.models.conformal_wrapper import ConformalForecaster
from src.models.regime_forecast_engine import RegimeForecastEngine


def test_bocpd_detects_synthetic_changepoint() -> None:
    detector = BayesianOnlineChangepointDetector()
    y = np.r_[np.full(90, 5.0), np.zeros(120)]
    result = detector.detect(y)

    assert result["change_points"], "Expected at least one detected change point."
    assert any(abs(cp - 90) <= 35 for cp in result["change_points"])


def test_trigger_reroute_returns_true_on_flip() -> None:
    y = np.zeros(30, dtype=float)
    assert BayesianOnlineChangepointDetector.trigger_reroute(y, original_zero_rate=0.10)


def test_conformal_coverage_guarantee() -> None:
    rng = np.random.default_rng(42)
    y_true = np.maximum(rng.normal(12.0, 2.5, size=200), 0.0)
    y_pred = np.maximum(y_true + rng.normal(0.0, 1.2, size=200), 0.0)

    wrapper = ConformalForecaster(base_model=None, coverage=0.90, regime="stable")
    wrapper.calibrate(y_true, y_pred)

    assert wrapper.coverage_achieved is not None
    assert wrapper.coverage_achieved >= 0.85


def test_blend_at_boundary_equals_pure_model() -> None:
    y_s = np.array([1.0, 2.0, 3.0], dtype=float)
    y_h = np.array([4.0, 5.0, 6.0], dtype=float)

    blend_low = RegimeForecastEngine.blend_forecast(y_s, y_h, zero_rate=0.30)
    blend_high = RegimeForecastEngine.blend_forecast(y_s, y_h, zero_rate=0.70)

    assert np.allclose(blend_low, y_s)
    assert np.allclose(blend_high, y_h)


def test_reorder_point_positive() -> None:
    wrapper = ConformalForecaster(base_model=None, coverage=0.90, regime="intermittent")
    y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)
    y_pred = np.array([0.1, 0.9, 0.2, 1.7, 0.1], dtype=float)
    wrapper.calibrate(y_true, y_pred)

    reorder_point = wrapper.reorder_point(np.array([0.2, 0.4, 0.1, 0.3], dtype=float), lead_time_days=3)
    assert reorder_point > 0.0
