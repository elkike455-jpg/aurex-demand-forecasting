from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.special import gammaln

LOGGER = logging.getLogger(__name__)


@dataclass
class _PosteriorState:
    """Normal-Inverse-Gamma posterior parameters for one run-length state."""

    mu: float
    kappa: float
    alpha: float
    beta: float


class BayesianOnlineChangepointDetector:
    r"""
    Bayesian Online Change-Point Detection on rolling zero-rate.

    The detector runs BOCPD on a derived signal:

    .. math::
        z_t = \frac{1}{W}\sum_{i=t-W+1}^{t}\mathbf{1}[y_i = 0]

    where :math:`W=30` by default. Each run-length hypothesis uses a
    Normal-Inverse-Gamma prior and a Student-t prior predictive density:

    .. math::
        p(z_t \mid r_{t-1}) = \text{St}\left(
            z_t;\ \mu_r,\ \sqrt{\frac{\beta_r(\kappa_r + 1)}{\alpha_r\kappa_r}},
            \nu=2\alpha_r
        \right)

    A change point is flagged when:

    .. math::
        P(R_t \le 5 \mid z_{1:t}) > 0.5

    Parameters
    ----------
    mu0 : float, default=0.5
        Prior mean of the rolling zero-rate.
    kappa0 : float, default=1.0
        Prior precision.
    alpha0 : float, default=1.0
        Prior shape.
    beta0 : float, default=1.0
        Prior rate.
    hazard_lambda : float, default=100.0
        Expected regime length in days. Hazard is ``1 / hazard_lambda``.

    Example
    -------
    >>> y = np.r_[np.full(90, 4.0), np.zeros(120)]
    >>> detector = BayesianOnlineChangepointDetector()
    >>> out = detector.detect(y)
    >>> bool(out["regime_changed"])
    True
    """

    def __init__(
        self,
        mu0: float = 0.5,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        hazard_lambda: float = 100.0,
    ) -> None:
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.hazard_lambda = float(hazard_lambda)
        self.hazard = 1.0 / max(self.hazard_lambda, 1.0)

    @staticmethod
    def rolling_zero_rate(y: np.ndarray, window: int = 30) -> np.ndarray:
        r"""
        Compute the rolling fraction of zero-demand observations.

        The signal is:

        .. math::
            z_t = \frac{1}{n_t}\sum_{i=t-window+1}^{t}\mathbf{1}[y_i = 0]

        where ``n_t`` is the number of available observations in the window.
        If ``n_t < 10`` the value is left as ``nan`` to enforce
        ``min_periods=10``.

        Parameters
        ----------
        y : np.ndarray
            Daily demand series.
        window : int, default=30
            Rolling window length.

        Returns
        -------
        np.ndarray
            Rolling zero-rate with length ``len(y)``.

        Example
        -------
        >>> y = np.array([1, 0, 0, 2, 0], dtype=float)
        >>> z = BayesianOnlineChangepointDetector.rolling_zero_rate(y, window=3)
        >>> bool(np.isnan(z[0]))
        True
        """
        values = np.asarray(y, dtype=float)
        n = len(values)
        zero_indicator = (np.nan_to_num(values, nan=0.0) == 0).astype(float)
        out = np.full(n, np.nan, dtype=float)

        if n == 0:
            return out

        csum = np.concatenate(([0.0], np.cumsum(zero_indicator)))
        for idx in range(n):
            start = max(0, idx - window + 1)
            count = idx - start + 1
            if count < 10:
                continue
            zeros = csum[idx + 1] - csum[start]
            out[idx] = zeros / count
        return out

    def detect(self, y: np.ndarray) -> Dict[str, object]:
        r"""
        Detect structural changes in rolling zero-rate using BOCPD.

        For each time step the algorithm updates:

        .. math::
            P(R_t = 0 \mid x_{1:t}) \propto \sum_r P(R_{t-1}=r)\,p(x_t \mid r)\,H

        .. math::
            P(R_t = r+1 \mid x_{1:t}) \propto P(R_{t-1}=r)\,p(x_t \mid r)\,(1-H)

        where ``H = 1 / hazard_lambda`` is the constant hazard and the
        predictive density comes from the Normal-Inverse-Gamma posterior.

        Parameters
        ----------
        y : np.ndarray
            Daily demand series.

        Returns
        -------
        dict
            Change-point metadata, run-length path, and reroute advice.

        Example
        -------
        >>> y = np.r_[np.full(80, 3.0), np.zeros(120)]
        >>> out = BayesianOnlineChangepointDetector().detect(y)
        >>> out["recommended_action"] in {"REROUTE_TO_HURDLE", "MONITOR", "NO_ACTION"}
        True
        """
        y_arr = np.asarray(y, dtype=float)
        signal = self.rolling_zero_rate(y_arr)
        valid_idx = np.flatnonzero(np.isfinite(signal))

        run_length_path = np.full(len(y_arr), np.nan, dtype=float)
        if len(valid_idx) == 0:
            return {
                "change_points": [],
                "run_lengths": run_length_path.tolist(),
                "final_zero_rate": float("nan"),
                "regime_changed": False,
                "recommended_action": "NO_ACTION",
                "change_point_dates": [],
            }

        run_probs = np.array([1.0], dtype=float)
        posteriors: List[_PosteriorState] = [
            _PosteriorState(self.mu0, self.kappa0, self.alpha0, self.beta0)
        ]
        change_points: List[int] = []
        warmup = max(30, 10)

        for idx in valid_idx:
            x_t = float(signal[idx])
            pred = np.array([self._predictive_pdf(x_t, state) for state in posteriors], dtype=float)
            pred = np.clip(pred, 1e-300, None)

            growth = run_probs * pred * (1.0 - self.hazard)
            cp_prob = float(np.sum(run_probs * pred * self.hazard))

            new_probs = np.empty(len(run_probs) + 1, dtype=float)
            new_probs[0] = cp_prob
            new_probs[1:] = growth
            total = float(np.sum(new_probs))
            if not np.isfinite(total) or total <= 0.0:
                LOGGER.warning("BOCPD normalization failed at index %s; resetting posterior.", idx)
                new_probs = np.zeros_like(new_probs)
                new_probs[0] = 1.0
            else:
                new_probs /= total

            cp_mass = float(np.sum(new_probs[: min(6, len(new_probs))]))
            map_run_length = float(np.argmax(new_probs))
            run_length_path[idx] = map_run_length
            if idx >= warmup and cp_mass > 0.5:
                change_points.append(int(idx))

            updated_prior = self._update_state(
                _PosteriorState(self.mu0, self.kappa0, self.alpha0, self.beta0),
                x_t,
            )
            updated_growth = [self._update_state(state, x_t) for state in posteriors]
            posteriors = [updated_prior, *updated_growth]
            run_probs = new_probs

        final_zero_rate = float(np.nanmean(signal[max(0, len(signal) - 30) :]))
        start_zero_rate = float(signal[valid_idx[0]])
        current_intermittent = final_zero_rate >= 0.70
        start_intermittent = start_zero_rate >= 0.70
        regime_flips = self._threshold_crossings(signal, threshold=0.70)
        if current_intermittent != start_intermittent:
            change_points = sorted({*change_points, *regime_flips})
        recent_change = any(cp >= max(0, len(y_arr) - 60) for cp in change_points)
        regime_changed = bool(recent_change and (current_intermittent != start_intermittent))

        if regime_changed and (not start_intermittent) and current_intermittent:
            recommended_action = "REROUTE_TO_HURDLE"
        elif regime_changed and start_intermittent and (not current_intermittent):
            recommended_action = "REROUTE_TO_SARIMAX"
        elif recent_change:
            recommended_action = "MONITOR"
        else:
            recommended_action = "NO_ACTION"

        return {
            "change_points": change_points,
            "run_lengths": run_length_path.tolist(),
            "final_zero_rate": final_zero_rate,
            "regime_changed": regime_changed,
            "recommended_action": recommended_action,
            "change_point_dates": change_points,
        }

    @staticmethod
    def trigger_reroute(
        y: np.ndarray,
        original_zero_rate: float,
        threshold: float = 0.70,
    ) -> bool:
        r"""
        Trigger a lightweight daily reroute check from the latest 30 days.

        The fast production rule compares regime labels:

        .. math::
            \text{flip} =
            \mathbf{1}\left[\hat z_{t,30} \ge \tau\right] \ne
            \mathbf{1}\left[z_{\text{original}} \ge \tau\right]

        Parameters
        ----------
        y : np.ndarray
            Latest daily demand history.
        original_zero_rate : float
            Zero-rate observed when the product was initially routed.
        threshold : float, default=0.70
            Regime cutoff.

        Returns
        -------
        bool
            ``True`` when the regime classification flipped.

        Example
        -------
        >>> y = np.zeros(30, dtype=float)
        >>> BayesianOnlineChangepointDetector.trigger_reroute(y, original_zero_rate=0.10)
        True
        """
        values = np.asarray(y, dtype=float)
        if len(values) == 0:
            return False

        lookback = values[-30:]
        current_zero_rate = float(np.mean(np.nan_to_num(lookback, nan=0.0) == 0))
        current_regime = current_zero_rate >= threshold
        original_regime = float(original_zero_rate) >= threshold
        return bool(current_regime != original_regime)

    @staticmethod
    def _update_state(state: _PosteriorState, x_t: float) -> _PosteriorState:
        """Update a Normal-Inverse-Gamma posterior with one observation."""
        kappa_n = state.kappa + 1.0
        mu_n = (state.kappa * state.mu + x_t) / kappa_n
        alpha_n = state.alpha + 0.5
        beta_n = state.beta + 0.5 * (state.kappa * (x_t - state.mu) ** 2) / kappa_n
        return _PosteriorState(mu=mu_n, kappa=kappa_n, alpha=alpha_n, beta=beta_n)

    @staticmethod
    def _predictive_pdf(x_t: float, state: _PosteriorState) -> float:
        """Student-t predictive density induced by the Normal-Inverse-Gamma prior."""
        nu = 2.0 * state.alpha
        scale_sq = (state.beta * (state.kappa + 1.0)) / max(state.alpha * state.kappa, 1e-12)
        scale_sq = max(scale_sq, 1e-12)
        scale = float(np.sqrt(scale_sq))
        z = (x_t - state.mu) / scale

        log_pdf = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(scale)
            - ((nu + 1.0) / 2.0) * np.log1p((z * z) / nu)
        )
        return float(np.exp(log_pdf))

    @staticmethod
    def _threshold_crossings(signal: np.ndarray, threshold: float) -> List[int]:
        """Detect regime flips directly from the rolling zero-rate threshold."""
        valid_idx = np.flatnonzero(np.isfinite(signal))
        if len(valid_idx) < 2:
            return []

        crossings: List[int] = []
        prev_flag = bool(signal[valid_idx[0]] >= threshold)
        for idx in valid_idx[1:]:
            current_flag = bool(signal[idx] >= threshold)
            if current_flag != prev_flag:
                crossings.append(int(idx))
            prev_flag = current_flag
        return crossings


if __name__ == "__main__":
    detector = BayesianOnlineChangepointDetector()
    synthetic = np.r_[np.full(90, 5.0), np.zeros(120)]
    detected = detector.detect(synthetic)
    fast_flip = detector.trigger_reroute(np.zeros(30, dtype=float), original_zero_rate=0.10)

    print("BOCPD detect:", "PASS" if detected["recommended_action"] == "REROUTE_TO_HURDLE" else "FAIL")
    print("Fast reroute:", "PASS" if fast_flip else "FAIL")
