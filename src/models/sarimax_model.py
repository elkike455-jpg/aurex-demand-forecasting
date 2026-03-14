import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXModel:
    """Stabilized SARIMAX wrapper with small order search and forecast guardrails."""

    def __init__(self):
        """Store fitted model artifacts and stability parameters."""
        self.model = None
        self.results = None
        self.best_order = None
        self.best_seasonal_order = None
        self.upper_cap = None
        self.zero_rate = None
        self.last_week = None

    @staticmethod
    def _mae(y_true, y_pred):
        """Compute MAE used in internal order selection."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _safe_log_target(y):
        """Clean target, cap extreme spikes, then apply log1p transform."""
        y = np.asarray(y, dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        cap = np.percentile(y, 99.5) if len(y) else 0.0
        y_capped = np.clip(y, 0.0, cap)
        return np.log1p(y_capped)

    def _fit_single(self, y_train_log, X_train, order, seasonal_order):
        """Fit one SARIMAX specification and return fitted results."""
        model = SARIMAX(
            y_train_log,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)

    def _weekly_naive(self, steps):
        """Weekly seasonal naive baseline used for forecast stabilization blend."""
        if self.last_week is None or len(self.last_week) == 0:
            return np.zeros(steps, dtype=float)
        return np.resize(self.last_week, steps).astype(float)

    def fit(self, y_train, X_train):
        """Fit SARIMAX and select among a small candidate spec set via holdout MAE."""
        y_train = np.asarray(y_train, dtype=float)
        X_train = np.asarray(X_train, dtype=float)

        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Precompute structural stats for prediction caps and blend strength.
        self.zero_rate = float((y_train == 0).mean()) if len(y_train) else 0.0
        train_mean = float(np.mean(y_train)) if len(y_train) else 0.0
        p99 = float(np.percentile(y_train, 99)) if len(y_train) else 0.0
        self.upper_cap = max(1.0, 1.5 * p99, 5.0 * train_mean)
        self.last_week = y_train[-7:] if len(y_train) >= 7 else y_train.copy()

        specs = [
            ((1, 0, 1), (0, 1, 1, 7)),
            ((1, 0, 1), (1, 0, 1, 7)),
            ((2, 0, 1), (1, 0, 1, 7)),
        ]

        n = len(y_train)
        use_validation = n >= 220
        val_days = min(90, max(30, int(n * 0.12))) if use_validation else 0

        best_mae = np.inf
        best_spec = specs[0]

        if use_validation:
            # Time-aware split: latest block is validation.
            split = n - val_days
            y_sub = y_train[:split]
            X_sub = X_train[:split]
            y_val = y_train[split:]
            X_val = X_train[split:]

            y_sub_log = self._safe_log_target(y_sub)

            for order, seasonal_order in specs:
                try:
                    res = self._fit_single(y_sub_log, X_sub, order, seasonal_order)
                    pred = res.get_forecast(steps=len(y_val), exog=X_val)
                    pred_log = np.clip(np.asarray(pred.predicted_mean), -10, 12)
                    y_hat = np.maximum(np.expm1(pred_log), 0.0)
                    y_hat = np.clip(y_hat, 0.0, self.upper_cap)
                    mae = self._mae(y_val, y_hat)
                    if np.isfinite(mae) and mae < best_mae:
                        best_mae = mae
                        best_spec = (order, seasonal_order)
                except Exception:
                    continue

        self.best_order, self.best_seasonal_order = best_spec

        # Refit best spec on full train.
        y_train_log = self._safe_log_target(y_train)
        self.results = self._fit_single(y_train_log, X_train, self.best_order, self.best_seasonal_order)
        return self

    def forecast(self, steps, X_test):
        """Forecast with clipping and weekly-naive blend to reduce variance explosions."""
        X_test = np.asarray(X_test, dtype=float)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        pred = self.results.get_forecast(steps=steps, exog=X_test)

        y_pred_log = np.clip(np.asarray(pred.predicted_mean), -10, 12)
        y_pred = np.maximum(np.expm1(y_pred_log), 0.0)

        ci = np.asarray(pred.conf_int())
        ci = np.clip(ci, -10, 12)
        conf_low = np.maximum(np.expm1(ci[:, 0]), 0.0)
        conf_up = np.maximum(np.expm1(ci[:, 1]), 0.0)

        # Hard cap protects downstream metrics from explosive tails.
        y_pred = np.clip(y_pred, 0.0, self.upper_cap)
        conf_low = np.clip(conf_low, 0.0, self.upper_cap)
        conf_up = np.clip(conf_up, 0.0, max(self.upper_cap * 1.2, 1.0))

        # Blend with seasonal naive for smoother, more robust output.
        naive = self._weekly_naive(steps)
        alpha = 0.85 if (self.zero_rate or 0.0) < 0.15 else 0.65
        y_pred = alpha * y_pred + (1.0 - alpha) * naive
        conf_low = alpha * conf_low + (1.0 - alpha) * naive
        conf_up = alpha * conf_up + (1.0 - alpha) * naive

        conf_low = np.minimum(conf_low, y_pred)
        conf_up = np.maximum(conf_up, y_pred)

        return y_pred, conf_low, conf_up
