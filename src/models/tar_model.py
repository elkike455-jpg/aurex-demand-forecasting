import numpy as np

from .hurdle_model import HurdleModel


class TARModel:
    """Two-regime threshold autoregressive model with recursive multi-step forecasts."""

    def __init__(self, threshold_delay=7, lag_orders=(1, 7, 14), ridge_alpha=1e-3):
        """Store fitted TAR parameters and forecast guardrails."""
        self.threshold_delay = int(threshold_delay)
        self.lag_orders = tuple(sorted(int(lag) for lag in lag_orders))
        self.ridge_alpha = float(ridge_alpha)
        self.threshold_ = None
        self.low_coef_ = None
        self.high_coef_ = None
        self.low_sigma_ = 0.0
        self.high_sigma_ = 0.0
        self.fallback_coef_ = None
        self.fallback_sigma_ = 0.0
        self.upper_cap = None
        self.last_week = None
        self.train_y_ = None
        self.n_exog_ = 0
        self.best_config_ = None
        self.use_stable_fallback_ = False
        self.stable_level_ = 0.0
        self.use_hurdle_fallback_ = False
        self.hurdle_model_ = None
        self.train_X_ = None

    @staticmethod
    def _safe_log(y):
        """Clean and stabilize the target before autoregressive fitting."""
        y = np.asarray(y, dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, 0.0, None)
        return np.log1p(y)

    @staticmethod
    def _mae(y_true, y_pred):
        """Compute MAE for threshold selection."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    def _weekly_naive(self, steps):
        """Weekly seasonal naive baseline used to stabilize TAR forecasts."""
        if self.last_week is None or len(self.last_week) == 0:
            return np.zeros(int(steps), dtype=float)
        return np.resize(self.last_week, int(steps)).astype(float)

    def _stable_baseline(self, steps):
        """Stable fallback forecast for series where TAR validation is unreliable."""
        steps = int(steps)
        naive = self._weekly_naive(steps)
        y_pred = 0.7 * naive + 0.3 * float(self.stable_level_)
        spread = max(float(self.fallback_sigma_), 0.5 * float(np.std(self.last_week)) if self.last_week is not None and len(self.last_week) else 0.0)
        conf_low = np.maximum(y_pred - 1.96 * spread, 0.0)
        conf_up = y_pred + 1.96 * spread
        y_pred = np.clip(y_pred, 0.0, self.upper_cap)
        conf_low = np.clip(conf_low, 0.0, self.upper_cap)
        conf_up = np.clip(conf_up, 0.0, max(self.upper_cap * 1.2, 1.0))
        conf_low = np.minimum(conf_low, y_pred)
        conf_up = np.maximum(conf_up, y_pred)
        return y_pred, conf_low, conf_up

    def _score_hurdle_holdout(self, y_raw, X):
        """Score a Hurdle baseline on the same time-ordered holdout split used for TAR."""
        min_history = 60
        if len(y_raw) < min_history + 20:
            return None

        val_size = min(max(45, int(len(y_raw) * 0.15)), len(y_raw) - min_history)
        if val_size < 20:
            return None

        split = len(y_raw) - val_size
        y_sub = y_raw[:split]
        X_sub = X[:split]
        y_val = y_raw[split:]
        X_val = X[split:]

        try:
            model = HurdleModel().fit(X_sub, y_sub)
            preds, _, _ = model.forecast(X_val, X_sub, y_sub)
        except Exception:
            return None

        return {
            "score": float(self._mae(y_val, preds)),
            "bias_ratio": float(abs(np.mean(preds - y_val)) / max(float(np.mean(y_val)), 1e-6)),
        }

    def _design_row(self, y_log_history, x_row):
        """Build one design row from lagged target values plus contemporaneous exogenous data."""
        parts = [1.0]
        for lag in self.lag_orders:
            parts.append(float(y_log_history[-lag]))
        if self.n_exog_:
            parts.extend(np.asarray(x_row, dtype=float).tolist())
        return np.asarray(parts, dtype=float)

    def _build_training_matrices(self, y_raw, y_log, X, threshold_delay=None, lag_orders=None):
        """Create aligned design rows for all trainable timestamps."""
        threshold_delay = int(self.threshold_delay if threshold_delay is None else threshold_delay)
        lag_orders = self.lag_orders if lag_orders is None else tuple(sorted(int(lag) for lag in lag_orders))
        start = max(max(lag_orders), threshold_delay)
        rows = []
        targets_log = []
        thresholds = []

        for t in range(start, len(y_raw)):
            row = [1.0]
            for lag in lag_orders:
                row.append(float(y_log[t - lag]))
            if self.n_exog_:
                row.extend(np.asarray(X[t], dtype=float).tolist())
            rows.append(np.asarray(row, dtype=float))
            targets_log.append(float(y_log[t]))
            thresholds.append(float(y_raw[t - threshold_delay]))

        if not rows:
            return (
                np.empty((0, 1 + len(lag_orders) + self.n_exog_), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )

        return np.vstack(rows), np.asarray(targets_log, dtype=float), np.asarray(thresholds, dtype=float)

    def _solve_ridge(self, X, y):
        """Solve a small ridge regression system without external dependencies."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        penalty = self.ridge_alpha * np.eye(n_features)
        penalty[0, 0] = 0.0
        return np.linalg.solve(X.T @ X + penalty, X.T @ y)

    def _fit_threshold_split(self, X_all, y_all_log, threshold_values, split_value):
        """Fit separate regressions for low/high regimes for one threshold candidate."""
        mask_low = threshold_values <= split_value
        mask_high = ~mask_low

        min_rows = max(X_all.shape[1] + 2, 15)
        if mask_low.sum() < min_rows or mask_high.sum() < min_rows:
            return None

        low_coef = self._solve_ridge(X_all[mask_low], y_all_log[mask_low])
        high_coef = self._solve_ridge(X_all[mask_high], y_all_log[mask_high])

        pred_log = np.empty_like(y_all_log)
        pred_log[mask_low] = X_all[mask_low] @ low_coef
        pred_log[mask_high] = X_all[mask_high] @ high_coef

        resid = y_all_log - pred_log
        low_sigma = float(np.std(resid[mask_low])) if mask_low.any() else 0.0
        high_sigma = float(np.std(resid[mask_high])) if mask_high.any() else 0.0

        return {
            "threshold": float(split_value),
            "low_coef": low_coef,
            "high_coef": high_coef,
            "low_sigma": low_sigma,
            "high_sigma": high_sigma,
        }

    def _predict_from_split(self, X, threshold_values, fit_result):
        """Generate one-step predictions from a fitted threshold split."""
        mask_low = threshold_values <= fit_result["threshold"]
        mask_high = ~mask_low
        pred_log = np.empty(X.shape[0], dtype=float)
        pred_log[mask_low] = X[mask_low] @ fit_result["low_coef"]
        pred_log[mask_high] = X[mask_high] @ fit_result["high_coef"]
        return pred_log

    def _fit_single_config(self, y_raw, y_log, X, threshold_delay, lag_orders):
        """Fit one TAR configuration on the supplied data."""
        X_all, y_all_log, threshold_values = self._build_training_matrices(
            y_raw,
            y_log,
            X,
            threshold_delay=threshold_delay,
            lag_orders=lag_orders,
        )

        if len(y_all_log) == 0:
            return None

        fallback_coef = self._solve_ridge(X_all, y_all_log)
        fallback_pred_log = X_all @ fallback_coef
        fallback_sigma = float(np.std(y_all_log - fallback_pred_log))

        candidate_thresholds = np.unique(
            np.percentile(
                threshold_values,
                [15, 30, 45, 55, 70, 85],
            )
        )

        best_fit = None
        best_mae = np.inf

        for threshold in candidate_thresholds:
            fit_result = self._fit_threshold_split(X_all, y_all_log, threshold_values, threshold)
            if fit_result is None:
                continue

            pred_log = self._predict_from_split(X_all, threshold_values, fit_result)
            pred = np.maximum(np.expm1(np.clip(pred_log, -10, 12)), 0.0)
            pred = np.clip(pred, 0.0, self.upper_cap)
            score = self._mae(y_raw[-len(y_all_log):], pred)

            if np.isfinite(score) and score < best_mae:
                best_mae = score
                best_fit = fit_result

        return {
            "threshold_delay": int(threshold_delay),
            "lag_orders": tuple(lag_orders),
            "best_fit": best_fit,
            "fallback_coef": fallback_coef,
            "fallback_sigma": fallback_sigma,
            "score": float(best_mae) if np.isfinite(best_mae) else np.inf,
        }

    def _score_config_holdout(self, y_raw, X, threshold_delay, lag_orders):
        """Score one TAR configuration on a time-ordered holdout block."""
        min_history = max(max(lag_orders), threshold_delay) + 30
        if len(y_raw) < min_history + 20:
            return None

        val_size = min(max(45, int(len(y_raw) * 0.15)), len(y_raw) - min_history)
        if val_size < 20:
            return None

        split = len(y_raw) - val_size
        y_sub = y_raw[:split]
        X_sub = X[:split]
        y_val = y_raw[split:]
        X_val = X[split:]

        y_sub_log = self._safe_log(y_sub)
        trained = self._fit_single_config(y_sub, y_sub_log, X_sub, threshold_delay, lag_orders)
        if trained is None:
            return None

        fit_result = trained["best_fit"]
        fallback_coef = trained["fallback_coef"]

        history_raw = list(np.asarray(y_sub, dtype=float))
        history_log = list(self._safe_log(history_raw))
        preds = np.zeros(len(y_val), dtype=float)

        for i in range(len(y_val)):
            if len(history_log) < max(lag_orders):
                pred_log = float(np.mean(history_log)) if history_log else 0.0
            else:
                row = [1.0]
                for lag in lag_orders:
                    row.append(float(history_log[-lag]))
                if self.n_exog_:
                    row.extend(np.asarray(X_val[i], dtype=float).tolist())
                row = np.asarray(row, dtype=float)

                if fit_result is not None and len(history_raw) >= threshold_delay:
                    threshold_source = float(history_raw[-threshold_delay])
                    coef = fit_result["low_coef"] if threshold_source <= fit_result["threshold"] else fit_result["high_coef"]
                    pred_log = float(row @ coef)
                else:
                    pred_log = float(row @ fallback_coef)

            pred = float(np.maximum(np.expm1(np.clip(pred_log, -10, 12)), 0.0))
            pred = float(np.clip(pred, 0.0, self.upper_cap))
            preds[i] = pred

            # Use recursive rollout in validation so TAR is scored under real forecast conditions.
            history_raw.append(pred)
            history_log.append(float(np.log1p(pred)))

        naive = np.resize(y_sub[-7:] if len(y_sub) >= 7 else y_sub, len(y_val)).astype(float)
        if len(naive) != len(y_val):
            naive = np.zeros(len(y_val), dtype=float)
        stable_level = float(np.mean(y_sub[-28:])) if len(y_sub) else 0.0
        stable_baseline = 0.7 * naive + 0.3 * stable_level
        preds = 0.85 * preds + 0.15 * naive

        score = self._mae(y_val, preds)
        naive_score = self._mae(y_val, naive)
        stable_score = self._mae(y_val, stable_baseline)
        val_mean = max(float(np.mean(y_val)), 1e-6)
        val_std = max(float(np.std(y_val)), 1e-6)
        bias_ratio = abs(float(np.mean(preds - y_val))) / val_mean
        variance_ratio = float(np.std(preds)) / val_std
        unstable = (
            score > min(naive_score, stable_score) * 1.03
            or bias_ratio > 0.20
            or variance_ratio > 1.20
        )
        if unstable:
            score = max(score, stable_score) + abs(score - stable_score)

        return {
            "threshold_delay": int(threshold_delay),
            "lag_orders": tuple(lag_orders),
            "score": float(score),
            "raw_score": float(self._mae(y_val, preds)),
            "naive_score": float(naive_score),
            "stable_score": float(stable_score),
            "bias_ratio": float(bias_ratio),
            "variance_ratio": float(variance_ratio),
            "unstable": bool(unstable),
        }

    def fit(self, y_train, X_train):
        """Fit a simple two-regime TAR model and choose the threshold by holdout MAE."""
        y_raw = np.asarray(y_train, dtype=float)
        X = np.asarray(X_train, dtype=float)

        y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = np.clip(y_raw, 0.0, None)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_exog_ = X.shape[1] if X.ndim == 2 else 0
        self.train_y_ = y_raw.copy()
        self.train_X_ = X.copy()
        self.use_stable_fallback_ = False
        self.use_hurdle_fallback_ = False

        train_mean = float(np.mean(y_raw)) if len(y_raw) else 0.0
        p99 = float(np.percentile(y_raw, 99)) if len(y_raw) else 0.0
        train_std = float(np.std(y_raw)) if len(y_raw) else 0.0
        self.upper_cap = max(1.0, 1.5 * p99, 5.0 * train_mean)
        self.last_week = y_raw[-7:] if len(y_raw) >= 7 else y_raw.copy()
        self.stable_level_ = float(np.mean(y_raw[-28:])) if len(y_raw) else 0.0
        zero_rate = float((y_raw == 0).mean()) if len(y_raw) else 0.0
        cv = train_std / max(train_mean, 1e-6)

        y_log = self._safe_log(y_raw)
        candidate_delays = sorted({1, self.threshold_delay, 7, 14, 28})
        candidate_lag_sets = [
            tuple(sorted(set(self.lag_orders))),
            (1, 7, 14, 28),
            (1, 2, 7, 14),
        ]

        best_config = None
        best_score = np.inf

        for delay in candidate_delays:
            for lag_orders in candidate_lag_sets:
                if max(lag_orders) >= len(y_raw):
                    continue
                scored = self._score_config_holdout(y_raw, X, delay, lag_orders)
                if scored is None:
                    continue
                if np.isfinite(scored["score"]) and scored["score"] < best_score:
                    best_score = scored["score"]
                    best_config = scored

        if best_config is None:
            best_config = {
                "threshold_delay": self.threshold_delay,
                "lag_orders": tuple(sorted(set(self.lag_orders))),
            }

        self.threshold_delay = int(best_config["threshold_delay"])
        self.lag_orders = tuple(best_config["lag_orders"])
        self.best_config_ = dict(best_config)
        if best_config.get("unstable", False):
            self.use_stable_fallback_ = True
        if zero_rate >= 0.12 and best_config.get("variance_ratio", 0.0) > 0.95:
            self.use_stable_fallback_ = True
        if best_config.get("bias_ratio", 0.0) > 0.12:
            self.use_stable_fallback_ = True

        hurdle_score = self._score_hurdle_holdout(y_raw, X)
        if hurdle_score is not None:
            self.best_config_["hurdle_holdout_mae"] = hurdle_score["score"]
            tar_holdout = float(best_config.get("score", np.inf))
            self.best_config_["tar_holdout_mae"] = tar_holdout
            if hurdle_score["score"] <= tar_holdout:
                self.use_hurdle_fallback_ = True

        if zero_rate >= 0.20:
            self.use_hurdle_fallback_ = True
        if zero_rate >= 0.10 and cv >= 1.25:
            self.use_hurdle_fallback_ = True
        if cv >= 1.75:
            self.use_hurdle_fallback_ = True

        try:
            self.hurdle_model_ = HurdleModel().fit(X, y_raw)
        except Exception:
            self.hurdle_model_ = None

        trained = self._fit_single_config(y_raw, y_log, X, self.threshold_delay, self.lag_orders)
        if trained is None:
            n_features = 1 + len(self.lag_orders) + self.n_exog_
            self.fallback_coef_ = np.zeros(n_features, dtype=float)
            self.fallback_coef_[0] = float(np.mean(y_log)) if len(y_log) else 0.0
            return self

        self.fallback_coef_ = trained["fallback_coef"]
        self.fallback_sigma_ = trained["fallback_sigma"]

        best_fit = trained["best_fit"]
        if best_fit is None:
            self.threshold_ = None
            self.use_stable_fallback_ = True
            return self

        self.threshold_ = best_fit["threshold"]
        self.low_coef_ = best_fit["low_coef"]
        self.high_coef_ = best_fit["high_coef"]
        self.low_sigma_ = best_fit["low_sigma"]
        self.high_sigma_ = best_fit["high_sigma"]
        return self

    def forecast(self, steps, X_test):
        """Recursively forecast future demand using the fitted threshold rule."""
        X_test = np.asarray(X_test, dtype=float)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        steps = int(steps)
        if self.use_hurdle_fallback_ and self.hurdle_model_ is not None:
            return self.hurdle_model_.forecast(X_test, self.train_X_, self.train_y_)
        if self.use_stable_fallback_:
            return self._stable_baseline(steps)

        history_raw = list(np.asarray(self.train_y_, dtype=float))
        history_log = list(self._safe_log(history_raw))

        y_pred = np.zeros(steps, dtype=float)
        conf_low = np.zeros(steps, dtype=float)
        conf_up = np.zeros(steps, dtype=float)

        use_threshold = (
            self.threshold_ is not None
            and self.low_coef_ is not None
            and self.high_coef_ is not None
        )

        for i in range(steps):
            x_row = X_test[i] if i < len(X_test) else np.zeros(self.n_exog_, dtype=float)

            if len(history_log) < max(self.lag_orders):
                pred_log = float(np.mean(history_log)) if history_log else 0.0
                sigma = self.fallback_sigma_
            else:
                row = self._design_row(np.asarray(history_log, dtype=float), x_row)
                if use_threshold and len(history_raw) >= self.threshold_delay:
                    threshold_source = float(history_raw[-self.threshold_delay])
                    if threshold_source <= self.threshold_:
                        pred_log = float(row @ self.low_coef_)
                        sigma = self.low_sigma_
                    else:
                        pred_log = float(row @ self.high_coef_)
                        sigma = self.high_sigma_
                else:
                    pred_log = float(row @ self.fallback_coef_)
                    sigma = self.fallback_sigma_

            pred_log = float(np.clip(pred_log, -10, 12))
            pred = float(np.maximum(np.expm1(pred_log), 0.0))
            pred = float(np.clip(pred, 0.0, self.upper_cap))

            y_pred[i] = pred
            conf_low[i] = max(pred - 1.96 * sigma, 0.0)
            conf_up[i] = pred + 1.96 * sigma

            history_raw.append(pred)
            history_log.append(np.log1p(pred))

        naive = self._weekly_naive(steps)
        alpha = 0.75
        y_pred = alpha * y_pred + (1.0 - alpha) * naive
        conf_low = alpha * conf_low + (1.0 - alpha) * naive
        conf_up = alpha * conf_up + (1.0 - alpha) * naive

        y_pred = np.clip(y_pred, 0.0, self.upper_cap)
        conf_low = np.clip(conf_low, 0.0, self.upper_cap)
        conf_up = np.clip(conf_up, 0.0, max(self.upper_cap * 1.2, 1.0))
        conf_low = np.minimum(conf_low, y_pred)
        conf_up = np.maximum(conf_up, y_pred)

        return y_pred, conf_low, conf_up
