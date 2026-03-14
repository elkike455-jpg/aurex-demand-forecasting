import numpy as np

from .hurdle_model import HurdleModel
from .regime_detection import MarkovRegimeDetector
from .sarimax_model import SARIMAXModel


class RegimeForecastEngine:
    """Selects forecasting strategy using regime signals + validation guardrails."""

    def __init__(self):
        """Track selected model name and diagnostics for reporting."""
        self.model_name = None
        self.selection_info = {}

    @staticmethod
    def _select_intermittent_regime(y_train, regimes):
        """
        Map arbitrary Markov labels to semantic regime meaning.
        Intermittent regime = highest zero-rate (tie-break: lower mean demand).
        """
        y = np.asarray(y_train, dtype=float)
        r = np.asarray(regimes)

        candidates = []
        for regime_id in np.unique(r):
            mask = r == regime_id
            if mask.sum() == 0:
                continue
            y_reg = y[mask]
            zero_rate = float((y_reg == 0).mean())
            mean_demand = float(np.mean(y_reg))
            candidates.append((zero_rate, -mean_demand, int(regime_id)))

        if not candidates:
            raise ValueError("No valid regimes found for mapping.")

        return sorted(candidates, reverse=True)[0][2]

    @staticmethod
    def _mae(y_true, y_pred):
        """MAE helper used in rolling model selection."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _structural_force_hurdle(y_train):
        """
        Hard guardrails: bypass SARIMAX on clearly intermittent/high-dispersion series.
        """
        y = np.asarray(y_train, dtype=float)
        if len(y) == 0:
            return True, "empty_series"

        zero_rate = float((y == 0).mean())
        mean = float(np.mean(y))
        std = float(np.std(y))
        cv = std / max(mean, 1e-6)

        if zero_rate >= 0.20:
            return True, "high_zero_rate_guardrail"

        if zero_rate >= 0.10 and cv >= 1.50:
            return True, "high_dispersion_guardrail"

        return False, None

    def _rolling_splits(self, n, val_days, n_folds=3):
        """Generate time-ordered rolling splits for robust validation."""
        splits = []
        last_train_end = n - val_days
        if last_train_end <= max(90, val_days):
            return splits

        max_back = min(last_train_end - max(90, val_days), val_days * (n_folds - 1))
        step = max(1, max_back // max(1, (n_folds - 1)))

        for i in range(n_folds):
            train_end = last_train_end - (n_folds - 1 - i) * step
            val_start = train_end
            val_end = min(train_end + val_days, n)
            if train_end < max(90, val_days) or (val_end - val_start) < 20:
                continue
            splits.append((train_end, val_start, val_end))

        return splits

    def _validation_select_model(self, y_train, X_train, regime_prefers_hurdle):
        """
        Compare Hurdle vs SARIMAX on rolling folds.
        Keep regime prior when both models are too close.
        """
        n = len(y_train)
        if n < 260:
            return regime_prefers_hurdle, {
                "used_validation": False,
                "reason": "short_series",
            }

        val_days = min(120, max(45, int(n * 0.12)))
        splits = self._rolling_splits(n, val_days, n_folds=3)

        if not splits:
            return regime_prefers_hurdle, {
                "used_validation": False,
                "reason": "no_valid_rolling_splits",
            }

        h_maes = []
        s_maes = []

        for train_end, val_start, val_end in splits:
            y_subtrain = y_train[:train_end]
            X_subtrain = X_train[:train_end]
            y_val = y_train[val_start:val_end]
            X_val = X_train[val_start:val_end]

            try:
                h_model = HurdleModel().fit(X_subtrain, y_subtrain)
                h_pred, _, _ = h_model.forecast(X_val, X_subtrain, y_subtrain)
                h_maes.append(self._mae(y_val, h_pred))
            except Exception:
                pass

            try:
                s_model = SARIMAXModel().fit(y_subtrain, X_subtrain)
                s_pred, _, _ = s_model.forecast(len(y_val), X_val)
                s_maes.append(self._mae(y_val, s_pred))
            except Exception:
                pass

        metrics = {
            "used_validation": True,
            "val_days": int(val_days),
            "n_val_folds": int(len(splits)),
        }

        avg_h = float(np.mean(h_maes)) if h_maes else np.inf
        avg_s = float(np.mean(s_maes)) if s_maes else np.inf

        metrics["val_mae_hurdle"] = float(avg_h) if np.isfinite(avg_h) else None
        metrics["val_mae_sarimax"] = float(avg_s) if np.isfinite(avg_s) else None

        if (not np.isfinite(avg_h)) and (not np.isfinite(avg_s)):
            metrics["reason"] = "both_val_models_failed"
            return regime_prefers_hurdle, metrics

        if (not np.isfinite(avg_h)) and np.isfinite(avg_s):
            metrics["reason"] = "hurdle_val_failed"
            return False, metrics

        if np.isfinite(avg_h) and (not np.isfinite(avg_s)):
            metrics["reason"] = "sarimax_val_failed"
            return True, metrics

        rel_gap = abs(avg_h - avg_s) / max(min(avg_h, avg_s), 1e-6)
        metrics["relative_gap"] = float(rel_gap)

        if rel_gap < 0.03:
            metrics["reason"] = "models_close_keep_regime_prior"
            return regime_prefers_hurdle, metrics

        choose_hurdle = avg_h < avg_s
        metrics["reason"] = "rolling_validation_override"
        return choose_hurdle, metrics

    def run(self, y_train, X_train, y_test, X_test):
        """
        Main orchestration:
        1) infer regime prior,
        2) apply structural guardrails,
        3) optional rolling validation selection,
        4) fit chosen model + safety fallback.
        """
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        self.selection_info = {}

        try:
            # Markov regime signal is used as a prior, not as a hard decision.
            detector = MarkovRegimeDetector()
            detector.fit(np.log1p(np.clip(y_train, 0.0, None)))
            _, regimes = detector.get_regime_probabilities()

            intermittent_regime = self._select_intermittent_regime(y_train, regimes)
            intermittent_share = float((regimes == intermittent_regime).mean())
            regime_prefers_hurdle = intermittent_share > 0.5

            self.selection_info["regime_mode"] = "markov"
            self.selection_info["intermittent_share"] = intermittent_share
        except Exception:
            # Fallback prior when Markov cannot be identified.
            regime_prefers_hurdle = (y_train == 0).mean() >= 0.5
            self.selection_info["regime_mode"] = "zero_rate_fallback"
            self.selection_info["intermittent_share"] = float((y_train == 0).mean())

        force_hurdle, guardrail_reason = self._structural_force_hurdle(y_train)

        if force_hurdle:
            use_hurdle = True
            self.selection_info["used_validation"] = False
            self.selection_info["reason"] = guardrail_reason
        else:
            use_hurdle, val_info = self._validation_select_model(y_train, X_train, regime_prefers_hurdle)
            self.selection_info.update(val_info)

        if use_hurdle:
            model = HurdleModel().fit(X_train, y_train)
            y_pred, conf_low, conf_up = model.forecast(X_test, X_train, y_train)
            self.model_name = "Regime-Hurdle"
            return y_pred, conf_low, conf_up, self.model_name

        model = SARIMAXModel().fit(y_train, X_train)
        y_pred, conf_low, conf_up = model.forecast(len(y_test), X_test)

        # Last-resort fallback if SARIMAX forecast still explodes.
        train_mean = max(float(np.mean(y_train)), 1e-6)
        exploded = (not np.all(np.isfinite(y_pred))) or (float(np.mean(y_pred)) > 20.0 * train_mean)

        if exploded:
            model = HurdleModel().fit(X_train, y_train)
            y_pred, conf_low, conf_up = model.forecast(X_test, X_train, y_train)
            self.model_name = "Regime-Hurdle"
            self.selection_info["reason"] = "sarimax_exploded_fallback_hurdle"
        else:
            self.model_name = "Regime-SARIMAX"

        return y_pred, conf_low, conf_up, self.model_name
