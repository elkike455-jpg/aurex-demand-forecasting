import numpy as np

from .croston_model import CrostonSBAModel
from .hurdle_model import HurdleModel
from .regime_detection import MarkovRegimeDetector
from .sarimax_model import SARIMAXModel
from .tsb_model import TSBModel, sbc_classify


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
    def _max_zero_run(y_train):
        """Length of the longest consecutive zero streak."""
        y = np.asarray(y_train, dtype=float)
        longest = 0
        current = 0
        for value in y:
            if value == 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)

    def _series_profile(self, y_train):
        """
        Build structural diagnostics used for model routing.

        We combine SBC-style intermittent classification with zero-rate and
        longest-zero-run so ultra-sparse series can be handled more
        conservatively.
        """
        y = np.asarray(y_train, dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        nonzero = y[y > 0]
        zero_rate = float((y == 0).mean()) if len(y) else 1.0
        mean = float(np.mean(y)) if len(y) else 0.0
        std = float(np.std(y)) if len(y) else 0.0
        cv = std / max(mean, 1e-6)
        nonzero_mean = float(np.mean(nonzero)) if len(nonzero) else 0.0
        max_zero_run = self._max_zero_run(y)
        sbc = sbc_classify(y)

        ultra_sparse = (
            zero_rate >= 0.85
            or max_zero_run >= 28
            or (zero_rate >= 0.70 and nonzero_mean <= 2.5)
        )

        return {
            "zero_rate": zero_rate,
            "mean": mean,
            "std": std,
            "cv": cv,
            "nonzero_mean": nonzero_mean,
            "max_zero_run": max_zero_run,
            "sbc": sbc,
            "ultra_sparse": bool(ultra_sparse),
        }

    @staticmethod
    def _structural_force_hurdle(profile):
        """
        Hard guardrails: bypass SARIMAX on clearly intermittent/high-dispersion series.
        """
        if profile["mean"] == 0.0 and profile["zero_rate"] >= 1.0:
            return True, "empty_series"

        zero_rate = profile["zero_rate"]
        cv = profile["cv"]
        regime = profile["sbc"]["regime"]

        if regime == "lumpy":
            return True, "lumpy_series_guardrail"

        if zero_rate >= 0.20:
            return True, "high_zero_rate_guardrail"

        if zero_rate >= 0.10 and cv >= 1.50:
            return True, "high_dispersion_guardrail"

        return False, None

    @staticmethod
    def blend_forecast(y_sarimax, y_hurdle, zero_rate):
        """
        Blend stable and intermittent forecasts in the transitional zero-rate zone.

        Weight is linear in the 30%-70% interval:
            w = (zero_rate - 0.30) / (0.70 - 0.30)

        so 0.30 -> pure SARIMAX and 0.70 -> pure HURDLE.
        """
        y_s = np.asarray(y_sarimax, dtype=float)
        y_h = np.asarray(y_hurdle, dtype=float)
        if y_s.shape != y_h.shape:
            raise ValueError("blend_forecast requires arrays with the same shape.")

        w = (float(zero_rate) - 0.30) / 0.40
        w = float(np.clip(w, 0.0, 1.0))
        return (1.0 - w) * y_s + w * y_h

    def _choose_structural_model(self, profile):
        """
        Route obviously sparse series before expensive validation.

        Returns:
            "tsb", "croston_sba", "hurdle", or None when SARIMAX/Hurdle
            validation should decide.
        """
        sbc = profile["sbc"]

        if sbc["tsb_domain"]:
            return "intermittent_family", "intermittent_family_domain"

        if sbc["regime"] == "lumpy":
            return "hurdle", "lumpy_hurdle_domain"

        return None, None

    def _select_intermittent_model(self, y_train, profile):
        """
        Choose between TSB and Croston-SBA inside the intermittent family.

        We use a simple recent holdout so the engine doesn't default to TSB
        when Croston-SBA better matches the observed zero-heavy pattern.
        """
        y = np.asarray(y_train, dtype=float)
        use_weekly = bool(profile["ultra_sparse"])

        if len(y) < 90:
            model_name = "croston_sba" if profile["zero_rate"] >= 0.90 else "tsb"
            reason = (
                "intermittent_short_series_croston"
                if model_name == "croston_sba"
                else "intermittent_short_series_tsb"
            )
            return model_name, {
                "used_validation": False,
                "reason": reason,
                "intermittent_use_weekly": use_weekly,
            }

        val_size = min(90, max(28, int(len(y) * 0.15)))
        if len(y) - val_size < 45:
            val_size = max(14, len(y) // 5)

        split = len(y) - val_size
        y_subtrain = y[:split]
        y_val = y[split:]

        metrics = {
            "used_validation": True,
            "intermittent_val_days": int(len(y_val)),
            "intermittent_use_weekly": use_weekly,
        }

        tsb_mae = np.inf
        croston_mae = np.inf

        try:
            tsb_model = TSBModel(use_weekly=use_weekly, warn_if_not_tsb_domain=False).fit(y_subtrain)
            tsb_pred, _, _ = tsb_model.forecast(len(y_val))
            tsb_mae = self._mae(y_val, tsb_pred)
            metrics["val_mae_tsb"] = float(tsb_mae)
        except Exception:
            metrics["val_mae_tsb"] = None

        try:
            croston_model = CrostonSBAModel(use_weekly=use_weekly).fit(y_subtrain)
            croston_pred, _, _ = croston_model.forecast(len(y_val))
            croston_mae = self._mae(y_val, croston_pred)
            metrics["val_mae_croston_sba"] = float(croston_mae)
        except Exception:
            metrics["val_mae_croston_sba"] = None

        if (not np.isfinite(tsb_mae)) and (not np.isfinite(croston_mae)):
            fallback = "croston_sba" if profile["zero_rate"] >= 0.90 else "tsb"
            metrics["reason"] = "intermittent_both_models_failed"
            return fallback, metrics

        if np.isfinite(tsb_mae) and not np.isfinite(croston_mae):
            metrics["reason"] = "intermittent_croston_failed"
            return "tsb", metrics

        if np.isfinite(croston_mae) and not np.isfinite(tsb_mae):
            metrics["reason"] = "intermittent_tsb_failed"
            return "croston_sba", metrics

        rel_gap = abs(tsb_mae - croston_mae) / max(min(tsb_mae, croston_mae), 1e-6)
        metrics["intermittent_relative_gap"] = float(rel_gap)

        if rel_gap < 0.03:
            preferred = "croston_sba" if profile["zero_rate"] >= 0.90 else "tsb"
            metrics["reason"] = "intermittent_models_close_keep_prior"
            return preferred, metrics

        if croston_mae < tsb_mae:
            metrics["reason"] = "intermittent_validation_croston"
            return "croston_sba", metrics

        metrics["reason"] = "intermittent_validation_tsb"
        return "tsb", metrics

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
        profile = self._series_profile(y_train)
        self.selection_info["zero_rate"] = profile["zero_rate"]
        self.selection_info["cv"] = profile["cv"]
        self.selection_info["max_zero_run"] = profile["max_zero_run"]
        self.selection_info["sbc_regime"] = profile["sbc"]["regime"]
        self.selection_info["sbc_adi"] = profile["sbc"]["adi"]
        self.selection_info["sbc_cv2"] = profile["sbc"]["cv2"]
        self.selection_info["ultra_sparse"] = profile["ultra_sparse"]
        self.selection_info["routing_mode"] = None

        if profile["zero_rate"] < 0.30:
            # Paper invariant: dense/smooth products always route to SARIMAX.
            model = SARIMAXModel().fit(y_train, X_train)
            y_pred, conf_low, conf_up = model.forecast(len(y_test), X_test)
            self.model_name = "Regime-SARIMAX"
            self.selection_info["routing_mode"] = "SARIMAX"
            self.selection_info["used_validation"] = False
            self.selection_info["reason"] = "hard_zero_rate_guardrail_sarimax"
            return y_pred, conf_low, conf_up, self.model_name

        max_run = int(profile["max_zero_run"])
        if max_run > 365:
            self.selection_info = {
                "reason": "obsolescence_max_zero_run",
                "max_zero_run": max_run,
                "note": f"Product has {max_run} consecutive zero days. Likely discontinued. Forecasting unreliable.",
                "routing_mode": "OBSOLESCENCE_FLAG",
            }
            zeros = np.zeros(len(y_test))
            return zeros, zeros, zeros, "OBSOLESCENCE_FLAG"

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

        if 0.30 <= profile["zero_rate"] < 0.70:
            sarimax_model = SARIMAXModel().fit(y_train, X_train)
            sarimax_pred, sarimax_low, sarimax_up = sarimax_model.forecast(len(y_test), X_test)

            hurdle_model = HurdleModel().fit(X_train, y_train)
            hurdle_pred, hurdle_low, hurdle_up = hurdle_model.forecast(X_test, X_train, y_train)

            y_pred = self.blend_forecast(sarimax_pred, hurdle_pred, profile["zero_rate"])
            conf_low = self.blend_forecast(sarimax_low, hurdle_low, profile["zero_rate"])
            conf_up = self.blend_forecast(sarimax_up, hurdle_up, profile["zero_rate"])

            conf_low = np.minimum(conf_low, y_pred)
            conf_up = np.maximum(conf_up, y_pred)

            self.model_name = "Regime-BLEND"
            self.selection_info["routing_mode"] = "BLEND"
            self.selection_info["reason"] = "transitional_zero_rate_blend"
            self.selection_info["blend_weight"] = float(np.clip((profile["zero_rate"] - 0.30) / 0.40, 0.0, 1.0))
            return y_pred, conf_low, conf_up, self.model_name

        structural_model, structural_reason = self._choose_structural_model(profile)
        if structural_model == "intermittent_family":
            intermittent_model, intermittent_info = self._select_intermittent_model(y_train, profile)
            self.selection_info.update(intermittent_info)

            use_weekly = bool(intermittent_info.get("intermittent_use_weekly", False))
            if intermittent_model == "croston_sba":
                model = CrostonSBAModel(use_weekly=use_weekly).fit(y_train)
                y_pred, conf_low, conf_up = model.forecast(len(y_test))
                self.model_name = "Regime-CrostonSBA-Weekly" if use_weekly else "Regime-CrostonSBA"
                self.selection_info["routing_mode"] = "CROSTON_SBA"
                self.selection_info["croston_use_weekly"] = use_weekly
                self.selection_info["croston_alpha"] = model.params.get("alpha")
                return y_pred, conf_low, conf_up, self.model_name

            model = TSBModel(use_weekly=use_weekly, warn_if_not_tsb_domain=False).fit(y_train)
            y_pred, conf_low, conf_up = model.forecast(len(y_test))
            self.model_name = "Regime-TSB-Weekly" if use_weekly else "Regime-TSB"
            self.selection_info["routing_mode"] = "TSB"
            self.selection_info["tsb_use_weekly"] = use_weekly
            self.selection_info["tsb_alpha_d"] = model.params.get("alpha_d")
            self.selection_info["tsb_alpha_p"] = model.params.get("alpha_p")
            return y_pred, conf_low, conf_up, self.model_name

        force_hurdle, guardrail_reason = self._structural_force_hurdle(profile)

        if force_hurdle:
            use_hurdle = True
            self.selection_info["used_validation"] = False
            self.selection_info["reason"] = structural_reason or guardrail_reason
        else:
            use_hurdle, val_info = self._validation_select_model(y_train, X_train, regime_prefers_hurdle)
            self.selection_info.update(val_info)

        if use_hurdle:
            model = HurdleModel().fit(X_train, y_train)
            y_pred, conf_low, conf_up = model.forecast(X_test, X_train, y_train)
            self.model_name = "Regime-Hurdle"
            self.selection_info["routing_mode"] = "HURDLE"
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
            self.selection_info["routing_mode"] = "HURDLE"
            self.selection_info["reason"] = "sarimax_exploded_fallback_hurdle"
        else:
            self.model_name = "Regime-SARIMAX"
            self.selection_info["routing_mode"] = "SARIMAX"

        return y_pred, conf_low, conf_up, self.model_name
