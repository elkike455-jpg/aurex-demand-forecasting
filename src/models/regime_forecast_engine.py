import numpy as np
from .hurdle_model import HurdleModel
from .regime_detection import MarkovRegimeDetector
from .sarimax_model import SARIMAXModel


class RegimeForecastEngine:
    def __init__(self):
        self.model_name = None

    def run(self, y_train, X_train, y_test, X_test):
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            detector = MarkovRegimeDetector()
            detector.fit(np.log1p(np.clip(y_train, 0.0, None)))
            _, regimes = detector.get_regime_probabilities()
            use_hurdle = (regimes == 1).mean() > 0.5
        except Exception:
            use_hurdle = (y_train == 0).mean() >= 0.5

        if use_hurdle:
            model = HurdleModel().fit(X_train, y_train)
            y_pred, conf_low, conf_up = model.forecast(X_test, X_train, y_train)
            self.model_name = "Fallback-Hurdle"
            return y_pred, conf_low, conf_up, self.model_name

        model = SARIMAXModel().fit(y_train, X_train)
        y_pred, conf_low, conf_up = model.forecast(len(y_test), X_test)

        train_mean = max(float(np.mean(y_train)), 1e-6)
        exploded = (not np.all(np.isfinite(y_pred))) or (float(np.mean(y_pred)) > 20.0 * train_mean)

        if exploded:
            model = HurdleModel().fit(X_train, y_train)
            y_pred, conf_low, conf_up = model.forecast(X_test, X_train, y_train)
            self.model_name = "Fallback-Hurdle"
        else:
            self.model_name = "Fallback-SARIMAX"

        return y_pred, conf_low, conf_up, self.model_name
