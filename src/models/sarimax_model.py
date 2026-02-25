import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXModel:
    def __init__(self):
        self.model = None
        self.results = None

    def fit(self, y_train, X_train):
        y_train = np.asarray(y_train, dtype=float)
        X_train = np.asarray(X_train, dtype=float)

        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        cap = np.percentile(y_train, 99.5) if len(y_train) else 0.0
        y_capped = np.clip(y_train, 0.0, cap)
        y_train_log = np.log1p(y_capped)

        self.model = SARIMAX(
            y_train_log,
            exog=X_train,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results = self.model.fit(disp=False)
        return self

    def forecast(self, steps, X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        pred = self.results.get_forecast(steps=steps, exog=X_test)

        y_pred_log = np.clip(np.asarray(pred.predicted_mean), -10, 12)
        y_pred = np.maximum(np.expm1(y_pred_log), 0.0)

        ci = np.asarray(pred.conf_int())
        ci = np.clip(ci, -10, 12)
        conf_low = np.maximum(np.expm1(ci[:, 0]), 0.0)
        conf_up = np.maximum(np.expm1(ci[:, 1]), 0.0)

        return y_pred, conf_low, conf_up
