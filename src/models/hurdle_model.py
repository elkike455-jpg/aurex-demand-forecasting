import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class HurdleModel:
    def __init__(self):
        self.occurrence_model = None
        self.quantity_model = None

    def fit(self, X_train, y_train):
        z_train = (y_train > 0).astype(int)

        self.occurrence_model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])
        self.occurrence_model.fit(X_train, z_train)

        nz_idx = (y_train > 0)
        if nz_idx.sum() >= 10:
            y_size = np.log1p(y_train[nz_idx])

            self.quantity_model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0))
            ])
            self.quantity_model.fit(X_train[nz_idx], y_size)

        return self

    def forecast(self, X_test, X_train, y_train):
        p_test = self.occurrence_model.predict_proba(X_test)[:, 1]
        p_train = self.occurrence_model.predict_proba(X_train)[:, 1]

        if self.quantity_model is None:
            base = float(np.mean(y_train[y_train > 0])) if np.any(y_train > 0) else 0.0
            size_test = np.full(len(X_test), base, dtype=float)
            size_train = np.full(len(X_train), base, dtype=float)
        else:
            size_test = np.expm1(self.quantity_model.predict(X_test))
            size_train = np.expm1(self.quantity_model.predict(X_train))

        y_pred = np.maximum(p_test * size_test, 0)

        resid = y_train - (p_train * size_train)
        sigma = np.std(resid)

        conf_low = np.maximum(y_pred - 1.96 * sigma, 0)
        conf_up = y_pred + 1.96 * sigma

        return y_pred, conf_low, conf_up
