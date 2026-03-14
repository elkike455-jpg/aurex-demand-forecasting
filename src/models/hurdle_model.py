import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HurdleModel:
    def __init__(self):
        self.occurrence_model = None
        self.occurrence_constant_prob = None
        self.quantity_model = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        z_train = (y_train > 0).astype(int)
        classes = np.unique(z_train)

        if len(classes) < 2:
            # Degenerate case: all observations are zero or all are non-zero.
            # Use constant occurrence probability instead of logistic regression.
            self.occurrence_model = None
            self.occurrence_constant_prob = float(classes[0])
        else:
            self.occurrence_model = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
            self.occurrence_model.fit(X_train, z_train)
            self.occurrence_constant_prob = None

        nz_idx = y_train > 0
        if nz_idx.sum() >= 10:
            y_size = np.log1p(y_train[nz_idx])

            self.quantity_model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            self.quantity_model.fit(X_train[nz_idx], y_size)

        return self

    def forecast(self, X_test, X_train, y_train):
        X_test = np.asarray(X_test, dtype=float)
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if self.occurrence_model is None:
            p_test = np.full(len(X_test), self.occurrence_constant_prob, dtype=float)
            p_train = np.full(len(X_train), self.occurrence_constant_prob, dtype=float)
        else:
            p_test = self.occurrence_model.predict_proba(X_test)[:, 1]
            p_train = self.occurrence_model.predict_proba(X_train)[:, 1]

        if self.quantity_model is None:
            base = float(np.mean(y_train[y_train > 0])) if np.any(y_train > 0) else 0.0
            size_test = np.full(len(X_test), base, dtype=float)
            size_train = np.full(len(X_train), base, dtype=float)
        else:
            size_test = np.expm1(self.quantity_model.predict(X_test))
            size_train = np.expm1(self.quantity_model.predict(X_train))

        y_pred = np.maximum(p_test * size_test, 0.0)

        resid = y_train - (p_train * size_train)
        sigma = float(np.std(resid)) if len(resid) else 0.0

        conf_low = np.maximum(y_pred - 1.96 * sigma, 0.0)
        conf_up = y_pred + 1.96 * sigma

        return y_pred, conf_low, conf_up
