import numpy as np


class TSBModel:
    """Teunter-Syntetos-Babai model for intermittent demand forecasting."""

    def __init__(self, alpha=0.2, beta=0.1):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.p_t = None
        self.z_t = None
        self.resid_sigma = 0.0
        self.upper_cap = None

    @staticmethod
    def _mae(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    def _fit_single(self, y, alpha, beta):
        y = np.asarray(y, dtype=float)
        o = (y > 0).astype(float)

        p = float(o[0]) if len(o) else 0.0
        if np.any(o > 0):
            z = float(np.mean(y[o > 0]))
        else:
            z = 0.0

        fitted = np.zeros(len(y), dtype=float)
        for t in range(len(y)):
            fitted[t] = max(p * z, 0.0)
            p = alpha * o[t] + (1.0 - alpha) * p
            if o[t] > 0:
                z = beta * y[t] + (1.0 - beta) * z

        return fitted, p, z

    def fit(self, y_train):
        y = np.asarray(y_train, dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, 0.0, None)

        if len(y) == 0:
            self.p_t = 0.0
            self.z_t = 0.0
            self.resid_sigma = 0.0
            self.upper_cap = 1.0
            return self

        train_mean = float(np.mean(y))
        p99 = float(np.percentile(y, 99))
        self.upper_cap = max(1.0, 2.0 * p99, 6.0 * train_mean)

        # Small grid search on smoothing factors using holdout MAE.
        n = len(y)
        if n >= 120:
            val_size = min(max(30, int(n * 0.12)), n - 60)
            split = n - val_size
            y_sub = y[:split]
            y_val = y[split:]
        else:
            y_sub = y
            y_val = None

        alpha_grid = [0.05, 0.10, 0.20, 0.30]
        beta_grid = [0.05, 0.10, 0.20, 0.30]

        best = None
        best_score = np.inf

        for alpha in alpha_grid:
            for beta in beta_grid:
                fitted, p, z = self._fit_single(y_sub, alpha, beta)
                if y_val is None:
                    score = self._mae(y_sub, fitted)
                else:
                    y_hat = np.full(len(y_val), max(p * z, 0.0), dtype=float)
                    score = self._mae(y_val, y_hat)

                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best = (alpha, beta)

        if best is None:
            best = (self.alpha, self.beta)

        self.alpha, self.beta = float(best[0]), float(best[1])

        fitted_full, self.p_t, self.z_t = self._fit_single(y, self.alpha, self.beta)
        resid = y - fitted_full
        self.resid_sigma = float(np.std(resid)) if len(resid) else 0.0
        return self

    def forecast(self, steps):
        steps = int(steps)
        mean_fcst = max(float(self.p_t) * float(self.z_t), 0.0)

        y_pred = np.full(steps, mean_fcst, dtype=float)
        conf_low = np.maximum(y_pred - 1.96 * self.resid_sigma, 0.0)
        conf_up = y_pred + 1.96 * self.resid_sigma

        if self.upper_cap is not None:
            y_pred = np.clip(y_pred, 0.0, self.upper_cap)
            conf_low = np.clip(conf_low, 0.0, self.upper_cap)
            conf_up = np.clip(conf_up, 0.0, max(self.upper_cap * 1.2, 1.0))

        return y_pred, conf_low, conf_up
