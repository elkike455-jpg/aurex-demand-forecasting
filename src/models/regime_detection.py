import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class MarkovRegimeDetector:
    def __init__(self, k_regimes=2):
        self.k_regimes = k_regimes
        self.model = None
        self.results = None

    def fit(self, y_train_log):
        y = np.asarray(y_train_log, dtype=float)
        y = y[np.isfinite(y)]

        if len(y) < 100:
            raise ValueError("Serie muy corta para Markov.")
        if np.std(y) < 1e-8:
            raise ValueError("Serie casi constante; Markov no identificable.")

        self.model = MarkovRegression(
            endog=y,
            k_regimes=self.k_regimes,
            switching_variance=True,
            trend="c",
        )
        self.results = self.model.fit(disp=False, maxiter=200)
        return self

    def get_regime_probabilities(self):
        probs = np.asarray(self.results.smoothed_marginal_probabilities)
        regimes = np.argmax(probs, axis=1)
        return probs, regimes

    def get_transition_matrix(self):
        return self.results.regime_transition
