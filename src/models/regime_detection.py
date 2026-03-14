import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class MarkovRegimeDetector:
    """Fits a low-dimensional Markov switching model to infer latent demand regimes."""

    def __init__(self, k_regimes=2):
        """Initialize detector with number of latent regimes."""
        self.k_regimes = k_regimes
        self.model = None
        self.results = None

    def fit(self, y_train_log):
        """Train Markov regression on a log-transformed target series."""
        y = np.asarray(y_train_log, dtype=float)
        y = y[np.isfinite(y)]

        # Guardrails: Markov switching needs enough variation and sample size.
        if len(y) < 100:
            raise ValueError("Series too short for Markov regime detection.")
        if np.std(y) < 1e-8:
            raise ValueError("Series almost constant; Markov not identifiable.")

        self.model = MarkovRegression(
            endog=y,
            k_regimes=self.k_regimes,
            switching_variance=True,
            trend="c",
        )
        self.results = self.model.fit(disp=False, maxiter=200)
        return self

    def get_regime_probabilities(self):
        """Return smoothed probabilities and hard labels (argmax over regimes)."""
        probs = np.asarray(self.results.smoothed_marginal_probabilities)
        regimes = np.argmax(probs, axis=1)
        return probs, regimes

    def get_transition_matrix(self):
        """Return estimated regime transition matrix P(i -> j)."""
        return self.results.regime_transition
