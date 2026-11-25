import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42


def transformed_outcome(y, treatment, p):
    """Z = Y*T/p - Y*(1-T)/(1-p). Under randomization with P(T=1)=p, E[Z | X] = tau(X)."""
    if not 0 < p < 1:
        raise ValueError("Propensity p must be strictly between 0 and 1.")
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    return y * t / p - y * (1 - t) / (1 - p)


class ClassTransformation:
    """Single regressor on the transformed outcome; predictions are the uplift estimate."""

    def __init__(self, base_regressor=None, propensity=None):
        self.base_regressor = (
            base_regressor if base_regressor is not None
            else HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        )
        self.propensity = propensity

    def fit(self, X, treatment, outcome):
        t = np.asarray(treatment).astype(int)
        # The randomized treated share is the propensity unless a known value is supplied.
        p = float(self.propensity) if self.propensity is not None else float(t.mean())
        self.propensity_ = p
        z = transformed_outcome(outcome, t, p)
        self.model_ = clone(self.base_regressor).fit(np.asarray(X), z)
        return self

    def predict_uplift(self, X):
        return self.model_.predict(np.asarray(X))
