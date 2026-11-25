import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

RANDOM_STATE = 42


def default_base_learner():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


class TLearner:
    """Reference model from UP01: separate arm models, uplift = mu1 - mu0."""

    def __init__(self, base_learner=None):
        self.base_learner = base_learner if base_learner is not None else default_base_learner()

    def fit(self, X, treatment, outcome):
        X = np.asarray(X)
        t = np.asarray(treatment).astype(int)
        y = np.asarray(outcome).astype(int)
        self.model_control_ = clone(self.base_learner).fit(X[t == 0], y[t == 0])
        self.model_treated_ = clone(self.base_learner).fit(X[t == 1], y[t == 1])
        return self

    def predict_uplift(self, X):
        X = np.asarray(X)
        return self.model_treated_.predict_proba(X)[:, 1] - self.model_control_.predict_proba(X)[:, 1]
