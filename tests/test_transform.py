import numpy as np
from sklearn.base import BaseEstimator

from uplift_transform.transform import ClassTransformation, transformed_outcome


class RecordingRegressor(BaseEstimator):
    """Records the target it was fit on and predicts the training mean."""

    def fit(self, X, y):
        self.y_ = np.asarray(y)
        self.mean_ = float(np.mean(self.y_))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_transformed_outcome_identity_with_fixed_propensity():
    # p = 0.5: treated responders get 2, control responders get -2, non-responders get 0.
    z = transformed_outcome(y=[1, 1, 0, 1], treatment=[1, 0, 1, 0], p=0.5)
    np.testing.assert_allclose(z, [2, -2, 0, -2])


def test_transformed_outcome_mean_equals_difference_in_arm_means():
    # With p equal to the observed treated share, mean(Z) is exactly mean(Y | T=1) - mean(Y | T=0).
    y = np.array([1, 0, 1, 1, 0])
    t = np.array([1, 1, 0, 0, 0])
    p = t.mean()
    z = transformed_outcome(y, t, p)
    np.testing.assert_allclose(z.mean(), y[t == 1].mean() - y[t == 0].mean())


def test_class_transformation_regresses_on_transformed_target():
    X = np.arange(10).reshape(5, 2)
    t = np.array([1, 1, 0, 0, 0])
    y = np.array([1, 0, 1, 1, 0])

    model = ClassTransformation(base_regressor=RecordingRegressor()).fit(X, t, y)

    np.testing.assert_allclose(model.propensity_, 0.4)
    np.testing.assert_allclose(model.model_.y_, transformed_outcome(y, t, 0.4))
    np.testing.assert_allclose(model.predict_uplift(X), np.full(5, model.model_.mean_))
