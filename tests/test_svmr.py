import numpy as np
import pytest
from sklearn.datasets import make_regression

from models.svmr import SVRmodel

@pytest.mark.parametrize("kernel", ["linear", "polynomial", "rbf", "combined"])
def test_SVRmodel(kernel):
    X, y = make_regression(n_samples=200, n_features=4, n_informative=3, noise=10.0, random_state=42)

    svrmodel = SVRmodel(kernel=kernel, C=1.0, gamma=0.1, degree=3, coef0=1.0, alpha=0.5, epsilon=0.1)

    result = svrmodel.fit(X, y)

    predictions = svrmodel.predict(X)
    r2_score = svrmodel.score(X, y)

    assert result is svrmodel
    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    assert np.isfinite(r2_score)