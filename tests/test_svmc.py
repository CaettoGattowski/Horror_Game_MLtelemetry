import numpy as np
import pytest
from sklearn.datasets import make_classification
from models.svmc import SVCmodel

@pytest.mark.parametrize("kernel", ["linear", "polynomial", "rbf", "combined"])
def test_SVCmodel(kernel):
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=42)

    svcmodel = SVCmodel(kernel="rbf", C=1.0, gamma=0.1)

    svcmodel.fit(X, y)
    
    predictions = svcmodel.predict(X)

    accuracy= svcmodel.score(X, y)

    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, np.unique(y)))
    assert 0.0 <= accuracy <= 1.0