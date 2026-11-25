import numpy as np
import pytest

from models.kernels import linear_kernel, polynomial_kernel, gaussian_kernel, combined_kernel

X_valid = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

Z_valid = np.array([
    [5.0, 6.0],
    [7.0, 8.0]
])
# input validation tests
KERNELS =[ linear_kernel, polynomial_kernel, gaussian_kernel, combined_kernel]

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_1d_input(kernel):
    X_invalid = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_3d_input(kernel):
    X_invalid = np.ones((2, 3, 4))

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_mismatched_features(kernel):
    X_invalid = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_empty_samples(kernel):
    X_invalid = np.empty((0, 2))

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_zero_features(kernel):
    X_invalid = np.empty((2, 0))
    Z_invalid = np.empty((3, 0))

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_invalid)

@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_reject_non_finite_values(kernel, invalid_value):
    X_invalid = X_valid.copy()
    X_invalid[0, 0] = invalid_value

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

@pytest.mark.parametrize("kernel", KERNELS)
def test_reject_non_numeric_input(kernel):
    X_invalid = np.array([
        ["hello", "world"],
        ["one", "two"]
    ])

    with pytest.raises(ValueError):
        kernel(X_invalid, Z_valid)

# gamma validation tests
GAMMA_KERNELS = [polynomial_kernel, gaussian_kernel, combined_kernel]

@pytest.mark.parametrize("kernel", GAMMA_KERNELS)
@pytest.mark.parametrize("invalid_gamma", [ 0, -1, np.nan, np.inf, -np.inf, "invalid", True ])
def test_reject_invalid_gamma(kernel, invalid_gamma):
    with pytest.raises(ValueError):
        kernel(X_valid, Z_valid, gamma=invalid_gamma)

# degree validation tests
DEGREE_KERNELS = [polynomial_kernel, combined_kernel]

@pytest.mark.parametrize("kernel", DEGREE_KERNELS)
@pytest.mark.parametrize("invalid_degree", [ 0, -1, 2.5, True, "3"])
def test_reject_invalid_degree(kernel, invalid_degree):
    with pytest.raises(ValueError):
        kernel(X_valid, Z_valid, degree=invalid_degree)

# coef0 validation tests
COEF0_KERNELS = [polynomial_kernel, combined_kernel]

@pytest.mark.parametrize("kernel", COEF0_KERNELS)
@pytest.mark.parametrize("invalid_coef0", [ np.nan, np.inf, -np.inf, "invalid", True])
def test_reject_invalid_coef0(kernel, invalid_coef0):
    with pytest.raises(ValueError):
        kernel(X_valid, Z_valid, coef0=invalid_coef0)

# aplha validation tests
@pytest.mark.parametrize("invalid_alpha", [ -0.1, 1.1, np.nan, np.inf, -np.inf, "invalid", True])
def test_reject_invalid_alpha(invalid_alpha):
    with pytest.raises(ValueError):
        combined_kernel(X_valid, Z_valid, alpha=invalid_alpha)

