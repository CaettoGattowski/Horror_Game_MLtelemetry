import numpy as np
from models.kernels import polynomial_kernel, gaussian_kernel, combined_kernel

X = np.array([
    [1, 2, 3],
    [2, 1, 0]
])

Z = np.array([
    [4, 5, 6],
    [1, 0, 1]
])

# Test 1: alpha = 1 should match Gaussian Kernel
def test_alpha_one_matches_gaussian():
    result = combined_kernel(X, Z, alpha=1.0)
    expected = gaussian_kernel(X, Z)

    assert np.allclose(result, expected)
    print("Test 1 for combined kernel passed")

# Test 2: aplha = 0 should match Polynomial Kernel
def test_alpha_zero_matches_polynomial():
    result = combined_kernel(X, Z, alpha=0.0)
    expected = polynomial_kernel(X, Z)

    assert np.allclose(result, expected)
    print("Test 2 for combined kernel passed")

# Test 3: aplha = 0.5 should give equal contribution
def test_alpha_half():
    result = combined_kernel(X, Z, alpha=0.5)
    expected = (0.5 * gaussian_kernel(X, Z) + 0.5 * polynomial_kernel(X, Z))

    assert np.allclose(result, expected)
    print("Test 3 for combined kernel passed")

# Test 4: Output Shape
def test_output_shape():
    Z_test = np.array([
        [4, 5, 6],
        [1, 0, 1],
        [2, 3, 4]
    ])

    result = combined_kernel(X, Z_test)
    
    assert result.shape == (2, 3)
    print("Test 4 for combined kernel passed")

# Test 5: Symmetry
def test_symmetry():
    result = combined_kernel(X, X)
    
    assert np.allclose(result, result.T)
    print("Test 5 for combined kernel passed")

if __name__ == "__main__":
    test_alpha_one_matches_gaussian()
    test_alpha_zero_matches_polynomial()
    test_alpha_half()
    test_output_shape()
    test_symmetry()
