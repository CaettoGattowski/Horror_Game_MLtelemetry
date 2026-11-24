import numpy as np
from models.kernels import gaussian_kernel

X = np.array([
    [1, 2, 3],
    [2, 1, 0]
])

Z = np.array([
    [4, 5, 6],
    [1, 0, 1]
])

# Test 1: Diagonal elements should equal to 1
def test_diagonal_equals_one():
    result = gaussian_kernel(X, X)

    expected = np.array([1.0, 1.0])

    assert np.allclose(np.diag(result), expected)
    print("Test 1 for gaussian kernel passed")

# Test 2: Symmetry
def test_symmetry():
    result = gaussian_kernel(X, X)

    assert np.allclose(result, result.T)
    print("Test 2 for gaussian kernel passed")

# Test 3: Known Values for confirmation manually
def test_known_values():
    result = gaussian_kernel(X, Z, gamma=1.0)

    # Χ1 to Z1 (1-4)^2 + (2-5)^2 + (3-6)^2 = 27
    # X1 to Z2 (1-1)^2 + (2-0)^2 + (3-1)^2 = 8
    # X2 to Z1 (2-4)^2 + (1-5)^2 + (0-6)^2 = 56
    # X2 to Z2 (2-1)^2 + (1-0)^2 + (0-1)^2 = 3
    expected = np.array([
        [np.exp(-27), np.exp(-8)],
        [np.exp(-56), np.exp(-3)]
    ])
    assert np.allclose(result, expected)
    print("Test 3 for gaussian kernel passed")

# Test 4: Output Shape
def test_output_shape():
    Z_test = np.array([
        [4, 5, 6],
        [1, 0, 1],
        [2, 3, 4]
    ])
    result = gaussian_kernel(X, Z_test)

    assert result.shape == (2,3)
    print("Test 4 for gaussian kernel passed")

# Test 5: Range
def test_range():
    result = gaussian_kernel(X, Z)

    assert np.all(result > 0) and np.all(result <=1)
    print("Test 5 for gaussian kernel passed")

if __name__ == "__main__":
    test_diagonal_equals_one()
    test_symmetry()
    test_known_values()
    test_output_shape()
    test_range()

