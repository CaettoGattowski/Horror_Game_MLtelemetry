import numpy as np
from models.kernels import linear_kernel, polynomial_kernel

X = np.array([
    [1, 2, 3],
    [2, 0, 1]
])

Z = np.array([
    [4, 5, 6],
    [1, 1, 1]
])

# Test 1: Degree should match the linear kernel
def test_degree_one_matches_linear():
    result = polynomial_kernel(X, Z, gamma=1, degree=1, coef0=0)
    expected = linear_kernel(X, Z)

    assert np.allclose(result, expected)
    print("Test 1 for polynomial kernel passed")

# Test 2: Degree 2
def test_degree_two():
    result = polynomial_kernel(X, Z, gamma=1, degree=2, coef0=1)

    expected = np.array([
        [1089, 49],
        [225, 16]
    ])

    assert np.allclose(result, expected)
    print("Test 2 for polynomial kernel passed")

# Test 3: Degree 3
def test_degree_three():
    result = polynomial_kernel(X, Z, gamma=1, degree=3, coef0=1)
    
    expected = np.array([
        [35937, 343],
        [3375, 64]
    ])

    assert np.allclose(result, expected)
    print("Test 3 for polynomial kernel passed")

# Test 4: Symmetry
def test_symmetry():
    result = polynomial_kernel(X, X, gamma=1, degree=2, coef0=1)

    assert np.allclose(result, result.T)
    print("Test 4 for polynomial kernel passed")

# Test 5: Output Shape
def test_output_shape():
    result = polynomial_kernel(X, Z, gamma=1, degree=2, coef0=1)

    assert result.shape == (2,2)
    print("Test 5 for polynomial kernel passed")

if __name__ == "__main__":
    test_degree_one_matches_linear()
    test_degree_two()
    test_degree_three()
    test_symmetry()
    test_output_shape()