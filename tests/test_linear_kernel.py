import numpy as np
from models.kernels import linear_kernel

X = np.array([
    [1, 2, 3],
    [2, 0, 1]
])

Z = np.array([
    [4, 5, 6],
    [1, 1, 1]
])

# Test 1: Expected Output
K = linear_kernel(X, Z)

expected = np.array([
    [32, 6],
    [14, 3]
])

print("Kernel Matrix:")
print(K)

assert np.allclose(K, expected)
print("Test 1 for linear kernel passed")

# Test 2: Symmestry
K_xx = linear_kernel(X, X)

assert np.allclose(K_xx, K_xx.T)
print("Test 2 for linear kernel passed")

# Test 3: Diagonal
expected_diagonal = np.array([14, 5])

assert np.allclose(np.diag(K_xx), expected_diagonal)
print("Test 3 for linear kernel passed")

# Test 4: Invalid Dimensions
try:
    linear_kernel(np.array([1, 2, 3]), Z)
except ValueError:
    print("Test 4 for linear kernel passed")
else:
    raise AssertionError("Expected ValueError for 1D input")

# Test 5: Mismatched features
try: 
    linear_kernel(X, np.array([[1, 2]]))
except ValueError:
    print("Test 5 for linear kernel passed")
else:
    raise AssertionError("Expected ValueError for mismatched features")