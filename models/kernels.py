import numpy as np

def linear_kernel(X, Z):
    # we guarantee all the data are numpy arrays as floats so that we can also accept for example python lists
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # input validation | we want matrices not vectors or scalars | we want the matrices to be of the same number of features so they can be comparable
    if X.ndim != 2 or Z.ndim!=2:
        raise ValueError("Inputs must be 2D arrays")

    if X.shape[1] != Z.shape[1]:
        raise ValueError("X and Z must the same number of features")

    # compute pairwise dot products
    lK = np.matmul(X, Z.T)

    return lK


def polynomial_kernel(X, Z, gamma=1.0, degree=3, coef0=1.0):
    # reuse the linear kernel
    linear_matrix = linear_kernel(X, Z)

    # compute polynomial kernel
    pK = np.power(gamma * linear_matrix + coef0, degree)
    return pK