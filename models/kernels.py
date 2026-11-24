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
    # reuse the linear matrix
    linear_matrix = linear_kernel(X, Z)

    # compute polynomial kernel
    pK = np.power(gamma * linear_matrix + coef0, degree)
    return pK

def gaussian_kernel(X, Z, gamma=1.0):

    # reuse the linear matrix
    linear_matrix = linear_kernel(X, Z)

    # calculate the squared norms of X, Z
    X_sqnorm = np.sum(np.power(X, 2), axis=1)[:, np.newaxis]
    Z_sqnorm = np.sum(np.power(Z, 2), axis=1)[np.newaxis, :]

    # compute pairwise squared distances
    squared_distances = X_sqnorm + Z_sqnorm - 2 * linear_matrix

    # prevent negative distances due to floating-point precision
    squared_distances = np.maximum(squared_distances, 0)

    # compute gaussian kernel
    gK = np.exp(-gamma * squared_distances)
    return gK

def combined_kernel(X, Z, alpha=0.5, gamma=1.0, degree=3, coef0=1.0):
    # reuse the polynomial kernel matrix
    polynomial_matrix = polynomial_kernel(X, Z, gamma=gamma, degree=degree, coef0=coef0)

    # reuse the gaussian kernel matrix
    gaussian_matrix = gaussian_kernel(X, Z, gamma=gamma)

    # compute the combined composite kernel
    combK = alpha * gaussian_matrix + (1 - alpha) * polynomial_matrix
    # for a=1 we have gaussian
    # for a=0 we have polynomial
    # for a=0.5 we have equal contribution of both

    return combK
    