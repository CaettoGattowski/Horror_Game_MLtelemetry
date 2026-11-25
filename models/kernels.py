import numpy as np


def input_validation(X, Z):
    try:
        # guarantees arithmetic numpy array
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Inputs must contain numeric values")

    # ndim != 2 rejects 1D or 3D inputs
    if X.ndim != 2 or Z.ndim !=2:
        raise ValueError("Inputs must be 2D arrays")
    # shape[1] mismatch rejects different number of features
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X and Z must have the same number of features")
    # shape[0] rejects arrays without samples
    if X.shape[0] == 0 or Z.shape[0] == 0:
        raise ValueError("Inputs must contain at least one sample")
    # shape[1] == 0 rejects arrays without features
    if X.shape[1] == 0:
        raise ValueError("Inputs must contain at least one feature")
    # np.isfinite() rejects NaN and +-inf
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Z)):
        raise ValueError("Inputs must contain only finite values")
    # return the validated arrays
    return X, Z
    
def gamma_validation(gamma):
    if isinstance(gamma, (bool, np.bool_)):
        raise ValueError("gamma must be a finite positive number")
    try:
        gamma = float(gamma)
    except (TypeError, ValueError):
        raise ValueError("gamma must be a finite positive number")
    
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive number")
    
    return gamma

def degree_validation(degree):
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree <= 0:
        raise ValueError("degree must be a positive integer") 
    
    return int(degree)

def coef0_validation(coef0):
    if isinstance(coef0, (bool, np.bool_)):
        raise ValueError("coef0 must be a finite number scalar")

    try:
        coef0 = float(coef0)
    except (TypeError, ValueError):
        raise ValueError("coef0 must be a finite numeric scalar")
    
    if not np.isfinite(coef0):
        raise ValueError("coef0 must be a finite numeric scalar")

    return coef0

def alpha_validation(alpha):
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha msut be a finite number between 0 and 1")

    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be a finite number between 0 and 1")

    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1")

    return alpha
def linear_kernel(X, Z):
    
    # use the validated inputs
    X, Z = input_validation(X, Z)

    # input validation | we want matrices not vectors or scalars | we want the matrices to be of the same number of features so they can be comparable
    if X.ndim != 2 or Z.ndim!=2:
        raise ValueError("Inputs must be 2D arrays")

    if X.shape[1] != Z.shape[1]:
        raise ValueError("X and Z must the same number of features")

    # compute pairwise dot products
    lK = np.matmul(X, Z.T)

    return lK


def polynomial_kernel(X, Z, gamma=1.0, degree=3, coef0=1.0):

    # use the validated inputs and parameters
    X, Z = input_validation(X, Z)
    gamma = gamma_validation(gamma)
    degree = degree_validation(degree)
    coef0 = coef0_validation(coef0)

    # reuse the linear matrix
    linear_matrix = linear_kernel(X, Z)

    # compute polynomial kernel
    pK = np.power(gamma * linear_matrix + coef0, degree)
    return pK

def gaussian_kernel(X, Z, gamma=1.0):

    # use the validated inputs and parameters
    X, Z = input_validation(X, Z)
    gamma = gamma_validation(gamma)

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

    # use the validated inputs and parameters
    X, Z = input_validation(X, Z)
    alpha = alpha_validation(alpha)
    gamma = gamma_validation(gamma)
    degree = degree_validation(degree)
    coef0 = coef0_validation(coef0)

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
    