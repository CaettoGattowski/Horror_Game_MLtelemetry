from sklearn.svm import SVC
from models.kernels import (linear_kernel, polynomial_kernel, gaussian_kernel, combined_kernel)

class SVCmodel:
    def __init__(self, kernel="rbf", C=1.0, gamma=0.1,degree=3, coef0=1.0, alpha=0.5,tol=1e-3, max_iter=1000):

        # store the svm hyperparameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

        # create the svm calssifier
        self.model = SVC( kernel=self.compute_kernel, C=self.C, tol=self.tol, max_iter=self.max_iter)

    def compute_kernel(self, X, Z):

        # select the custom kernel
        if self.kernel == "linear":
            return linear_kernel(X, Z)

        elif self.kernel == "polynomial":
            return polynomial_kernel(X, Z, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        
        elif self.kernel == "rbf":
            return gaussian_kernel(X, Z, gamma=self.gamma)
        
        elif self.kernel == "combined":
            return combined_kernel(X, Z, alpha=self.alpha, gamma=self.gamma, degree=self.degree, coef0=self.coef0)

        else:
            raise ValueError(f"Unknown Kernel: {self.kernel}")
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):

        return self.model.predict(X)

    def score(self, X, y):

        return self.model.score(X, y)

    