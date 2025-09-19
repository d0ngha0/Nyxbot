import numpy as np


class RBFRegressor:
    def __init__(self, sigma=None):
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, x, c, s):
        # x: (d,), c: (k, d)
        return np.exp(-np.linalg.norm(x - c, axis=1)**2 / (2 * s**2))

    def _design_matrix(self, X):
        # X: (n_samples, n_features)
        n = X.shape[0]
        k = self.centers.shape[0]
        Phi = np.zeros((n, k))
        for i in range(n):
            Phi[i, :] = self._rbf(X[i], self.centers, self.sigma)
        return Phi

    def train(self, X, Y):
        """
        Train the RBF regressor.
        Parameters:
            X: (n_samples, n_features) - input data
            Y: (n_samples,) or (n_samples, output_dim) - target values
        """
        self.centers = X
        k = X.shape[0]
        if self.sigma is None:
            self.sigma = (np.max(X) - np.min(X)) / np.sqrt(2 * k)

        Phi = self._design_matrix(X)
        self.weights = np.linalg.pinv(Phi).dot(Y)

    def predict(self, X):
        """
        Predict output for new inputs.
        Parameters:
            X: (n_samples, n_features) - input data
        Returns:
            Y_pred: predicted values
        """
        Phi = self._design_matrix(X)
        return Phi.dot(self.weights)