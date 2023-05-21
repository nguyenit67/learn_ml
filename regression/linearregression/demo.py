import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iterations=50, seed=None) -> None:
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

        # set seed
        np.random.seed(seed)

    def fit(self, X, y):
        n_samples, num_features = X.shape

        self.weights = np.random.rand(num_features)
        self.bias = np.random.rand(1)

        self.total_loss = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            cost = 1 / (2 * n_samples) * np.sum((y - y_pred) ** 2)
            self.total_loss += cost(self.weights)

            dW = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)