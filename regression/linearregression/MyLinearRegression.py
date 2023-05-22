import numpy as np

print("Check reloaded?", np.random.rand(1)[0])


class LinearRegression:
    def __init__(self, learn_rate=0.01, num_iterations=50, seed=None):
        self.lr = learn_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

        np.random.seed(seed)  # set seed

    def predict(self, X: np.ndarray):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        num_samples, num_features = X.shape

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        self.weights = np.random.rand(num_features)
        self.bias = np.random.rand(1)

        self.total_loss = 0

        for i in range(self.num_iterations):  # sourcery skip: for-index-underscore
            y_pred = self.predict(X)

            cost = 1 / (2 * num_samples) * np.sum((y - y_pred) ** 2)
            self.total_loss += cost

            dW = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)

            # print(
            #     f"Iteration {i}\ncost={cost}\ndb={db}\ndW={dW}\ny_pred={y_pred}",
            #     "self.lr * dW",
            #     self.lr * dW,
            #     y_pred - y,
            #     X.T,
            #     self.weights,
            #     sep="\n",
            #     end="\n\n",
            # )

            self.weights -= self.lr * dW
            self.bias -= self.lr * db

        self.total_loss /= self.num_iterations
