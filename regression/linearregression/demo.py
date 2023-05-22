import numpy as np

from MyLinearRegression import LinearRegression


X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([30, 40, 60])
X_test = np.array([[2, 3, 4], [5, 6, 7]])

my_model = LinearRegression(learn_rate=0.01, num_iterations=10000, seed=7)
my_model.fit(X_train, y_train)
print("Predicted: ", my_model.predict(X_test))
print("Loss: ", my_model.total_loss)

my_model.weights, my_model.bias
