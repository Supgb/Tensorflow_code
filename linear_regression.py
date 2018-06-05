import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100,1)), X]
#theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
#y_pred = X_new_b.dot(theta_best)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X_new)

# plt.plot(X_new, y_pred, 'r-')
# plt.plot(X, y, 'b.')
# plt.axis([0, 2, 0, 15])
# plt.title('without models')
# plt.show()


"""Batch Gradient Descent"""
eta = 0.1
# n_iterations = 1000
m = 100

# theta = np.random.randn(2, 1)
#
# for iteration in range(n_iterations):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
#     theta = theta - eta*gradients

"""Stochastic Gradient Descent"""
n_epochs = 50
t0, t1 = 5,50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients