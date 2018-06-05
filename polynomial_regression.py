import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split


def plot_learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_error, test_error = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_error.append(mean_squared_error(y_train_predict, y_train[:m]))
        test_error.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_error), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_error), 'b-', linewidth=2, label="validation")
    plt.axis([0, 80, 0, 3.0])
    plt.legend(["train", "validation"])
    plt.show()


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(m, 1)

# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)
X_new = np.array([[0], [3]])

polynomial_regression_1 = Pipeline([
    ('poly_features', PolynomialFeatures(degree=30, include_bias=False)),
    ('scaler', StandardScaler()),
    ('sgd_reg', Ridge(alpha=0, solver='cholesky')),
])
polynomial_regression_1.fit(X, y)
y_pred_poly_1 = polynomial_regression_1.predict(X_new)

polynomial_regression_2 = Pipeline([
    ('poly_features', PolynomialFeatures(degree=30, include_bias=False)),
    ('scaler', StandardScaler()),
    ('sgd_reg', Ridge(alpha=10, solver='cholesky')),
])
polynomial_regression_2.fit(X, y)
y_pred_poly_2 = polynomial_regression_2.predict(X_new)

polynomial_regression_3 = Pipeline([
    ('poly_features', PolynomialFeatures(degree=30, include_bias=False)),
    ('scaler', StandardScaler()),
    ('sgd_reg', Ridge(alpha=100, solver='cholesky')),
])
polynomial_regression_3.fit(X, y)
y_pred_poly_3 = polynomial_regression_3.predict(X_new)


lin_reg_1 = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_reg', Ridge(alpha=0, solver='cholesky'))
])
lin_reg_1.fit(X, y)
y_pred_1 = lin_reg_1.predict(X_new)

lin_reg_2 = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_reg', Ridge(alpha=10, solver='cholesky'))
])
lin_reg_2.fit(X, y)
y_pred_2 = lin_reg_2.predict(X_new)

lin_reg_3 = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_reg', Ridge(alpha=100, solver='cholesky'))
])
lin_reg_3.fit(X, y)
y_pred_3 = lin_reg_3.predict(X_new)

plt.plot(X, y, 'b.')
# plt.plot(X_new, y_pred_1, 'g-', label='alpha=0')
# plt.plot(X_new, y_pred_2, 'r--', label='alpha=0.5')
# plt.plot(X_new, y_pred_3, 'y-+', label='alpha=100')
# plt.axis([0,3, 0, 10])
# plt.legend(['', 'alpha=0', 'alpha=0.5', 'alpha=100'])
# plt.show()


plt.axis([0,3,0,10])
plt.legend(['dataset', 'alpha=0', 'alpha=0.5', 'alpha=100'])
plt.show()




# plot_learning_curve(polynomial_regression, X, y)
# plot_learning_curve(lin_reg, X, y)

# lin_reg.fit(X_poly, y)

# curve fitting ------- START
# C2 = lin_reg.intercept_
# B2 = lin_reg.coef_[0, 0]
# A2 = lin_reg.coef_[0, 1]
# x2 = np.arange(-3, 3, 0.01)
# y2 = A2*x2*x2 + B2*x2 + C2
#
# plt.plot(x2, y2, 'r')
# plt.plot(X, y, 'b.')
# plt.show()
# curve fitting ------- END


