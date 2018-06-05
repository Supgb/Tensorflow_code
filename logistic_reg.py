from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_pro = log_reg.predict_proba(X_new)

plt.plot(X_new, y_pro[:, 0], 'r-')
plt.plot(X_new, y_pro[:, 1], 'b--')
plt.legend(['Non-Virginica', 'Virginica'])
plt.show()

# Softmax Regression
X_1 = iris['data'][:, (2, 3)] # petal length, petal width
y_1 = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
softmax_reg.fit(X_1, y_1)
