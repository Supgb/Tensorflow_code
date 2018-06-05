import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


mnist = datasets.fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
X_train, X_val, y_train, y_val = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(np.float32))
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
y_train = y_train.astype(np.int)
dnn_clf_1 = tf.estimator.DNNClassifier(hidden_units=[300, 100], feature_columns=feature_columns, n_classes=10)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], feature_columns=feature_columns, n_classes=10)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

y_pred = list(dnn_clf.predict(X_val))
accuracy_score(y_val, y_pred)

