import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()
m, n = housing.data.shape
housing_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
error = tf.matmul(X, theta) - y
mse = tf.reduce_mean(tf.square(error), name='mse')

with tf.Session() as sess:
    theta_value = theta.eval()
    print(mse.eval())
