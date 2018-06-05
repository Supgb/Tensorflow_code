import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime


checkpoints_dir = 'tf_tmp_checkpoints/'
root_log_dir = 'tf_log'
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)

n_epochs = 3000
learning_rate = 0.14

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing.data.astype(np.float32))
housing_bias_scaled = np.c_[np.ones((m, 1)), housing_scaled]

X = tf.constant(housing_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1), name='theta')
error = tf.matmul(X, theta) - y
mse = tf.reduce_mean(tf.square(error), name='mse')
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta-learning_rate*gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            summary_str = mse_summary.eval()
            file_writer.add_summary(summary_str, epoch)
        sess.run(training_op)

    best_theta = theta.eval()
    saver.save(sess, checkpoints_dir)
    file_writer.close()

