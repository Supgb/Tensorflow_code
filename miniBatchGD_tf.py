import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from datetime import datetime


checkpoints_dir = 'tf_tmp_checkpoints/'
root_log_dir = 'tf_log'
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing.data.astype(np.float32))
housing_bias_scaled = np.c_[np.ones((m, 1)), housing_scaled]
target = housing.target.reshape(-1, 1)

X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')


def fetch_batch(epoch, batch_index, batch_size):
    start = batch_index * batch_size
    if batch_index == 206:
        X_batch = housing_bias_scaled[start:, :]
        y_batch = housing.target.reshape(-1, 1)[start:, :]
    else:
        X_batch = housing_bias_scaled[start:start + batch_size, :]
        y_batch = target[start:start + batch_size, :]
    return X_batch, y_batch


n_epochs = 50
batch_size = 100
n_batches = int(np.ceil(m/batch_size))
eta = 0.01

theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1), name='theta')
prediction = tf.matmul(X, theta, name='prediction')

with tf.name_scope('loss') as scope:
    error = prediction - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, 'tf_tmp_checkpoints/mini_BatchGD.ckpt')
    for epoch in range(n_epochs):

        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    save_path = saver.save(sess, checkpoints_dir)





