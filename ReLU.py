import tensorflow as tf
import numpy as np
from datetime import datetime


root_log_dir = 'tf_log'
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)


def relu(X):
    with tf.variable_scope('relu', reuse=True):
        threshold = tf.get_variable('threshold')
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, threshold, name='max')


n_features = 3
with tf.variable_scope('relu'):
    threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))

X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')

# merge_summary_op = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())







