import tensorflow as tf
import numpy as np
from datetime import datetime


def neural_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


def shuffle_batch(X, y, batch_size):
    rnd_index = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_index in np.array_split(rnd_index, n_batches):
        X_batch, y_batch = X[batch_index], y[batch_index]
        yield X_batch, y_batch


if __name__ == '__main__':

    checkpoints_dir = 'tf_tmp_checkpoints/'
    root_log_dir = 'tf_log'
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    log_dir = '{}/run-{}/'.format(root_log_dir, now)

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train.reshape(-1, 784)/255.0, X_test.reshape(-1, 784)/255.0
    n_epochs = 40
    batch_size = 50


    # the number of features
    n_inputs = 28 * 28
    # the number of neurons in each hidden layer
    n_hidden_1 = 300
    n_hidden_2 = 100
    # the number of categories
    n_outputs = 10

    # Data of features and labels
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')


    # Version 1
    # with tf.name_scope('dnn'):
    #     hidden_1 = neural_layer(X, n_hidden_1, 'hidden1', 'relu')
    #     hidden_2 = neural_layer(hidden_1, n_hidden_2, 'hidden2', 'relu')
    #     logits = neural_layer(hidden_2, n_outputs, 'output')


    # Version 2
    # with tf.name_scope('dnn'):
    #     hidden_1 = tf.contrib.layers.fully_connected(X, n_hidden_1, scope='hidden1')
    #     hidden_2 = tf.contrib.layers.fully_connected(hidden_1, n_hidden_2, scope='hidden2')
    #     logits = tf.contrib.layers.fully_connected(hidden_2, n_outputs, scope='output', activation_fn=None)

    # Version 3
    with tf.name_scope('dnn3'):
        hidden_1 = tf.layers.dense(X, n_hidden_1, activation=tf.nn.relu, name='hidden1')
        hidden_2 = tf.layers.dense(hidden_1, n_hidden_2, activation=tf.nn.relu, name='hidden2')
        logits = tf.layers.dense(hidden_2, n_outputs, activation=None, name='output')

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        loss = tf.reduce_mean(cross_entropy, name='loss')

    learning_rate = 0.01

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Batch accuracy: ", accuracy_batch, "Val accuracy: ", accuracy_val)

        saver.save(sess, checkpoints_dir+'my_dnn3_final.ckpt')

    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    file_writer.close()







