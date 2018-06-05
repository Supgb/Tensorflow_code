from datetime import datetime
from models.Plain_tf_implement_dnn_clf import shuffle_batch
import tensorflow as tf


# Loading training set and reshape it to (-1, 784)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train.reshape(-1, 784), X_test.reshape(-1, 784)

n_inputs = 28 * 28
n_hiddens1 = 300
n_hiddens2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

# Batch normalization params
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None,
}

# Training params
n_epochs = 40
batch_size = 50
learning_rate = 0.01

# Save and log dir
checkpoints_dir = 'tf_tmp_checkpoints/'
root_log_dir = 'tf_log'
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)

# Draw the DNN graph
with tf.name_scope('dnn_batch_normalization'):
    with tf.contrib.framework.arg_scope(
        [tf.contrib.layers.fully_connected],
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params=bn_params,
    ):
        hidden_1 = tf.contrib.layers.fully_connected(X, n_hiddens1, activation_fn=tf.nn.relu, scope='hidden1')
        hidden_2 = tf.contrib.layers.fully_connected(hidden_1, n_hiddens2, activation_fn=tf.nn.relu, scope='hidden2')
        logits = tf.contrib.layers.fully_connected(hidden_2, n_outputs, activation_fn=None, scope='output')

# Define the loss
# Compute the cross entropy with the output that before go through the softmax function
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits,
    )
    loss = tf.reduce_mean(cross_entropy, name='loss')

# Define the traininig operation
# Using the Gradient Descent.
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Define the evaluation operation
# Using the in_top_k() function,
# which can compute whether or not the highest score of logits
# is corresponding to the target y.
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

# Initializer and saver for the model
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution phase
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={
                is_training: True,
                X: X_batch,
                y: y_batch
            })
        batch_accuracy = accuracy.eval(feed_dict={
            is_training: True,
            X: X_batch,
            y: y_batch
        })
        val_accuracy = accuracy.eval(feed_dict={
            is_training: False,
            X: X_test,
            y: y_test
        })
        batch_loss = loss.eval(feed_dict={
            is_training: True,
            X: X_batch,
            y: y_batch
        })
        print(epoch, 'batch accuracy: ', batch_accuracy, 'test accuracy: ', val_accuracy,
              'loss: ', batch_loss)
    saver.save(sess, checkpoints_dir+'batch_normalization_final.ckpt')

file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
file_writer.close()
