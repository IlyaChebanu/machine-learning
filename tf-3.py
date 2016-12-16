import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from scipy.misc import imread
from numpy import linalg

# one_hot => one on, rest off
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# print(mnist.train.images[0])

digit = cv2.imread('six.png', 0).astype('float64')
digit = digit.reshape(784,-1)
# digit = normalize(digit)
digit = digit.flatten()
# digit = np.array([digit, [28, 28]])

# print(digit)
# print(np.shape(digit))
# print(np.shape(mnist.train.images[0]))

n_epochs = 0
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# Matrix => heigh x width
x = tf.placeholder('float', [None, n_chunks, chunk_size]) # Flatten 28x28 to 784
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sesh.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch + 1, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

print(x)
train_neural_network(x)

# predict = recurrent_neural_network(digit)
