import numpy as np

import imp
amat = imp.load_source('amat', '../amat.py')
from amat import AMat
# load data
train_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all
test_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_test.amat').all
# note: last entry is label
x_train, y_train = train_data[:, :-1], train_data[:, -1:]
x_test, y_test = test_data[:, :-1], test_data[:, -1:]

def augment_label(y_in):
    new_y_train = np.zeros((y_in.shape[0], 10))
    for i, y in enumerate(new_y_train):
        label = y_in[i][0]
        new_y_train[i][label] = 1
    return new_y_train


y_train = augment_label(y_train)
y_test = augment_label(y_test)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())
# saver = tf.train.Saver({'W': W, 'b':b})
# saver.restore(sess, 'beginner.ckpt')

y = tf.nn.softmax(tf.matmul(x,W) + b)  # is this a NN?
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for j in range(100):
    for i in range(240):
      batch = (x_train[i * 50: (i + 1) * 50],
               y_train[i * 50: (i + 1) * 50])
      # batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
# save_path = saver.save(sess, "beginner.ckpt")
