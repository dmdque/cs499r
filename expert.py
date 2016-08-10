import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from amat import AMat

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def transform_input(M, x):
    v = np.zeros(x.shape)
    print x.shape
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            c = np.matrix('{}; {}; 1'.format(i, j))
            ct = M * c
            # TODO: add interpolation
            ct = ct.astype(int)
            if ct.item(0) < x.shape[0] and ct.item(1) < x.shape[1]:
                print ct.item(0), ct.item(1)
                v[i][j] = x.item(ct.item(0), ct.item(1))
    return v


def format_label(label):
    labels = np.zeros(10)
    labels[int(label[0])] = 1
    return np.array(labels)


# load data
train_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all[:100]
test_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_test.amat').all[:100]

# note: last entry is label
x_train, y_train = train_data[:, :-1], train_data[:, -1:]
x_test, y_test = test_data[:, :-1], test_data[:, -1:]
y_train = np.array(map(format_label, y_train))
y_test = np.array(map(format_label, y_test))

# # reshape
# x_train = x_train.reshape(x_train.shape[0], 1, DIM, DIM)
# x_test = x_test.reshape(x_test.shape[0], 1, DIM, DIM)

# input_shape = x_train.shape[1:]  # should be (28, 28)



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

saver = tf.train.Saver()
saver.restore(sess, "/Users/danielque/git/cs499r/ex_model.ckpt")
# sess.run(tf.initialize_all_variables())

# x_prime = tf.py_func(transform_input, [theta, x], 
# x_prime = tf.matmul(theta, x)
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# for i in range(1000):
  # batch = mnist.train.next_batch(50)
  # train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# batch = (x_train[:100], y_train[:100])
# train_step.run(feed_dict={x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Finally, we can evaluate our accuracy on the test data. This should be about 91% correct.
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
# save_path = saver.save(sess, "/Users/danielque/git/cs499r/ex_model.ckpt")

