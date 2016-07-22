import imp
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

amat = imp.load_source('amat', '../amat.py')
from amat import AMat
spatial_transformer = imp.load_source('spatial_transformer', '../../spatial-transformer-tensorflow/spatial_transformer.py')
from spatial_transformer import transformer


def augment_label(y_in):
    """Converts label from single label to vector label.
    Converts from size (1, 1) to (1, 10).
    eg.
        [4] -> [0, 0, 0, 1, 0, 0, 0, 0, 0]
    """
    new_y_train = np.zeros((y_in.shape[0], 10))
    for i, y in enumerate(new_y_train):
        label = y_in[i][0]
        new_y_train[i][label] = 1
    return new_y_train


def load_data():
    train_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all
    test_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_test.amat').all
    # note: last entry is label
    x_train, y_train = train_data[:, :-1], train_data[:, -1:]
    x_test, y_test = test_data[:, :-1], test_data[:, -1:]
    y_train = augment_label(y_train)
    y_test = augment_label(y_test)
    return x_train, y_train, x_test, y_test


class SpatiallyInvariantNetwork:
    def __init__(self):
        """Initializes the model and loads variables from file."""
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None, 784])
        # note y_ isn't actually used for training
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        transform = tf.Variable(initial_value=np.identity(784), dtype=tf.float32)
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        transformed_x = tf.matmul(x, transform)
        y = tf.nn.softmax(tf.matmul(transformed_x, W) + b)
        # how to do categorical entropy?
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                       reduction_indices=[1]))

        opt = tf.train.GradientDescentOptimizer(10 ** -2)
        grads_and_vars = opt.compute_gradients(cross_entropy,
                                               var_list=[transform])


        self.train_step = opt.minimize(cross_entropy)
        # self.train_step = opt.apply_gradients(grads_and_vars)
        sess.run(tf.initialize_all_variables())

        # load from file
        saver = tf.train.Saver({'W': W, 'b':b})
        saver.restore(sess, 'beginner.ckpt')

        self.x = x
        self.y = y
        self.y_ = y_
        self.sess = sess
        self.grads_and_vars = grads_and_vars
        self.transform = transform


    def run(self, x_train, y_train):
        """Trains the T layer of the model."""
        print 'train'
        print self.sess.run(self.transform)
        with self.sess.as_default():
            for i in range(100):
                self.train_step.run(feed_dict={self.x: [x_train], self.y_: [y_train]})
        print self.sess.run(self.transform)


    def evaluate(self, x_test, y_test):
        """Evaluates the model."""
        y = self.y
        y_ = self.y_
        print 'evaluate'

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={self.x: [x_test], self.y_: [y_test]}))
        print y.eval(feed_dict={self.x: [x_test], self.y_: [y_test]})
        print self.sess.run(self.transform)


# sin = SpatiallyInvariantNetwork()
# # x_train, y_train, x_test, y_test = load_data()
# # ex = (x_test[0], y_test[0])
ex = None
with open('objs.pickle') as f:
    ex = pickle.load(f)

# T = np.diag(np.random.rand(784))
# xt = np.matrix(ex[0]) * T
# plt.imshow(xt.reshape(28, 28), cmap='gray', interpolation='none')
# plt.show()

# # sin.run(ex[0], ex[1])
# sin.evaluate(ex[0], ex[1])


im = ex[0]
im = im.reshape(1, 28, 28, 1)  # why this shape?

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
x = tf.cast(im, 'float32')
# Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):
    initial = np.array([[1, -.6763, 0], [0, 1, 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()

    theta = tf.Variable(initial_value=initial, name='theta')
    h_fc1 = tf.zeros([1, 6]) + theta  # takes advantage of TF's broadcasting
    h_trans = transformer(x, h_fc1, (28, 28))

# Run session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
y = sess.run(h_trans, feed_dict={x: im})

plt.imshow(y[0].reshape(28, 28), cmap='gray', interpolation='none')
plt.show()
