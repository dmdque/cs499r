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

        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # note y_ isn't actually used for training
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # Create localisation network and convolutional layer
        with tf.variable_scope('spatial_transformer_0'):
            initial = np.array([[1, 0, 0], [0, 1, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()

            theta = tf.Variable(initial_value=initial, name='theta')
            h_fc1 = tf.zeros([1, 6]) + theta  # takes advantage of TF's broadcasting
            h_trans = transformer(x, h_fc1, (28, 28))

        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        transformed_x = tf.reshape(h_trans, (1, 784))  # forces batch of size 1
        y = tf.nn.softmax(tf.matmul(transformed_x, W) + b)
        # how to do categorical entropy?
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                       reduction_indices=[1]))

        opt = tf.train.GradientDescentOptimizer(10 ** -2)
        grads_and_vars = opt.compute_gradients(cross_entropy,
                                               var_list=[theta])

        # train_step = opt.minimize(cross_entropy, var_list=[theta])
        train_step = opt.apply_gradients(grads_and_vars)
        sess.run(tf.initialize_all_variables())

        # load from file
        saver = tf.train.Saver({'W': W, 'b':b})
        saver.restore(sess, 'beginner.ckpt')

        self.x = x
        self.y = y
        self.y_ = y_
        self.sess = sess
        self.grads_and_vars = grads_and_vars
        self.h_trans = h_trans
        self.transformed_x = transformed_x
        self.train_step = train_step
        self.theta = theta


    def run(self, x_train, y_train):
        """Trains the T layer of the model."""
        print 'train'
        # print self.sess.run(self.transform)
        # with self.sess.as_default():
            # for i in range(100):
                # self.train_step.run(feed_dict={self.x: [x_train], self.y_: [y_train]})
        # print self.sess.run(self.transform)


    def evaluate(self, x_test, y_test):
        """Evaluates the model."""
        x = self.x
        y = self.y
        y_ = self.y_
        h_trans = self.h_trans
        theta = self.theta
        transformed_x = self.transformed_x
        sess = self.sess
        print 'evaluate'

        # reset theta before testing again
        initial = np.array([[1, 0, 0], [0, 1, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        assign_op = theta.assign(initial)
        self.sess.run(assign_op)

        # Intermediate values:
        # x_prime

        for i in range(10):
            x_test = x_test.reshape(1, 28, 28, 1)  # why this shape?

            print 'theta', self.sess.run(theta, feed_dict={x: x_test})
            x_prime = self.sess.run(h_trans, feed_dict={x: x_test})

            x_prime = x_prime.reshape(1, 784)  # this forces batches of size 1
            print 'y', self.sess.run(
                y,
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })

            self.train_step.run(
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })

        prediction = tf.argmax(y,1)
        y_out = self.sess.run(
            prediction,
            feed_dict={
                x: x_test,
                y_: [y_test],
            })

        # x_prime = self.sess.run(h_trans, feed_dict={x: x_test})
        # f, axarr = plt.subplots(2, sharey=True)
        # axarr[0].imshow(x_test.reshape((28, 28)), cmap='gray', interpolation='none')
        # axarr[1].imshow(x_prime.reshape((28, 28)), cmap='gray', interpolation='none')
        # plt.title('Actual value: {}, Predicted value: {}'.format(np.argmax(y_test), y_out))
        # plt.show()

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        is_correct = self.sess.run(
            correct_prediction,
            feed_dict={
                x: x_test,
                y_: [y_test],
            })
        is_correct = is_correct[0]
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(accuracy.eval(feed_dict={self.x: [x_test], self.y_: [y_test]}))
        # print y.eval(feed_dict={self.x: [x_test], self.y_: [y_test]})
        return is_correct


sin = SpatiallyInvariantNetwork()
x_train, y_train, x_test, y_test = load_data()
# ex = (x_test[0], y_test[0])
ex = None
with open('objs.pickle') as f:
    ex = pickle.load(f)

# T = np.diag(np.random.rand(784))
# xt = np.matrix(ex[0]) * T
# plt.imshow(xt.reshape(28, 28), cmap='gray', interpolation='none')
# plt.show()

# plt.imshow(ex[0].reshape(28, 28), cmap='gray', interpolation='none')
# plt.show()

# # sin.run(ex[0], ex[1])
num_correct = 0
for i in range(len(x_test)):
    if sin.evaluate(x_test[i], y_test[i]):
        num_correct += 1

print 'correct: {} out of {}. {}%'.format(
    num_correct,
    len(x_test),
    float(num_correct) / len(x_test))