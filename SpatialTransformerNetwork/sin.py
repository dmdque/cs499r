# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# sess.run(tf.initialize_all_variables())

# y = tf.nn.softmax(tf.matmul(x,W) + b)  # is this a NN?
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# for i in range(1000):
  # batch = mnist.train.next_batch(50)
  # train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
import numpy as np

import imp
amat = imp.load_source('amat', '../amat.py')
from amat import AMat
# load data
train_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all[:20]
test_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_test.amat').all[:20]
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
import ipdb; ipdb.set_trace()



import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# TODO: normalize
# from cs499r/sampler.py
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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)




class SpatiallyInvariantNetwork:
    def __init__(self):
        """Initializes the model and loads variables from file."""
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        # transform = tf.Variable(tf.zeros([2, 3]))
        # transform = tf.Variable(tf.ones([784]))
        # transform = tf.Variable(tf.zeros([784]))
        transform = tf.Variable(tf.truncated_normal([784], stddev=.1))
        W = tf.Variable(tf.zeros([784,10]))
        # W = tf.Print(W, [W], message='W: ')
        b = tf.Variable(tf.zeros([10]))

        # TODO: implement interpolation here
        # transformed_x = tf.map_fn(transform_input, )
        transformed_x = tf.add(x, transform)
        # transformed_x = tf.matmul(x, transform)
        # transformed_x = x

        y = tf.nn.softmax(tf.matmul(transformed_x, W) + b)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                       reduction_indices=[1]))
        # cross_entropy = tf.Print(cross_entropy,
                                 # [cross_entropy],
                                 # message='cross_entropy')

        opt = tf.train.GradientDescentOptimizer(0.5)
        grads_and_vars = opt.compute_gradients(
            cross_entropy,
            var_list=[transform])


        # # # can't print this
        # grads_and_vars = tf.Print(grads_and_vars[0],
                        # [grads_and_vars[0]],
                        # message='grads_and_vars')


        # self.train_step = opt.minimize(cross_entropy)
        self.train_step = opt.apply_gradients(grads_and_vars)
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


    def run(self):
        """Trains the model."""
        print 'train'
        print self.sess.run(self.transform)
        with self.sess.as_default():
            for i in range(1000):
                batch = mnist.train.next_batch(50)
                # batch = train_data
                self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

        # vv = self.sess.run([grad for grad, var in self.grads_and_vars])
        # import ipdb; ipdb.set_trace()


    def evaluate(self):
        """Evaluates the model."""
        y = self.y
        y_ = self.y_
        print 'evaluate'

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels}))
        print self.sess.run(self.transform)

sin = SpatiallyInvariantNetwork()
sin.run()
sin.evaluate()
