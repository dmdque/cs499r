import imp
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

spatial_transformer = imp.load_source(
    'spatial_transformer',
    '../../spatial-transformer-tensorflow/spatial_transformer.py')
from spatial_transformer import transformer

# Feature flags
DISPLAY_PLOTS = True
RESTRICT_ROTATE = True
THETA_TRAIN_ITERATIONS = 200


def load_data_from_pickle(fname):
    print 'Begin loading data.'
    with open(fname) as f:
        data = pickle.load(f)
        print 'Finished loading data.'
        return data


class SpatiallyInvariantNetwork:
    def __init__(self):
        """Initializes the model and loads variables from file."""
        print 'Begin setup.'
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # note y_ isn't actually used for training
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # Create localisation network and convolutional layer
        with tf.variable_scope('spatial_transformer_0'):
            if RESTRICT_ROTATE:
                initial = 0.0
                theta = tf.Variable(initial_value=initial, name='theta')
                sin = tf.sin(theta)
                cos = tf.cos(theta)
                # rot_matrix = tf.constant([[cos, -sin, 0], [sin, cos, 0]])
                rot_matrix = [cos, -sin, tf.constant(0.0), sin, cos, tf.constant(0.0)]
            else:
                initial = np.array([[1, 0, 0], [0, 1, 0]])
                initial = initial.astype('float32')
                initial = initial.flatten()
                theta = tf.Variable(initial_value=initial, name='theta')
                rot_matrix = tf.identity(theta)

            h_fc1 = tf.zeros([1, 6]) + rot_matrix  # takes advantage of TF's broadcasting
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
        print 'Finished setup.'


    def run(self, x_train, y_train):
        """Trains the T layer of the model."""
        print 'train'
        # print self.sess.run(self.transform)
        # with self.sess.as_default():
            # for i in range(100):
                # self.train_step.run(feed_dict={self.x: [x_train], self.y_: [y_train]})
        # print self.sess.run(self.transform)


    def evaluate(self, x_test, y_test, num):
        """Evaluates the model."""
        x = self.x
        y = self.y
        y_ = self.y_
        h_trans = self.h_trans
        theta = self.theta
        transformed_x = self.transformed_x
        sess = self.sess

        # reset theta before testing again
        if RESTRICT_ROTATE:
            initial = 0.0
        else:
            initial = np.array([[1, 0, 0], [0, 1, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
        assign_op = theta.assign(initial)
        self.sess.run(assign_op)

        # Intermediate values:
        # x_prime

        if DISPLAY_PLOTS:
            orig_y_out = self.sess.run(
                y,
                feed_dict={
                    x: x_test.reshape(1, 28, 28, 1),
                    y_: [y_test],
                })
            orig_pred = np.argmax(orig_y_out)
            orig_confidence = np.max(orig_y_out)

        for i in range(THETA_TRAIN_ITERATIONS):
            x_test = x_test.reshape(1, 28, 28, 1)  # why this shape?

            self.train_step.run(
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })

        if DISPLAY_PLOTS:
            y_out = self.sess.run(
                y,
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })
            prediction = np.argmax(y_out)
            confidence = np.max(y_out)

            index = np.arange(10)
            x_prime = self.sess.run(h_trans, feed_dict={x: x_test})
            # f, axarr = plt.subplots(3, sharey=True)


            ax1 = plt.subplot(221)
            ax1.imshow(x_test.reshape((28, 28)), cmap='gray', interpolation='none')
            ax1.set_title('Original, y={}, y1={}, c1={:5.4f}'.format(np.argmax(y_test), orig_pred, orig_confidence))

            ax2 = plt.subplot(222)
            ax2.imshow(x_prime.reshape((28, 28)), cmap='gray', interpolation='none')
            ax2.set_title('Transformed, yn={}, cn={:5.4f}'.format(prediction, confidence))

            ax3 = plt.subplot(223)
            ax3.cla()
            ax3.bar(index, orig_y_out[0])
            ax3.set_ylim([0, 1])

            ax4 = plt.subplot(224)
            ax4.cla()
            ax4.bar(index, y_out[0])
            ax4.set_ylim([0, 1])
            plt.savefig('figures/sin{}.png'.format(num))
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

def main():
    sin = SpatiallyInvariantNetwork()
    x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')

    num_correct = 0
    num_examples = min(100, 50000)
    for i in range(num_examples):
        print '{} of {}'.format(i, num_examples)
        x_test[i] = x_test[i].reshape((28, 28)).transpose().reshape(784,)  # flip image
        if sin.evaluate(x_test[i], y_test[i], i):
            num_correct += 1

    print 'correct: {} out of {}. {}%'.format(
        num_correct,
        len(x_test),
        float(num_correct) * 100 / num_examples)

if __name__ == '__main__':
    main()


# todo:
    # other paper that shows small changes
        # http://cs231n.stanford.edu/reports2016/119_Report.pdf
    # try another objective?
    # non convex - can't switch class?
    # change number of iterations
    # *flip dataset
    # use more complex main network

# done:
    # graph distributions
    # restrict to rotations
