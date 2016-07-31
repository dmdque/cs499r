from datetime import datetime
import imp
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import load_data_from_pickle
from models import beginner
from models import sin_lenet as lenet

spatial_transformer = imp.load_source(
    'spatial_transformer',
    '../../spatial-transformer-tensorflow/spatial_transformer.py')
from spatial_transformer import transformer

# Feature flags
DISPLAY_PLOTS = True
RESTRICT_ROTATE = True
MAX_THETA_TRAIN_ITERATIONS = 1000
MODEL = 'BEGINNER'
TOLERANCE = 10 ** -5
NUM_EXAMPLES = min(100, 50000)

if MODEL == 'LENET':
    MODEL_CKPT = 'lenet-weak.ckpt'
elif MODEL == 'BEGINNER':
    MODEL_CKPT = 'beginner.ckpt'


def model_lenet():
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    if RESTRICT_ROTATE:
        initial = 0.0
        theta = tf.Variable(initial_value=initial, name='theta')
        sin = tf.sin(theta)
        cos = tf.cos(theta)
        rot_matrix = [cos, -sin, tf.constant(0.0),
                      sin, cos, tf.constant(0.0)]
    else:
        initial = np.array([[1, 0, 0], [0, 1, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        theta = tf.Variable(initial_value=initial, name='theta')
        rot_matrix = tf.identity(theta)
    h_fc1 = tf.zeros([1, 6]) + rot_matrix  # takes advantage of TF's broadcast
    h_trans = transformer(x, h_fc1, (28, 28))
    transformed_x = tf.reshape(h_trans, (1, 784))  # forces batch of size 1
    if MODEL == 'LENET':
        net, model_var_dict = lenet(transformed_x)
    elif MODEL == 'BEGINNER':
        W, b, net, model_var_dict = beginner(transformed_x)
    y = tf.nn.softmax(net)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    opt = tf.train.GradientDescentOptimizer(10 ** -2)
    train_step = opt.minimize(cross_entropy,
                              var_list=[theta])
    return x, y_, theta, rot_matrix, h_fc1, h_trans, transformed_x, net, model_var_dict, y, cross_entropy, train_step


class SpatiallyInvariantNetwork:
    def __init__(self):
        """Initializes the model and loads variables from file."""
        print 'Begin setup.'
        x, y_, theta, rot_matrix, h_fc1, h_trans, transformed_x, net, model_var_dict, y, cross_entropy, train_step = model_lenet()
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(model_var_dict)
        saver.restore(sess, MODEL_CKPT)
        self.cross_entropy = cross_entropy
        self.h_trans = h_trans
        self.sess = sess
        self.theta = theta
        self.train_step = train_step
        self.transformed_x = transformed_x
        self.x = x
        self.y = y
        self.y_ = y_
        print 'Finished setup.'


    def run(self, x_train, y_train):
        """Trains the T layer of the model."""
        pass


    def evaluate(self, x_test, y_test, num):
        """Evaluates the model."""
        x = self.x
        y = self.y
        y_ = self.y_
        h_trans = self.h_trans
        theta = self.theta
        transformed_x = self.transformed_x
        sess = self.sess
        cross_entropy = self.cross_entropy
        train_step = self.train_step

        # reset theta before testing again
        if RESTRICT_ROTATE:
            initial = 0.0
        else:
            initial = np.array([[1, 0, 0], [0, 1, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
        assign_op = theta.assign(initial)
        sess.run(assign_op)

        if DISPLAY_PLOTS:
            orig_prediction = sess.run(
                y,
                feed_dict={
                    x: x_test.reshape(1, 28, 28, 1),
                    y_: [y_test],
                })
            orig_prediction = orig_prediction[0]

        # Train theta
        last_ce = 0
        for i in range(MAX_THETA_TRAIN_ITERATIONS):
            x_test = x_test.reshape(1, 28, 28, 1)  # batch, width, height, channels
            ts, ce = sess.run(
                [train_step, cross_entropy],
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })
            if abs(ce - last_ce) < TOLERANCE or i == MAX_THETA_TRAIN_ITERATIONS - 1:
                print '{} iterations, delta ent: {}'.format(i, abs(ce - last_ce))
                break
            last_ce = ce

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        eval_y, eval_h_trans, eval_correct_prediction = sess.run(
            [y, h_trans, correct_prediction],
            feed_dict={
                x: x_test,
                y_: [y_test],
            })
        eval_y = eval_y[0]
        eval_correct_prediction = eval_correct_prediction[0]

        if DISPLAY_PLOTS:
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)
            ax1.imshow(x_test.reshape((28, 28)), cmap='gray', interpolation='none')
            ax2.imshow(eval_h_trans.reshape((28, 28)), cmap='gray', interpolation='none')
            ax1.set_title('Original, y={}, y1={}, c1={:5.4f}'.format(np.argmax(y_test), np.argmax(orig_prediction), np.max(orig_prediction)))
            ax2.set_title('Transformed, yn={}, cn={:5.4f}'.format(np.argmax(eval_y), np.max(eval_y)))

            ax3.cla()
            ax4.cla()
            ax3.bar(np.arange(10), orig_prediction)
            ax4.bar(np.arange(10), eval_y)
            ax3.set_ylim([0, 1])
            ax4.set_ylim([0, 1])

            plt.savefig('figures-test/sin{}.png'.format(num))
            # plt.show()

        return eval_correct_prediction


def main():
    sin = SpatiallyInvariantNetwork()
    x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')

    num_correct = 0
    for i in range(NUM_EXAMPLES):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print '[{}] Evaluating {} of {}'.format(timestamp, i, NUM_EXAMPLES)
        x_test[i] = x_test[i].reshape((28, 28)).transpose().reshape(784,)  # flip image
        if sin.evaluate(x_test[i], y_test[i], i):
            num_correct += 1

    print 'correct: {} out of {}. {}%'.format(
        num_correct,
        len(x_test),
        float(num_correct) * 100 / NUM_EXAMPLES)

if __name__ == '__main__':
    main()


# todo:
    # other paper that shows small changes
        # http://cs231n.stanford.edu/reports2016/119_Report.pdf
    # try another objective?
    # non convex - can't switch class? -- seen some examples where it does (sin85.png)
    # change number of iterations for training theta to a threshold
    # use more complex main network

# done:
    # graph distributions
    # restrict to rotations
    # flip dataset

# random restart
# x optimize for each class, then pick the best
# stochastic gradient descent

    # tensorboard
    # merged, summary
