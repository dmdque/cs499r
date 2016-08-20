from datetime import datetime
import imp
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import load_data_from_pickle
from models import beginner
from models import lenet
from models import small_fnn

spatial_transformer = imp.load_source(
    'spatial_transformer',
    '../../spatial-transformer-tensorflow/spatial_transformer.py')
from spatial_transformer import transformer

# Feature flags
NUM_TEST_EXAMPLES = 100
NUM_TRAIN_EXAMPLES = 1000
NUM_VALID_EXAMPLES = 20
DISPLAY_PLOTS = True
RESTRICT_ROTATE = True
MAX_THETA_TRAIN_ITERATIONS = 1000
MODEL = 'LENET'
TOLERANCE = 10 ** -5
SAVEFIG_DIR = 'figures-test'
USE_PRETRAIN = False

if MODEL == 'LENET':
    MODEL_CKPT = 'lenet-97.ckpt'
elif MODEL == 'BEGINNER':
    MODEL_CKPT = 'beginner.ckpt'
elif MODEL == 'SMALL_FNN':
    MODEL_CKPT = 'small_fnn.ckpt'


def model_sin():
    """
    Create model and return tensors necessary to run the model.

    Creates both training and testing phase tensors.
    """
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
    transformed_x = transformer(x, h_fc1, (28, 28))
    if MODEL == 'LENET':
        net, model_var_dict = lenet(transformed_x)
    elif MODEL == 'BEGINNER':
        transformed_x = tf.reshape(transformed_x, (1, 28, 28, 1))
        W, b, net, model_var_dict = beginner(transformed_x)
    elif MODEL == 'SMALL_FNN':
        transformed_x = tf.reshape(transformed_x, (1, 784))
        net, model_var_dict = small_fnn(transformed_x)
    y = tf.nn.softmax(net)

    # test phase tensors
    test_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    test_opt = tf.train.GradientDescentOptimizer(10 ** -2)
    test_train_step = test_opt.minimize(test_cross_entropy,
                              var_list=[theta])
    # train phase tensors
    train_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    train_opt = tf.train.GradientDescentOptimizer(10 ** -2)
    theta_train_step = train_opt.minimize(train_cross_entropy,
                              var_list=model_var_dict.values())

    # return all tensors since references are required to run operations
    return (x, y_, theta, rot_matrix, h_fc1, transformed_x, net,
            model_var_dict, y, test_cross_entropy, test_train_step,
            train_cross_entropy, train_opt, theta_train_step)


class SpatiallyInvariantNetwork:
    def __init__(self):
        """Initializes the model and loads variables from file."""
        print 'Begin setup.'
        x, y_, theta, rot_matrix, h_fc1, transformed_x, net, model_var_dict, y, test_cross_entropy, test_train_step, train_cross_entropy, train_opt, theta_train_step = model_sin()
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(model_var_dict)
        if USE_PRETRAIN:
            saver.restore(sess, MODEL_CKPT)

        self.test_cross_entropy = test_cross_entropy
        self.sess = sess
        self.theta = theta
        self.test_train_step = test_train_step
        self.transformed_x = transformed_x
        self.x = x
        self.y = y
        self.y_ = y_
        self.train_cross_entropy = train_cross_entropy
        self.train_opt = train_opt
        self.theta_train_step = theta_train_step
        print 'Finished setup.'


    def run(self, x_train, y_train):
        """Trains the T layer of the model."""
        sess = self.sess
        x = self.x
        y = self.y
        y_ = self.y_
        theta = self.theta
        train_cross_entropy = self.train_cross_entropy
        train_opt = self.train_opt
        theta_train_step = self.theta_train_step

        # reset theta before testing again
        if RESTRICT_ROTATE:
            initial = 0.0
        else:
            initial = np.array([[1, 0, 0], [0, 1, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
        assign_op = theta.assign(initial)
        sess.run(assign_op)

        # Train theta
        last_entropy = 0
        for i in range(MAX_THETA_TRAIN_ITERATIONS):
            x_train = x_train.reshape(1, 28, 28, 1)  # batch, width, height, channels
            ts, curr_entropy = sess.run(
                [theta_train_step, train_cross_entropy],
                feed_dict={
                    x: x_train,
                    y_: [y_train],  # tensor isn't used in this phase
                })
            if abs(curr_entropy - last_entropy) < TOLERANCE:
                break
            last_entropy = curr_entropy

        # Train main network
        ts = sess.run(
            theta_train_step,
            feed_dict={
                x: x_train,
                y_: [y_train],
            })


    def evaluate(self, x_test, y_test, num, is_validate=False):
        """Evaluates the model."""
        x = self.x
        y = self.y
        y_ = self.y_
        theta = self.theta
        transformed_x = self.transformed_x
        sess = self.sess
        test_cross_entropy = self.test_cross_entropy
        test_train_step = self.test_train_step

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
        last_entropy = 0
        for i in range(MAX_THETA_TRAIN_ITERATIONS):
            x_test = x_test.reshape(1, 28, 28, 1)  # batch, width, height, channels
            ts, curr_entropy = sess.run(
                [test_train_step, test_cross_entropy],
                feed_dict={
                    x: x_test,
                    y_: [y_test],
                })
            if abs(curr_entropy - last_entropy) < TOLERANCE:
                break
            last_entropy = curr_entropy
        else:
            pass

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        eval_y, eval_transformed_x, eval_correct_prediction = sess.run(
            [y, transformed_x, correct_prediction],
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
            ax2.imshow(eval_transformed_x.reshape((28, 28)), cmap='gray', interpolation='none')
            ax1.set_title('Original, y={}, y1={}, c1={:5.4f}'.format(np.argmax(y_test), np.argmax(orig_prediction), np.max(orig_prediction)))
            ax2.set_title('Transformed, yn={}, cn={:5.4f}'.format(np.argmax(eval_y), np.max(eval_y)))

            ax3.cla()
            ax4.cla()
            ax3.bar(np.arange(10), orig_prediction)
            ax4.bar(np.arange(10), eval_y)
            ax3.set_ylim([0, 1])
            ax4.set_ylim([0, 1])

            if is_validate:
                plt.savefig('{}/sin{}.png'.format('figures-valid', num))
            else:
                plt.savefig('{}/sin{}.png'.format(SAVEFIG_DIR, num))
            # plt.show()

        return eval_correct_prediction


def validate(sin, x_valid, y_valid, base_i=0):
    print 'Validation'
    num_correct = 0
    for i in range(NUM_VALID_EXAMPLES):
        x_valid_i = (x_valid[i]
                     .reshape((28, 28))
                     .transpose()
                     .reshape(784,))  # flip image
        if sin.evaluate(x_valid_i, y_valid[i], i + base_i, is_validate=True):
            num_correct += 1

    print '[Validation] correct: {} out of {}. {}%'.format(
        num_correct,
        NUM_VALID_EXAMPLES,
        float(num_correct) * 100 / NUM_VALID_EXAMPLES)


def main():
    print 'DISPLAY_PLOTS:', DISPLAY_PLOTS
    print 'RESTRICT_ROTATE:', RESTRICT_ROTATE
    print 'MAX_THETA_TRAIN_ITERATIONS:', MAX_THETA_TRAIN_ITERATIONS
    print 'MODEL:', MODEL
    print 'TOLERANCE:', TOLERANCE
    print 'NUM_TEST_EXAMPLES:', NUM_TEST_EXAMPLES
    print 'SAVEFIG_DIR:', SAVEFIG_DIR
    print 'MODEL_CKPT:', MODEL_CKPT

    sin = SpatiallyInvariantNetwork()
    x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')
    # TODO: split a proper validation set
    x_valid = x_test
    y_valid = y_test

    # Training phase
    print 'Training phase'
    for i in range(NUM_TRAIN_EXAMPLES):
        if i % 10 == 0:
            print 'Training on example {}'.format(i)
        if i % 100 == 0:
            validate(sin,
                     x_valid[i:i + NUM_VALID_EXAMPLES],
                     y_valid[i:i + NUM_VALID_EXAMPLES],
                     base_i=i)
        x_train_i = (x_train[i]
                     .reshape((28, 28))
                     .transpose()
                     .reshape(784,))  # flip image
        sin.run(x_train_i, y_train[i])

    # Testing phase
    print 'Testing phase'
    num_correct = 0
    for i in range(NUM_TEST_EXAMPLES):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print '[{}] Evaluating {} of {}'.format(timestamp, i, NUM_TEST_EXAMPLES)
        x_test_i = (x_test[i]
                     .reshape((28, 28))
                     .transpose()
                     .reshape(784,))  # flip image
        if sin.evaluate(x_test_i, y_test[i], i):
            num_correct += 1

    print 'correct: {} out of {}. {}%'.format(
        num_correct,
        NUM_TEST_EXAMPLES,
        float(num_correct) * 100 / NUM_TEST_EXAMPLES)

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

# test for each class separately, use the best
    # plot belief in each digit vs rotation
# training
# random restart
# tuning hyper parameters
    # james berkstra

# reduce p of current classification
    # reclassify
    # non-random exploration
