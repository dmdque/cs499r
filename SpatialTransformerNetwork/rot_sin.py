# Explores digit classification confidence for all possible rotations.
# Produces charts for us to look at.
# To experiment, you'll want to modify the MODEL variable.

from scipy.ndimage.interpolation import rotate

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load_mnist_data
from models import lenet
from models import small_fnn


NUM_EXAMPLES = 100
NUM_ROTATIONS = 360
MODEL = 'BEGINNER'

if MODEL == 'LENET':
    SAVEFIG_DIR = 'figures-lenet-rotations'
elif MODEL == 'BEGINNER':
    SAVEFIG_DIR = 'figures-beginner-rotations'
elif MODEL == 'SMALL_FNN':
    SAVEFIG_DIR = 'figures-small-fnn-rotations'


# These functions connect the inputs and optimizer to predefined models.
# Must be defined here because we need a reference to the tensors so they can
# be evaluated.


def model_small_fnn():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    model, model_var_dict = small_fnn(x)
    y = tf.nn.softmax(model)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    return y, model_var_dict, x, y_, cross_entropy


def model_lenet():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    model, model_var_dict = lenet(x)
    y = tf.nn.softmax(model)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    return y, model_var_dict, x, y_, cross_entropy


def model_simple():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    model_var_dict = {
        'W': W,
        'b': b,
    }
    return x, y_, W, b, y, cross_entropy, model_var_dict


def main():
    """
    Explore digit classification confidence for all rotations. Produces charts
    for us to look at.
    """
    x_train, y_train, x_test, y_test = load_mnist_data()

    if MODEL == 'LENET':
        y, model_var_dict, x, y_, cross_entropy = model_lenet()
        ckpt_fname = 'lenet'
    elif MODEL == 'BEGINNER':
        x, y_, W, b, y, cross_entropy, model_var_dict = model_simple()
        ckpt_fname = 'beginner.ckpt'
    if MODEL == 'SMALL_FNN':
        y, model_var_dict, x, y_, cross_entropy = model_small_fnn()
        ckpt_fname = 'small_fnn.ckpt'

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(model_var_dict)
    saver.restore(sess, ckpt_fname)
    confidence_matches = [0 for e in range(10)]
    incorrect_confidence_diffs = []
    correct_confidence_diffs = []

    for i in range(NUM_EXAMPLES):
        plts = []
        for j in range(1, 13):
            plts.append(plt.subplot(13, 1, j))

        entropy_vals = []
        confidence_vals = [[] for e in range(10)]
        delta_angle = 360 / NUM_ROTATIONS
        for j in range(NUM_ROTATIONS):
            angle = j * delta_angle
            x_case = x_test[i].reshape((28, 28))
            x_case = rotate(x_case, angle, reshape=False)
            y_case = y_test[i]  # rename for later conciseness

            prediction, entropy = sess.run(
                [y, cross_entropy],
                feed_dict={
                    x: x_case.reshape(1, 784),
                    y_: [y_case],
                })
            prediction = prediction[0]
            entropy_vals.append(entropy)
            for digit in range(10):
                confidence_vals[digit].append(prediction[digit])

        plts[0].cla()
        plts[0].set_title('Original')
        plts[0].imshow(x_test[i].reshape((28, 28)),
                       cmap='gray',
                       interpolation='none')
        plts[0].axis('off')

        corr_pred = np.argmax(y_case)
        print '[Example {}] Correct Prediction: {}'.format(i, corr_pred)
        # max confidence over rotation for each digit
        digit_confidences = [np.max(confidence_vals[digit])
                             for digit in range(10)]
        # note [::-1] reverses order of np array
        digit_confidence_order = np.argsort(digit_confidences)[::-1]
        correct_rank = np.where(digit_confidence_order == corr_pred)
        correct_rank = correct_rank[0][0]
        confidence_matches[correct_rank] += 1
        confidence_diff = abs(digit_confidences[corr_pred] -
                              np.max(digit_confidences))
        if correct_rank != 0:  # only do if it was incorrect
            incorrect_confidence_diffs.append(confidence_diff)
        else:
            # if it was correct, how much did it beat the others by?
            corr_conf_diff = (digit_confidences[digit_confidence_order[0]] -
                              digit_confidences[digit_confidence_order[1]])
            correct_confidence_diffs.append(corr_conf_diff)

        print ('[Example {}] Confidence matches so far: {}'
               .format(i, confidence_matches[0]))

        for digit in range(10):
            plts[digit + 1].cla()
            confidence_vals[digit] = np.roll(confidence_vals[digit],
                                             NUM_ROTATIONS / 2)
            x_axis = range(-NUM_ROTATIONS / 2, NUM_ROTATIONS / 2)
            plts[digit + 1].fill_between(x_axis, 0, confidence_vals[digit])
            print '[Example {}] {}: {} {}'.format(
                i,
                digit,
                np.max(confidence_vals[digit]),
                np.argmax(confidence_vals[digit]))
            plts[digit + 1].axis('off')
        plts[11].cla()
        entropy_vals = np.roll(entropy_vals, NUM_ROTATIONS / 2)
        entropy_x_axis = range(-NUM_ROTATIONS / 2, NUM_ROTATIONS / 2)
        plts[11].fill_between(entropy_x_axis, 0, entropy_vals, facecolor='red')
        plts[11].set_yticks([])
        plt.savefig('{}/fig{}.png'.format(SAVEFIG_DIR, i))
        print 'Saved figure {} ({} total)'.format(i, NUM_EXAMPLES)

    print 'Model used: {}'.format(MODEL)
    print 'Confidence matches: {}'.format(confidence_matches)
    confidence_matches_pdf = (np.array(confidence_matches).astype(float) /
                              NUM_EXAMPLES)
    confidence_matches_cdf = np.cumsum(confidence_matches_pdf)
    plt.clf()

    plt1 = plt.subplot(2, 1, 1)
    plt2 = plt.subplot(2, 2, 3)
    plt3 = plt.subplot(2, 2, 4)

    plt1.plot(range(1, 11), confidence_matches_pdf)
    plt1.plot(range(1, 11), confidence_matches_cdf, 'r--')
    plt1.set_title('% cases with correct class with top confidence')
    plt1.set_xlabel('Confidence ranking')
    # When correct class doesn't have highest confidence
    plt2.plot(range(1, len(incorrect_confidence_diffs) + 1),
              incorrect_confidence_diffs, '*')
    # Confidence difference for correct and best classes
    plt2.set_title('Confidence delta (incorrect)')
    plt2.set_xlabel('Case #')
    plt2.set_yscale('log')

    plt3.plot(range(1, len(correct_confidence_diffs) + 1),
              correct_confidence_diffs, '*')
    # Confidence differences between correct class and second best, when corr
    # class was highest conf
    plt3.set_title('Confidence delta (correct)')
    plt3.set_xlabel('examples')
    plt3.set_yscale('log')
    plt.tight_layout()
    plt.savefig('{}/confidence-matches{}.png'.format(SAVEFIG_DIR,
                                                     NUM_EXAMPLES))


if __name__ == '__main__':
    main()
