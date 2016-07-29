import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

def lenet(images):
    images = tf.reshape(images, (1, 28, 28, 1))
    layer0 = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
    layer1 = slim.layers.max_pool2d(net, [2,2], scope='pool1')
    layer2 = slim.layers.conv2d(net, 50, [5,5], scope='conv2')
    layer3 = slim.layers.max_pool2d(net, [2,2], scope='pool2')
    layer4 = slim.layers.flatten(net, scope='flatten3')
    layer5 = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    layer6 = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fully_connected5')
    return net

import imp
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

SAVE_RESTORE = 'SAVE'

def load_data_from_pickle(fname):
    print 'Begin loading data.'
    with open(fname) as f:
        data = pickle.load(f)
        print 'Finished loading data.'
        return data


sess = tf.InteractiveSession()
# x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

images = tf.reshape(x, (1, 28, 28, 1))
layer0 = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
layer1 = slim.layers.max_pool2d(layer0, [2,2], scope='pool1')
layer2 = slim.layers.conv2d(layer1, 50, [5,5], scope='conv2')
layer3 = slim.layers.max_pool2d(layer2, [2,2], scope='pool2')
layer4 = slim.layers.flatten(layer3, scope='flatten3')
layer5 = slim.layers.fully_connected(layer4, 500, scope='fully_connected4')
layer6 = slim.layers.fully_connected(layer5, 10, activation_fn=None, scope='fully_connected5')

y = tf.nn.softmax(layer6)
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
# L0_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv1')
# L1_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool1')
# L2_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv2')
# L3_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool2')
# L4_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='flatten3')
# L5_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected4')
# L6_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected5')
# saver = tf.train.Saver({
    # 'C1W': L0_vars[0],
    # 'C1b': L0_vars[1],
    # 'C2W': L2_vars[0],
    # 'C2b': L2_vars[1],
    # 'FC4W': L5_vars[0],
    # 'FC4b': L5_vars[1],
    # 'FC5W': L6_vars[0],
    # 'FC5b': L6_vars[1],
# })
if SAVE_RESTORE == 'RESTORE':
    saver.restore(sess, 'lenet.ckpt')

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# opt = tf.train.RMSPropOptimizer(0.01, 0.9)
# train_step = opt.minimize(cross_entropy)
batch_size = 1

if SAVE_RESTORE == 'SAVE':
    num_training_examples = 10
    for i in range(num_training_examples):
        print '{} of {}'.format(i, num_training_examples)
        batch = mnist.train.next_batch(1)
        # batch = (x_train[i * batch_size: (i + 1) * batch_size],
               # y_train[i * batch_size: (i + 1) * batch_size])
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})


y = tf.Print(y, [tf.argmax(y, 1)])
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_correct = 0
num_tests = 100
for i in range(num_tests):
    num_correct += correct_prediction.eval(feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]]})
print '{} of {}, {}'.format(num_correct, num_tests, float(num_correct) / num_tests)
if SAVE_RESTORE == 'SAVE':
    save_path = saver.save(sess, "lenet.ckpt")
# slim.losses.softmax_cross_entropy(predictions, labels)
# total_loss = slim.losses.get_total_loss()
# tf.scalar_summary('loss', total_loss)

# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
# train_op = slim.learning.create_train_op(total_loss, optimizer)

# slim.learning.train(train_op, log_dir, save_summaries_secs=20)

