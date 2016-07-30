import imp
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from lenet import lenet
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

SAVE_RESTORE = 'RESTORE'

def load_data_from_pickle(fname):
    print 'Begin loading data.'
    with open(fname) as f:
        data = pickle.load(f)
        print 'Finished loading data.'
        return data


sess = tf.InteractiveSession()
# x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')


x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

model, model_var_dict = lenet(x)
y = tf.nn.softmax(model)
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(model_var_dict)
if SAVE_RESTORE == 'RESTORE':
    saver.restore(sess, 'lenet-weak.ckpt')

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# opt = tf.train.RMSPropOptimizer(0.01, 0.9)
# train_step = opt.minimize(cross_entropy)
batch_size = 1

if SAVE_RESTORE == 'SAVE':
    num_training_examples = 20
    for i in range(num_training_examples):
        print '{} of {}'.format(i, num_training_examples)
        batch = mnist.train.next_batch(50)
        batch = batch[0].reshape((50, 28, 28, 1)), batch[1]
        # batch = (x_train[i * batch_size: (i + 1) * batch_size],
               # y_train[i * batch_size: (i + 1) * batch_size])
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})


y = tf.Print(y, [tf.argmax(y, 1)])
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_correct = 0
num_tests = 1000
for i in range(num_tests):
    batch = mnist.test.images[i].reshape((28, 28, 1)), mnist.test.labels[i]
    num_correct += correct_prediction.eval(feed_dict={x: [batch[0]], y_: [batch[1]]})
print '{} of {}, {}'.format(num_correct, num_tests, float(num_correct) / num_tests)
if SAVE_RESTORE == 'SAVE':
    save_path = saver.save(sess, "lenet-weak.ckpt")
# slim.losses.softmax_cross_entropy(predictions, labels)
# total_loss = slim.losses.get_total_loss()
# tf.scalar_summary('loss', total_loss)

# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
# train_op = slim.learning.create_train_op(total_loss, optimizer)

# slim.learning.train(train_op, log_dir, save_summaries_secs=20)

