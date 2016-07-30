import os

import tensorflow as tf
import tensorflow.contrib.slim as slim


def sin_lenet(images):
    images = tf.reshape(images, (1, 28, 28, 1))
    net = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
    # net = slim.layers.conv2d(net, 50, [5,5], scope='conv2')
    # net = slim.layers.max_pool2d(net, [2,2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    # net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fully_connected5')

    L0_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv1')
    L1_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool1')
    L2_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv2')
    L3_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool2')
    L4_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='flatten3')
    L5_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected4')
    L6_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected5')
    var_dict = {
        'C1W': L0_vars[0],
        'C1b': L0_vars[1],
        # 'C2W': L2_vars[0],
        # 'C2b': L2_vars[1],
        # 'FC4W': L5_vars[0],
        # 'FC4b': L5_vars[1],
        'FC5W': L6_vars[0],
        'FC5b': L6_vars[1],
    }
    return net, var_dict


def lenet(images):
    net = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
    # net = slim.layers.conv2d(net, 50, [5,5], scope='conv2')
    # net = slim.layers.max_pool2d(net, [2,2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    # net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fully_connected5')

    L0_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv1')
    L1_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool1')
    L2_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv2')
    L3_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='pool2')
    L4_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='flatten3')
    L5_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected4')
    L6_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='fully_connected5')
    var_dict = {
        'C1W': L0_vars[0],
        'C1b': L0_vars[1],
        # 'C2W': L2_vars[0],
        # 'C2b': L2_vars[1],
        # 'FC4W': L5_vars[0],
        # 'FC4b': L5_vars[1],
        'FC5W': L6_vars[0],
        'FC5b': L6_vars[1],
    }
    return net, var_dict
