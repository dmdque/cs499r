import imp
import pickle

import numpy as np

amat = imp.load_source('amat', 'amat.py')
from amat import AMat


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
    print 'Begin loading data.'
    train_data = AMat('data/mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all
    test_data = AMat('data/mnist-rot/mnist_all_rotation_normalized_float_test.amat').all
    # note: last entry is label
    x_train, y_train = train_data[:, :-1], train_data[:, -1:]
    x_test, y_test = test_data[:, :-1], test_data[:, -1:]
    y_train = augment_label(y_train)
    y_test = augment_label(y_test)
    print 'Finished loading data.'
    return x_train, y_train, x_test, y_test


num_cases = 5000
x_train, y_train, x_test, y_test = load_data()
trunc_data = (x_train[:num_cases],
              y_train[:num_cases],
              x_test[:num_cases],
              y_test[:num_cases])
with open('mnist-rot-{}.pickle'.format(num_cases), 'w') as f:
    pickle.dump(trunc_data, f)
