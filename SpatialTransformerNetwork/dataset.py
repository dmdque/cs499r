import pickle


def load_data_from_pickle(fname):
    print 'Begin loading data.'
    with open(fname) as f:
        data = pickle.load(f)
        print 'Finished loading data.'
        return data


def load_mnist_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.train.images
    y_test = mnist.train.labels
    return x_train, y_train, x_test, y_test


def load_mnist_data_orig_format():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist
