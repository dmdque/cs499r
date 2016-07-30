from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load_data_from_pickle
from lenet import lenet


NUM_EXAMPLES = 100
NUM_ROTATIONS = 360
GRAPH_IMAGE_PD = False
PLT_SAVEFIG = True

def model():
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    model, model_var_dict = lenet(x)
    y = tf.nn.softmax(model)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                   reduction_indices=[1]))
    return y, model_var_dict, x, y_, cross_entropy

def main():
    x_train, y_train, x_test, y_test = load_data_from_pickle('mnist-rot-2000.pickle')

    y, model_var_dict, x, y_, cross_entropy = model()

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(model_var_dict)
    saver.restore(sess, 'lenet-97.ckpt')
    for i in range(NUM_EXAMPLES):
        x_test[i] = (x_test[i]
                     .reshape((28, 28))
                     .transpose()
                     .reshape(784,))
        if GRAPH_IMAGE_PD:
            plts = []
            for j in range(1, 2 * NUM_ROTATIONS + 1):
                plts.append(plt.subplot(3, NUM_ROTATIONS, j))
        plt3 = plt.subplot(3, 1, 3)
        entropy_vals = []
        delta_angle = 360 / NUM_ROTATIONS
        for j in range(NUM_ROTATIONS):
            angle = j * delta_angle
            x_case = x_test[i].reshape((28, 28))
            x_case = rotate(x_case, angle, reshape=False)
            y_case = y_test[i]

            prediction, entropy = sess.run(
                [y, cross_entropy],
                feed_dict={
                    x: x_case.reshape(1, 28, 28, 1),
                    y_: [y_case],
                })
            prediction = prediction[0]
            entropy_vals.append(entropy)

            if GRAPH_IMAGE_PD:
                plts[j].cla()
                plts[j].set_title('{}d'.format(angle))
                plts[j].imshow(x_case, cmap='gray', interpolation='none')
                plts[j].axis('off')
                plts[NUM_ROTATIONS + j].cla()
                plts[NUM_ROTATIONS + j].bar(np.arange(10), prediction)
                plts[NUM_ROTATIONS + j].set_title('y={}'.format(np.argmax(prediction)))
                plts[NUM_ROTATIONS + j].set_ylim([0, 1])
                plts[NUM_ROTATIONS + j].set_xticks(range(10))
        plt3.cla()
        plt3.plot(entropy_vals)
        plt3.set_title('Entropy vs Angle')
        if PLT_SAVEFIG:
            plt.savefig('figures-lenet-rotations/fig{}.png'.format(i))
            print 'Saved figure {} of {}'.format(i, NUM_EXAMPLES)
        else:
            plt.show()
            break  # only show one


if __name__ == '__main__':
    main()
