from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import theano
np.random.seed(1337)  # for reproducibility
theano.config.openmp = True

from scipy.misc import imresize

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, MaxPooling1D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

from amat import AMat


batch_size = 128
nb_classes = 10
nb_epoch = 12

DIM = 28

# load data
train_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat').all[:20]
test_data = AMat('mnist-rot/mnist_all_rotation_normalized_float_test.amat').all[:20]

# note: last entry is label
x_train, y_train = train_data[:, :-1], train_data[:, -1:]
x_test, y_test = test_data[:, :-1], test_data[:, -1:]

# reshape
x_train = x_train.reshape(x_train.shape[0], 1, DIM, DIM)
x_test = x_test.reshape(x_test.shape[0], 1, DIM, DIM)

#y_train = np_utils.to_categorical(y_train, nb_classes)
#y_test = np_utils.to_categorical(y_test, nb_classes)

input_shape = x_train.shape[1:]  # should be (28, 28)
#plt.imshow(train_data[0][:-1].reshape(28, 28))
# note try this: #plt.imshow(X_train[101].reshape(DIM, DIM), cmap='gray', interpolation='none')
#plt.show()

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]
print(weights)

## note: simple CNN for now. no locnet
#locnet = Sequential()
#locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
#locnet.add(Convolution2D(20, 5, 5))
#locnet.add(MaxPooling2D(pool_size=(2,2)))
#locnet.add(Convolution2D(20, 5, 5))
#locnet.add(Flatten())
#locnet.add(Dense(50))
#locnet.add(Activation('relu'))
#locnet.add(Dense(6, weights=weights))
#locnet.add(Activation('sigmoid'))
#locnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model = Sequential()
#model.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
#model.add(Convolution2D(32, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation('relu'))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model = Sequential()
model.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
model.add(Convolution2D(20, 5, 5))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(20, 5, 5))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(6, weights=weights))
model.add(Activation('sigmoid'))

## create model
#print(input_shape)
#model = Sequential()
#model.add(Dense(50, init='uniform', activation='relu', input_shape=input_shape))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=10, batch_size=10)
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#XX = model.input
#YY = model.layers[0].output
#F = theano.function([XX], YY)

#nb_epochs = 10 # you probably want to go longer than this
#batch_size = 256
#fig = plt.figure()

#try:
    #for e in range(nb_epochs):
        #print('-'*40)
        ##progbar = generic_utils.Progbar(x_train.shape[0])
        #for b in range(x_train.shape[0]/batch_size):
            #f = b * batch_size
            #l = (b+1) * batch_size
            #x_batch = x_train[f:l].astype('float32')
            #y_batch = y_train[f:l].astype('float32')
            #loss = model.train_on_batch(x_batch, y_batch)
            ##progbar.add(x_batch.shape[0], values=[("train loss", loss)])
        #scorev = model.evaluate(X_valid, y_valid, show_accuracy=True, verbose=0)[1]
        #scoret = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)[1]
        #print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))

        ##if e % 5 == 0:
            ##Xresult = F(x_batch[:9])
            ##plt.clf()
            ##for i in range(9):
                ##plt.subplot(3, 3, i+1)
                ##plt.imshow(Xresult[i, 0], cmap='gray')
                ##plt.axis('off')
            ##fig.canvas.draw()
            ##plt.show()
#except KeyboardInterrupt:
    #pass


