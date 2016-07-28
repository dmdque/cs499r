import numpy as np

theta = np.matrix(
    [[0, -1],
     [1, 0]])

xo = np.matrix(
    [[1, 1, 0],
     [1, 1, 1],
     [0, 1, 0]])
x = np.reshape(xo, (1, 9))
# this T corresponds to a 90 degree rotation
T = np.matrix(
    [[0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0]])

y = x * T
yout = np.reshape(y, (3, 3))
print 'x initial:\n', xo
print 'transformation T:\n', T
print 'y:\n', yout

y = y * T
print np.reshape(y, (3, 3))
y = y * T
print np.reshape(y, (3, 3))
