# Spatially Invariant Networks (SIN)

## Story
The goal is to implement a spatially invariant neural network.

The motivation is that as a human given the task of classifying an image that had been spatially perturbed (rotated, translated, skewed), we might look at the image from different angles in order to get a better understanding of what it is.
For example, we have trouble recognizing upside down faces.

By baking spatial invariance into the model, we theoretically maybe be able to train a smaller neural network which focuses only on classifying upright images. Additionally, it may save time in the training phase since training on an augmented dataset (as was made popular by [1]) would be unnecessary.

All credit for this idea goes to Pascal Poupart.

This projects includes an attempt to bring this idea to life. I implemented the testing phase using pretrained networks. Results weren't perfect, so we set out to investigate reasons why it might not be working.

Spatial transformations have been limited to rotations as a proof of concept. Generalizing to the space of transformations given by a 2x3 matrix may be too large of a step though.

## Usage
Modify the `MODEL` variable in `pretrain.py`, `sin.py`, and `rot_sin.py` to use the desired model.
Also ensure that the `SAVE_RESTORE` variable in `pretrain.py` is set to `'SAVE'`.

Run:

    python pretrain.py
    python sin.py
    python rot_sin.py


## Development
* `T_test.py`
* `beginner-88.ckpt`
* `beginner.ckpt`
* `beginner.py` contains pretrain code for the beginner network. Deprecated.
* `dataset.py` contains code to load data.
* `lenet-97.ckpt` checkpoint file for lenet with 97% accuracy.
* `lenet-weak.ckpt` checkpoint file for lenet with around 40% accuracy.
* `mnist-rot-1000.pickle` mnist-rot dataset in pickle format for fast loading.
* `mnist-rot-2000.pickle`
* `models.py` contains classifier models.
* `pickle_mnist_rot_data.py` script to convert mnist-rot dataset from AMAT to pickle format.
* `pretrain.py` pretrains the models defined in `models.py`.
* `requirements.txt`
* `rot_sin.py` graphs entropy and digit confidence over rotation.
* `sin.py` classifies mnist-rot examples and graphs the learned transformations.

### Shape stuff
Lenet model takes tensors of shape (None, 784) and reshapes them to (1, 28, 28, 1).
Beginner model takes tensors of shape (None, 784), and makes sure they're (None, 784).
The `small_fnn` model takes tensors of (None, 784). They get flattened anyway.
The transformer module which performs rotations requires a shape of (1, 28, 28, 1).
MNIST-ROT dataset comes in shape of (1, 784).
MNIST dataset comes in shape of (1, 784).
In general, input shape can be determined by observing input placeholders.
Though for slim.layers.conv2d, you just have to know it's (None, 28, 28, 1).



## Credits
None of this would be possible without the help of Pascal Poupart, Michael Noukhovitch, and Jason Sinn. Thank you for all your help!

[1] https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
https://github.com/daviddao/spatial-transformer-tensorflow


## License
MIT Licence.


TODO:
clean up files
clean up code
add some usage dogs
add some screenshots
write report
  3 pages


