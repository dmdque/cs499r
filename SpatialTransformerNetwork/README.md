# Spatially Invariant Networks (SIN)

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
