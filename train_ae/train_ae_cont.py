import h5py

import keras.layers as layers
import numpy as np
import tensorflow as tf

from keras import backend
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt

# cd simulations

def split(sz, n_train=0.8, n_valid=0.1, shuffle=True):
    # Percentage for the test dataset
    n_test = 1 - n_train - n_valid
    # Generate an index array
    idx = np.array(range(sz))
    # Get the datasets indexes
    idx_tst = np.random.choice(idx, int(n_test * sz), replace=False)
    idx = np.setdiff1d(idx, idx_tst, assume_unique=True)

    idx_vld = np.random.choice(idx, int(n_valid * sz), replace=False)
    idx_trn = np.setdiff1d(idx, idx_vld, assume_unique=True)

    # # Shuffle the train dataset
    if shuffle:
        np.random.shuffle(idx_trn)

    return idx_trn, idx_vld, idx_tst


def slicer(shp, idxs):
    # It is assumed that the first dimension is the samples
    slc = []
    # Iterate over the datasets
    for idx in idxs:
        idx.sort()
        slc += [tuple([idx] + [slice(None)] * (len(shp) - 2) + [2])]
    return tuple(slc)


dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]


idxs = split(dt.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)
trn = dt[slc_trn][:, :, :, np.newaxis]
vld = dt[slc_vld][:, :, :, np.newaxis]
tst = dt[slc_tst][:, :, :, np.newaxis]


acte = 'tanh'
actd = acte
# Encoder
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(200, 100, 1))
e = layers.Conv2D(3, (3, 3), strides= 2, activation=acte, padding='same')(inputs)
e = layers.Dropout(0.25)(e)
e = layers.Conv2D(9, (5, 5), strides=2, activation=acte,padding='same')(e)
e = layers.Dropout(0.25)(e)
e = layers.Conv2D(27,(5, 5), strides=5, activation=acte, padding='same')(e)
e = layers.Dropout(0.25)(e)

# Latent space
l = layers.Flatten()(e)
l = layers.Dense(100, activation='linear')(l)
l = layers.Dropout(0.25)(l)

# # Decoder
d = layers.Dense(1350, activation='linear')(l)
d = layers.Reshape((10, 5, 27))(d)
d = layers.Conv2DTranspose(27, (5, 5), strides=5, activation=actd)(d)
d = layers.Conv2DTranspose(9, (5, 5), strides=2, activation=actd, padding='same')(d)
d = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=actd, padding='same')(d)
decoded = layers.Conv2DTranspose(1, (5, 5), activation='linear', padding='same')(d)

ae = Model(inputs, decoded)
ae.summary()

ae.compile(optimizer="adam", loss="mse")
ae.fit(trn, trn,
        epochs=60,
        batch_size=64,
        shuffle=True,
        validation_data=(vld, vld))

print(1)

ae.evaluate(trn, trn)
# ae.evaluate(tst, tst)

dt_in = dt[53, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
print(np.mean((dt_in - dt_out[0, :, :, 0])**2))

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(dt_in[ :, :].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)

# tst = dt_out[0, :, :, 0]/dt_in 
# tst = tst.flatten()
# tst = tst[~np.isinf(tst)]
# tst.mean()
