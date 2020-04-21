import h5py

import keras.layers as layers
import numpy as np
import tensorflow as tf

from keras import backend
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt

# cd simulations

def swish(x, beta=1):
    return x * tf.math.sigmoid(beta * x)


get_custom_objects().update({"swish": layers.Activation(swish)})


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

act = 'tanh'
# Encoder
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(200, 100, 1))
e = layers.Conv2D(3, (5, 5), strides= 2, activation=act, padding='same')(inputs)
e = layers.Conv2D(9, (5, 5), strides=2, activation=act,padding='same')(e)
e = layers.Conv2D(27,(5, 5), strides=5, activation=act, padding='same')(e)

# Latent space
l = layers.Flatten()(e)
# l = layers.Dense(100, activation='tanh')(l)
l = layers.Conv2D(8,(5, 5), strides=1, activation=act, padding='same')(e)

# # # Decoder
# d = layers.Dense(1350, activation='tanh')(l)
# d = layers.Reshape((10, 5, 27))(d)
d = layers.Conv2DTranspose(27, (5, 5), strides=5, activation=act)(l)
d = layers.Conv2DTranspose(9, (5, 5), strides=2, activation=act, padding='same')(d)
d = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=act, padding='same')(d)
decoded = layers.Conv2DTranspose(1, (5, 5), activation='linear', padding='same')(d)

ae = Model(inputs, decoded)

ae.summary()

ae.compile(optimizer="adam", loss="mse")
hist = ae.fit(trn, trn,
            epochs=60,
            batch_size=64,
            shuffle=True,
            validation_data=(vld, vld))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.grid(True)

tst = dt[slc_tst][:, :, :, np.newaxis]
ae.evaluate(tst, tst)


dt_in = dt[129, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
np.mean((dt_in - dt_out[0, :, :, 0])**2)

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(dt_in[ :, :].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)


# Model 1
# # Encoder
# inputs = layers.Input(shape=dt.shape[1:])
# e = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# e = layers.MaxPooling2D((2, 2))(e)
# e = layers.Conv2D(32, (3, 3), activation='relu',padding='same')(e)
# e = layers.MaxPooling2D((2, 2))(e)
# e = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(e)
# e = layers.MaxPooling2D((5, 5))(e)

# # Latent space
# l = layers.Flatten()(e)
# l = layers.Dense(400, activation='linear')(l)

# # Decoder
# d = layers.Reshape((10, 5, 16))(l)
# d = layers.Conv2DTranspose(16, (3, 3), strides=5, activation='relu')(d)
# d = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(d)
# d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
# decoded = layers.Conv2DTranspose(3, (3, 3), activation='linear', padding='same')(d)


# Model 2
# Encoder
# inputs = layers.Input(shape=dt.shape[1:])
# inputs = layers.Input(shape=(200, 100, 1))
# e = layers.Conv2D(3, (2, 2), strides= 2, activation=act, padding='same')(inputs)
# e = layers.Conv2D(9, (2, 2), strides=2, activation=act,padding='same')(e)
# e = layers.Conv2D(27,(2, 2), strides=5, activation=act, padding='same')(e)


# # # Latent space
# l = layers.Flatten()(e)
# l = layers.Dense(100, activation='tanh')(l)

# # # Decoder
# d = layers.Dense(1350, activation='tanh')(l)
# d = layers.Reshape((10, 5, 27))(d)
# d = layers.Conv2DTranspose(27, (5, 5), strides=5, activation=act)(d)
# d = layers.Conv2DTranspose(9, (2, 2), strides=2, activation=act, padding='valid')(d)
# d = layers.Conv2DTranspose(3, (2, 2), strides=2, activation=act, padding='valid')(d)
# decoded = layers.Conv2DTranspose(1, (3, 3), activation='linear', padding='same')(d)


# Model 3
# inputs = layers.Input(shape=(200, 100, 1))
# e = layers.Conv2D(3, (6, 3), activation=act,padding='same')(inputs)
# e = layers.AveragePooling2D((2, 2), padding='same')(e)
# e = layers.Conv2D(9, (6, 3), activation=act, padding='same')(e)
# e = layers.AveragePooling2D((2, 2), padding='same')(e)
# e = layers.Conv2D(27, (6, 3), activation=act, padding='same')(e)
# e = layers.AveragePooling2D((5, 5), padding='same')(e)

# # Latent space
# l = layers.Flatten()(e)
# l = layers.Dense(100, activation='tanh')(l)

# d = layers.Dense(1350, activation='tanh')(l)
# d = layers.Reshape((10, 5, 27))(d)
# d = layers.Conv2D(27, (3, 3), activation=act, padding='same')(d)
# d = layers.UpSampling2D((5, 5))(d)
# d = layers.Conv2D(9, (3, 3), activation=act, padding='same')(d)
# d = layers.UpSampling2D((2, 2))(d)
# d = layers.Conv2D(3, (3, 3), activation=act, padding='same')(d)
# d = layers.UpSampling2D((2, 2))(d)
# decoded = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(d)