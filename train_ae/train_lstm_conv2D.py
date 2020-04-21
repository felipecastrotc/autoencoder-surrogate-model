import h5py
import sys
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
    idx_trn = np.random.choice(idx, int(n_train * sz), replace=False)
    idx = np.setdiff1d(idx, idx_trn, assume_unique=True)

    idx_vld = np.random.choice(idx, int(n_valid * sz), replace=False)
    idx_tst = np.setdiff1d(idx, idx_vld, assume_unique=True)

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


def format_data(dt, wd=20):
    # Get the simulation indexes
    idxs = dt.attrs['idx']
    n_t_stp = np.min(np.diff(idxs))
    exp_fct = n_t_stp//wd
    x_data = np.empty((len(idxs)*exp_fct, wd-1,  *dt.shape[1:-1], 1))
    y_data = np.empty((len(idxs)*exp_fct, *dt.shape[1:-1], 1))
    # Fill the matrix (sample, time, x, y, z)
    for i, idx in enumerate(idxs):
        for j in range(exp_fct-1):
            slc = slice(idx[0] + wd*j, idx[0] + wd*(j+1) - 1)
            x_data[exp_fct*i + j, :, :, :, 0] = dt[slc, :, :, 2]
            y_data[exp_fct*i + j, :, :, 0] =  dt[idx[0] + wd*(j+1), :, :, 2]
    return x_data, y_data


dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

x_data, y_data = format_data(dt, wd=5)

idxs = split(x_data.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)


inputs = layers.Input(shape=x_data.shape[1:])
x = layers.ConvLSTM2D(20, kernel_size=(3,3), padding='same',return_sequences=True)(inputs)
x = layers.ConvLSTM2D(20, kernel_size=(3,3), padding='same',return_sequences=True)(x)
x = layers.ConvLSTM2D(20, kernel_size=(3,3), padding='same',return_sequences=True)(x)
x = layers.ConvLSTM2D(20, kernel_size=(3,3), padding='same',return_sequences=False)(x)
x = layers.Conv2D(filters=30, kernel_size=(3,3), activation='tanh', padding='same')(x)
x = layers.Conv2D(filters=1, kernel_size=(3,3), activation='linear', padding='same')(x)

cvl = Model(inputs, x)
cvl.summary()


cvl.compile(optimizer="adam", loss="mse")
hist = cvl.fit(x_data[slc_trn[0]], y_data[slc_trn[0]],
               epochs=30,
               batch_size=1,
               shuffle=True,
               validation_data=(x_data[slc_vld[0]], y_data[slc_vld[0]]))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

cvl.save('lstm_conv2D.h5')

i = 50
dt_in = x_data[i][np.newaxis, :]
dt_out = cvl.predict(dt_in)
print(np.mean((y_data[i] - dt_out[0])**2))

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(y_data[i, :, :, 0].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)
