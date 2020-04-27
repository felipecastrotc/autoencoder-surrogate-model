import sys

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model

from utils import slicer, split, format_data

# cd simulations

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
x = layers.ConvLSTM2D(20, kernel_size=(3, 3), padding="same", return_sequences=True)(
    inputs
)
x = layers.ConvLSTM2D(20, kernel_size=(3, 3), padding="same", return_sequences=True)(x)
x = layers.ConvLSTM2D(20, kernel_size=(3, 3), padding="same", return_sequences=True)(x)
x = layers.ConvLSTM2D(20, kernel_size=(3, 3), padding="same", return_sequences=False)(x)
x = layers.Conv2D(filters=30, kernel_size=(3, 3), activation="tanh", padding="same")(x)
x = layers.Conv2D(filters=1, kernel_size=(3, 3), activation="linear", padding="same")(x)

cvl = Model(inputs, x)
cvl.summary()


cvl.compile(optimizer="adam", loss="mse")
hist = cvl.fit(
    x_data[slc_trn[0]],
    y_data[slc_trn[0]],
    epochs=30,
    batch_size=1,
    shuffle=True,
    validation_data=(x_data[slc_vld[0]], y_data[slc_vld[0]]),
)

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

cvl.save("lstm_conv2D.h5")

i = 50
dt_in = x_data[i][np.newaxis, :]
dt_out = cvl.predict(dt_in)
print(np.mean((y_data[i] - dt_out[0]) ** 2))

fig, ax = plt.subplots(2, figsize=(10, 10))
ax[0].pcolormesh(y_data[i, :, :, 0].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)
