import sys

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model

from utils import slicer, split, format_data


dt_fl = "data_compact.h5"
dt_dst = "model_ae-smp_4_scaled"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

x_data, y_data = format_data(dt, wd=4, get_y=True)

idxs = split(x_data.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)

tf.keras.backend.clear_session()
inputs = layers.Input(shape=x_data.shape[1:])
p = layers.LSTM(90, activation='relu')(inputs)
p = layers.BatchNormalization()(p)
# p = layers.Dropout(0.9)(p)
out = layers.Dense(x_data.shape[2], activation='linear')(p)

pred = Model(inputs, out)
pred.summary()

pred.compile(optimizer="adam", loss="mse")

hist = pred.fit(
    x_data[slc_trn[0]],
    y_data[slc_trn[0]],
    epochs=300,
    batch_size=8,
    shuffle=True,
    validation_data=(x_data[slc_vld[0]], y_data[slc_vld[0]]),
)

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

# Dense network
x_data, y_data = format_data(dt, wd=2, get_y=True)

x_data = np.squeeze(x_data)
idxs = split(x_data.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)

tf.keras.backend.clear_session()
inputs = layers.Input(shape=x_data.shape[1:])
# p = layers.LSTM(90, activation='relu')(inputs)
# p = layers.Dropout(0.9)(p)
p = layers.Dense(180, activation='relu')(inputs)
p = layers.Dropout(0.5)(p)
out = layers.Dense(y_data.shape[1], activation='linear')(p)

pred = Model(inputs, out)
pred.summary()

pred.compile(optimizer="adam", loss="mse")

hist = pred.fit(
    x_data[slc_trn[0]],
    y_data[slc_trn[0]],
    epochs=100,
    batch_size=8,
    shuffle=True,
    validation_data=(x_data[slc_vld[0]], y_data[slc_vld[0]]),
)

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
