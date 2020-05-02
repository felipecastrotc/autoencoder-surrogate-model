import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model

from utils import slicer, split

# cd simulations

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


acte = "tanh"
actd = acte
# Encoder
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(200, 100, 1))
e = layers.DepthwiseConv2D(
    (5, 5), strides=2, depth_multiplier=3, activation=acte, padding="same"
)(inputs)
e = layers.DepthwiseConv2D(
    (5, 5), strides=2, depth_multiplier=3, activation=acte, padding="same"
)(e)
e = layers.DepthwiseConv2D(
    (5, 5), strides=5, depth_multiplier=3, activation=acte, padding="same"
)(e)

# Latent space
l = layers.Flatten()(e)
l = layers.Dense(100, activation="linear")(l)

# Decoder
d = layers.Dense(1350, activation="linear")(l)
d = layers.Reshape((10, 5, 27))(d)
d0 = layers.Conv2DTranspose(9, (5, 5), strides=5, activation=actd)(d)
d1 = layers.Conv2DTranspose(9, (5, 5), strides=5, activation=actd)(d)
d2 = layers.Conv2DTranspose(9, (5, 5), strides=5, activation=actd)(d)
d = layers.Concatenate()([d0, d1, d2])
d0 = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=actd, padding="same")(d)
d1 = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=actd, padding="same")(d)
d2 = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=actd, padding="same")(d)
d = layers.Concatenate()([d0, d1, d2])
d0 = layers.Conv2DTranspose(1, (5, 5), strides=2, activation=actd, padding="same")(d)
d1 = layers.Conv2DTranspose(1, (5, 5), strides=2, activation=actd, padding="same")(d)
d2 = layers.Conv2DTranspose(1, (5, 5), strides=2, activation=actd, padding="same")(d)
d = layers.Concatenate()([d0, d1, d2])
decoded = layers.Conv2DTranspose(1, (5, 5), activation="linear", padding="same")(d)

ae = Model(inputs, decoded)
ae.summary()

ae.compile(optimizer="adam", loss="mse")
ae.fit(trn, trn, epochs=15, batch_size=64, shuffle=True, validation_data=(vld, vld))

# ae.evaluate(tst, tst)

dt_in = dt[129, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
print(np.mean((dt_in - dt_out[0, :, :, 0]) ** 2))

fig, ax = plt.subplots(2, figsize=(10, 10))
ax[0].pcolormesh(dt_in[:, :].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)

# tst = dt_out[0, :, :, 0]/dt_in
# tst = tst.flatten()
# tst = tst[~np.isinf(tst)]
# tst.mean()