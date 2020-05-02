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