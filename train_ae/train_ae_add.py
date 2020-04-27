import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from sklearn.manifold import TSNE

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

act = "tanh"
# Encoder
tf.keras.backend.clear_session()
n = [3, 9, 27]
inputs = layers.Input(shape=(200, 100, 1))
e = layers.Conv2D(n[0], (5, 5), strides=2, activation=act, padding="same")(inputs)
ed = layers.Conv2D(1, (1, 1), padding="same")(e)
e = layers.Conv2D(n[1], (5, 5), strides=2, activation=act, padding="same")(e)
e = layers.Conv2D(n[2], (5, 5), strides=5, activation=act, padding="same")(e)

# # Latent space
l = layers.Flatten()(e)
l = layers.Dense(50, activation="linear")(l)

l1 = layers.Flatten()(ed)
l1 = layers.Dense(50, activation="linear")(l1)

# # # Decoder
d = layers.Dense(1350, activation="linear")(l)
d = layers.Reshape((10, 5, 27))(d)
d = layers.Conv2D(n[-1], (1, 1), padding="same")(d)
d = layers.Conv2DTranspose(n[-1], (5, 5), strides=5, activation=act)(d)
d = layers.Conv2DTranspose(n[-2], (5, 5), strides=2, activation=act, padding="same")(d)
d1 = layers.Dense(5000, activation="linear")(l1)
d1 = layers.Reshape((100, 50, 1))(d1)
d1 = layers.Conv2D(n[-2], (1, 1), padding="same")(d1)
# d1 = layers.Conv2D(n[-2], (1, 1), padding='same')(ed)
d = layers.Add()([d1, d])
d = layers.Conv2DTranspose(n[-3], (5, 5), strides=2, activation=act, padding="same")(d)
decoded = layers.Conv2DTranspose(1, (5, 5), activation="linear", padding="same")(d)

ae = Model(inputs, decoded)
ae.summary()

ae.compile(optimizer="adam", loss="mse", metrics=["mse"])
ae.fit(trn, trn, epochs=60, batch_size=32, shuffle=True, validation_data=(vld, vld))


# ae.evaluate(tst, tst)

dt_in = dt[129, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
np.mean((dt_in - dt_out[0, :, :, 0]) ** 2)
x = np.abs(dt_in - dt_out[0, :, :, 0] / dt_in)
np.mean(x[~np.isinf(x)])

fig, ax = plt.subplots(2, figsize=(10, 10))
ax[0].pcolormesh(dt_in[:, :].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)

tst = dt_out[0, :, :, 0] / dt_in
tst = tst.flatten()
tst = tst[~np.isinf(tst)]
tst.mean()


aef = Model(inputs, [l])
aef.summary()

out_ae = aef.predict(dt[:, :, :, 2][:, :, :, np.newaxis])
# out_ae = np.squeeze(np.hstack(out_ae))

x = TSNE(perplexity=13).fit_transform(out_ae)

plt.plot(x[:, 0], x[:, 1], "o", alpha=0.5)
plt.grid(True)
