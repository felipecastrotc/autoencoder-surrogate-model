import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model

from utils import slicer, split, plot_red_comp
from utils_keras import loss_norm_error

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

# trn = dt[slc_trn][:, :, :, np.newaxis]
# vld = dt[slc_vld][:, :, :, np.newaxis]
# tst = dt[slc_tst][:, :, :, np.newaxis]

trn = dt[slc_trn]
vld = dt[slc_vld]
tst = dt[slc_tst]

# act = "tanh"
act = "elu"
# Encoder
tf.keras.backend.clear_session()
# n = [3, 9, 27]
# n = [17, 32, 56]
n = [17, 32, 55]
# lt = [65, 65]
lt = [54, 76]
# lt = [3, 3]
inputs = layers.Input(shape=trn.shape[1:])
e = layers.Conv2D(n[0], (5, 5), strides=2, activation=act, padding="same")(inputs)
ed = layers.Conv2D(1, (1, 1), padding="same")(e)
e = layers.Conv2D(n[1], (5, 5), strides=2, activation=act, padding="same")(e)
e = layers.Conv2D(n[2], (5, 5), strides=5, activation=act, padding="same")(e)

# Latent space
l = layers.Flatten()(e)
l = layers.Dense(lt[0], activation="linear")(l)

l1 = layers.Flatten()(ed)
l1 = layers.Dense(lt[1], activation="linear")(l1)

# Decoder
n_flat = np.prod(K.int_shape(e)[1:])
d = layers.Dense(n_flat, activation='linear')(l)
d = layers.Reshape(K.int_shape(e)[1:])(d)
d = layers.Conv2D(n[-1], (1, 1), padding="same")(d)
d = layers.Conv2DTranspose(n[-1], (5, 5), strides=5, activation=act)(d)
d = layers.Conv2DTranspose(n[-2], (5, 5), strides=2, activation=act, padding="same")(d)
d1 = layers.Dense(5000, activation="linear")(l1)
d1 = layers.Reshape((100, 50, 1))(d1)
d1 = layers.Conv2D(n[-2], (1, 1), padding="same")(d1)
# d1 = layers.Conv2D(n[-2], (1, 1), padding='same')(ed)
d = layers.Add()([d1, d])
d = layers.Conv2DTranspose(n[-3], (5, 5), strides=2, activation=act, padding="same")(d)
decoded = layers.Conv2DTranspose(trn.shape[-1], (5, 5), activation="linear", padding="same")(d)

ae = Model(inputs, decoded)
ae.summary()

# ae.compile(optimizer="adam", loss="mse", metrics=["mse"])
ae.compile(optimizer="adam", loss=loss_norm_error, metrics=["mse"])
hist = ae.fit(trn, trn, epochs=15, batch_size=4, shuffle=True, validation_data=(vld, vld))

# ae.evaluate(tst, tst)

i = 634
var = 1
org = dt[i]
# org = trn[i]
rec = np.squeeze(ae.predict(org[np.newaxis, :]))
# ((org[:,:,var] - rec[:, :, var])/org[:,:,var]).min()
plot_red_comp(org, rec, var, np.sum(lt), 0, 'AE')

# org = org[:, :, 1]
# rec = np.squeeze(ae.predict(org[np.newaxis, :, :, np.newaxis]))
# plot_red_comp(org, rec, var, np.sum(lt), 0, 'AE')