import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from utils import slicer, split

# cd simulations

dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

idxs = split(dt, n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)
y_trn, y_vld, y_tst = slicer(dt.shape, idxs, add=1)

trn = dt[slc_trn][:, :, :, np.newaxis]
y_trn = dt[y_trn][:, :, :, np.newaxis]
vld = dt[slc_vld][:, :, :, np.newaxis]
y_vld = dt[y_vld][:, :, :, np.newaxis]


act = 'tanh'
# Encoder
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(200, 100, 1))
e = layers.Conv2D(3, (5, 5), strides= 2, activation=act, padding='same')(inputs)
e = layers.Conv2D(9, (5, 5), strides=2, activation=act,padding='same')(e)
e = layers.Conv2D(27,(5, 5), strides=5, activation=act, padding='same')(e)

# Latent space
l = layers.Flatten()(e)
l = layers.Dense(200, activation='linear')(l)

# Decoder
d = layers.Dense(1350, activation='tanh')(l)
d = layers.Reshape((10, 5, 27))(d)
d = layers.Conv2DTranspose(27, (5, 5), strides=5, activation=act)(d)
d = layers.Conv2DTranspose(9, (5, 5), strides=2, activation=act, padding='same')(d)
d = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=act, padding='same')(d)
decoded = layers.Conv2DTranspose(1, (5, 5), activation='linear', padding='same')(d)
# Decoder pred
dp = layers.Dense(1350, activation='tanh')(l)
dp = layers.Reshape((10, 5, 27))(dp)
dp = layers.Conv2DTranspose(27, (5, 5), strides=5, activation=act)(dp)
dp = layers.Conv2DTranspose(9, (5, 5), strides=2, activation=act, padding='same')(dp)
dp = layers.Conv2DTranspose(3, (5, 5), strides=2, activation=act, padding='same')(dp)
decoded_pred = layers.Conv2DTranspose(1, (5, 5), activation='linear', padding='same')(dp)

ae = Model(inputs, [decoded, decoded_pred])
ae.summary()

ae.compile(optimizer="adam", loss=["mse", "mse"])
hist = ae.fit(trn, [trn, y_trn],
            epochs=60,
            batch_size=2,
            shuffle=True,
            validation_data=(vld, [vld, y_vld]))


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

min(hist.history['conv2d_transpose_8_loss'])
min(hist.history['conv2d_transpose_4_loss'])
plt.plot(hist.history['conv2d_transpose_8_loss'])
plt.plot(hist.history['conv2d_transpose_4_loss'])

plt.plot(hist.history['val_conv2d_transpose_8_loss'])
plt.plot(hist.history['val_conv2d_transpose_4_loss'])

plt.plot(hist.history['conv2d_transpose_4_loss'])
plt.plot(hist.history['val_conv2d_transpose_4_loss'])


i = 20
dt_in = dt[i, :, :, 2]
y_true = dt[i+1, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
np.mean((dt_in - dt_out[0][0, :, :, 0])**2)

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(dt_in.T, rasterized=True)
ax[1].pcolormesh(dt_out[0][0, :, :, 0].T, rasterized=True)


aef = Model(inputs, decoded)
aef.summary()

i = 20
dt_in = dt[i, :, :, 2]
dt_out = aef.predict(dt_in[np.newaxis, :, :, np.newaxis])
np.mean((dt_in - dt_out[0, :, :, 0])**2)

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(dt_in.T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)


# Saving models
# Full Model
ae.save('trn1_dual_model_v1.h5')

# Only the 'present' model
aef = Model(inputs, decoded)
aef.save('trn1_dual_model_pt1_v1.h5')

# Only the encoder
ae_enc = Model(inputs, l)
ae_enc.summary()
ae_enc.save('trn1_dual_model_pt1_enc_v1.h5')

# Only the decoder
ae_enc = Model(l, decoded)
ae_enc.summary()
ae_enc.save('trn1_dual_model_pt1_dec_v1.h5')


nn = load_model('trn1_dual_model_pt1_v1.h5')
nn.summary()

nn.layers

dec_input = layers.Input(shape=(100,))
ae_dec = Model(dec_input, )

ae_enc.summary()
