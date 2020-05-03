import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend
from keras import constraints as cnt
from keras import optimizers
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
cnt_mm = cnt.MinMaxNorm(min_value=-1, max_value=2)
# Encoder
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(200, 100, 1))
ed = fconv(inputs, 4, 2, 3)
e = fconv(ed, 3, 2, 9)
e = fconv(e, 3, 5, 27)

# Latent space
# l = layers.Flatten()(e)
# l = layers.Dense(100, activation='linear')(l)
# l = layers.Dropout(0.25)(l)
l = layers.Conv2D(4, (1,1), activation='linear', padding='same')(e)

# Decoder
# d = layers.Dense(1350, activation='linear')(l)
# d = layers.Reshape((10, 5, 27))(d)
d = fconv_e(l, 3, 5, 27)
d = fconv_e(d, 3, 2, 9)
d = fconv_e(d, 3, 2, 3)
decoded = layers.Conv2DTranspose(1, (5, 5), activation='linear', padding='same')(d)

ae = Model(inputs, decoded)
ae.summary()

# adam = optimizers.Adam(learning_rate=0.0001, amsgrad=False)
ae.compile(optimizer="adam", loss="mse")
hist = ae.fit(trn, trn,
              epochs=15,
              batch_size=32,
              shuffle=True,
              validation_data=(vld, vld))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])


tst = dt[slc_tst][:, :, :, np.newaxis]
ae.evaluate(tst, tst)

dt_in = dt[53, :, :, 2]
dt_out = ae.predict(dt_in[np.newaxis, :, :, np.newaxis])
np.mean((dt_in - dt_out[0, :, :, 0])**2)

fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].pcolormesh(dt_in[ :, :].T, rasterized=True)
ax[1].pcolormesh(dt_out[0, :, :, 0].T, rasterized=True)



def fconv(inputs, contract, stride, filters, alpha=1):

    in_nfilters = backend.int_shape(inputs)[-1]
    out_nfilters = int(alpha*filters)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    # Contract
    x = layers.Conv2D(int(in_nfilters*contract),
                     kernel_size=(5, 5),
                     strides=5,
                     padding='same',
                     use_bias=False,
                     activation=None)(inputs)
    x = layers.Activation('tanh')(x)
    # Expand
    x = layers.Conv2DTranspose(in_nfilters,
                               kernel_size=(5, 5),
                               strides=5,
                               padding='same',
                               use_bias=False,
                               activation=None)(x)
    x = layers.Activation('tanh')(x)
    x =  layers.Subtract()([inputs, x])
    # Reduce
    x = layers.Conv2D(out_nfilters,
                    kernel_size=(5, 5),
                    padding='same',
                    strides=stride,
                    use_bias=False,
                    activation=None)(x)
    x = layers.Activation('linear')(x)
    # This layers does not have an activation function, it is linear
    return x


def fconv_e(inputs, contract, stride, filters, alpha=1):

    in_nfilters = backend.int_shape(inputs)[-1]
    out_nfilters = int(alpha*filters)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    # Expand
    x = layers.Conv2DTranspose(int(in_nfilters*contract),
                               kernel_size=(5, 5),
                               strides=5,
                               padding='same',
                               use_bias=False,
                               activation=None)(inputs)
    x = layers.Activation('tanh')(x)
    # Contract
    x = layers.Conv2D(in_nfilters,
                      kernel_size=(5, 5),
                      strides=5,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)
    x = layers.Activation('tanh')(x)
    x =  layers.Subtract()([inputs, x])
    # Reduce/increase
    x = layers.Conv2DTranspose(out_nfilters,
                                kernel_size=(5, 5),
                                padding='same',
                                strides=stride,
                                use_bias=False,
                                activation=None)(x)
    x = layers.Activation('linear')(x)
    # This layers does not have an activation function, it is linear
    return x
