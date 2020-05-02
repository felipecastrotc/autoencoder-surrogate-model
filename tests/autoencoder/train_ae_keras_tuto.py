from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Conv2DTranspose, Conv3D, UpSampling3D, MaxPooling3D
from keras.layers import DepthwiseConv2D, SeparableConv2D

import h5py
import numpy as np

import matplotlib.pyplot as plt


dt_fl = 'nn_data.h5'
dt_dst = 'scaled_data'

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, 'r')
dt = f[dt_dst]

idxs = split(dt.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

print('The input data has shape of: {}'.format(dt.shape[1::]))
# CNN rules

# d = 100
# p = 0
# s = 1
# kk = [3, 2 2]
# np.floor((d - 2*p - k)/s) + 1

# # adapt this if using `channels_first` image data format

# input_dt = Input(shape=dt.shape[1::])
# x = Conv2D(6, (6, 3), activation='relu', data_format="channels_last",
#            padding='same')(input_dt)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(12, (6, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(30, (6, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((5, 5), padding='same')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = Conv2D(30, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((5, 5))(x)
# x = Conv2D(6, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

# autoencoder = Model(input_dt, decoded)
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# input_dt = Input(shape=dt.shape[1::])
# x = Conv2D(24, (6, 3), activation='relu', padding='same')(input_dt)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(12, (6, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(6, (6, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 1), padding='same')(x)

# x = Conv2D(6, (6, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 1))(x)
# x = Conv2D(12, (6, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(24, (6, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(3, (6, 3), activation='linear', padding='same')(x)

# # autoencoder = Model(input_dt, decoded)
# autoencoder = Model(input_dt, decoded)
# autoencoder.summary()
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

input_dt = Input(shape=dt.shape[1::])
x = SeparableConv2D(24, (6, 3), activation='selu',
                    padding='same', data_format='channels_last')(input_dt)
x = MaxPooling2D((2, 2), padding='same')(x)
x = SeparableConv2D(12, (6, 3), activation='selu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = SeparableConv2D(6, (6, 3), activation='selu', padding='same')(x)
encoded = MaxPooling2D((2, 1), padding='same')(x)

x = SeparableConv2D(6, (6, 3), activation='selu', padding='same')(encoded)
x = UpSampling2D((2, 1))(x)
x = SeparableConv2D(12, (6, 3), activation='selu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = SeparableConv2D(24, (6, 3), activation='selu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = SeparableConv2D(3, (6, 3), activation='linear', padding='same')(x)

# autoencoder = Model(input_dt, encoded)
# autoencoder.summary()

autoencoder = Model(input_dt, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(dt[slc_trn], dt[slc_trn],
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(dt[slc_vld], dt[slc_vld]))

autoencoder.save('tst_keras_14.h5')

# 7 o melhor
# 9 o melhor A tendência de aumentar o número de filtros melhora o resultado
# 11 o melhor mas devido ao aumento do número de épocas
# 12 o melhor além de ter sido otimizado com 50 epocas
# Try with normalization for all
# EStranho... Batch_size menores os valores são menores

autoencoder.evaluate(dt[slc_tst], dt[slc_tst])


dt_in = dt[10, :, :, :]
dt_out = autoencoder.predict(dt_in[np.newaxis, :])

plt.pcolormesh(dt_in[:, :, 2].T, rasterized=True)
plt.pcolormesh(dt_out[0, :, :, 2].T, rasterized=True)
