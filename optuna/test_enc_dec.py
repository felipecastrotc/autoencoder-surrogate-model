import os

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend, optimizers, regularizers
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects

from utils import plot_red_comp
from utils_keras import loss_norm_error, loss_norm_error_np

# Set custom keras functions
get_custom_objects().update({"loss_norm_error": loss_norm_error})

# Model to load
f_mdl = "model_ae-smp_4.h5"

# Trained dataset settings
dt_fl = "nn_data.h5"
dt_dst = "scaled_data"
# Open dataset file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

# Load model
ae = load_model(f_mdl)
ae.summary()

# Split encoder and decoder
inputs = layers.Input(shape=ae.layers[0].input_shape[1:])
enc_lyr = inputs
for layer in ae.layers[1:5]:
    enc_lyr = layer(enc_lyr)

enc = Model(inputs=inputs, outputs=enc_lyr)
enc.summary()

inputs = layers.Input(shape=ae.layers[5].input_shape[1:])
dec_lyr = inputs
for layer in ae.layers[5:]:
    dec_lyr = layer(dec_lyr)

dec = Model(inputs=inputs, outputs=dec_lyr)
dec.summary()

# Compact data
dt_compact = enc.predict(dt)

# Data to check
i = 634
cmp = dt_compact[i]

# Point analysis
plt.plot(cmp, 'o')
plt.grid(True)

# Image analysis
img = cmp.reshape((10, 9))
plt.pcolormesh(img, rasterized=True)

# Plot variable comparison
var = 2

org = dt[i]
lt_sz = cmp.shape[0]
# Recovered decoder
rec_dec = np.squeeze(dec.predict(cmp[np.newaxis, :]))
# Calculate loss
loss = loss_norm_error_np(org, rec_dec)
plot_red_comp(org, rec_dec, var, lt_sz, loss, "AE")

# Recovered full model
# rec = np.squeeze(ae.predict(org[np.newaxis, :]))
# Calculate loss
# loss = loss_norm_error_np(org, rec)
# plot_red_comp(org, rec, var, lt_sz, loss, "AE")
