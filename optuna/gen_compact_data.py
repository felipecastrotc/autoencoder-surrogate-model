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
dt_shape = dt_compact.shape

# Compact dataset file
f_c = h5py.File('data_compact.h5', 'a')

dt_nm = f_mdl.split('.')[0]
f_c.create_dataset(dt_nm, dt_shape, dtype=float)

dt_c = f_c[dt_nm]
dt_c[()] = dt_compact
dt_c.attrs['idx'] = dt.attrs['idx']
dt_c.attrs['cases'] = dt.attrs['cases']

# Scaling data
dt_snm = dt_nm + '_scaled'
f_c.create_dataset(dt_snm, dt_shape, dtype=float)
dt_sc = f_c[dt_snm]

dt_sc.attrs['mean'] = np.mean(dt_c, axis=0)
dt_sc.attrs['std'] = np.std(dt_c, axis=0)

dt_sc[()] = (dt_c - dt_sc.attrs['mean'])/dt_sc.attrs['std']

dt_sc.attrs['cases'] = dt_c.attrs['cases']
dt_sc.attrs['idx'] = dt_c.attrs['idx']

f_c.close()