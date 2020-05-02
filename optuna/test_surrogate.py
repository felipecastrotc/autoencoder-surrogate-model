import os

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend, optimizers, regularizers
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects

from utils import plot_red_comp, format_data
from utils_keras import loss_norm_error, loss_norm_error_np

# Set custom keras functions
get_custom_objects().update({"loss_norm_error": loss_norm_error})


# Original data
dt_fl = "nn_data.h5"
dt_dst = "scaled_data"
# Open dataset file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

# Latent data 
f_sc = h5py.File('data_compact.h5', 'r+')
dt_sc = f_sc['model_ae-smp_4_scaled']

# Load model
f_nxt = "model_dec-nxt_1.h5"
nxt = load_model(f_nxt)

# Encoder decoder to load 
f_mdl = "model_ae-smp_4.h5"
ae = load_model(f_mdl)

# Split encoder and decoder
inputs = layers.Input(shape=ae.layers[0].input_shape[1:])
enc_lyr = inputs
for layer in ae.layers[1:5]:
    enc_lyr = layer(enc_lyr)

enc = Model(inputs=inputs, outputs=enc_lyr)

# For latent space working surrogates
# inputs = layers.Input(shape=ae.layers[5].input_shape[1:])
# dec_lyr = inputs
# for layer in ae.layers[5:]:
#     dec_lyr = layer(dec_lyr)

# dec = Model(inputs=inputs, outputs=dec_lyr)
# dec.summary()

# i = 634
# data = dt_sc[[i, i+1]]
# out_sc = nxt.predict(data[np.newaxis])
# out_lt = out_sc*dt_sc.attrs['std'] + dt_sc.attrs['mean']

# plot_red_comp(dt[i+2], out, 2, 90, 0, 'Pred')

# mse = []
# for i in range(dt_sc.shape[0]-3):
#     data = dt_sc[[i, i+1]]
#     out_sc = nxt.predict(data[np.newaxis])
#     out_lt = out_sc*dt_sc.attrs['std'] + dt_sc.attrs['mean']
#     out = np.squeeze(dec.predict(out_lt))
#     mse += [np.mean((dt[i+2] - out)**2)]
# np.mean(mse)
# 0.3071

i = 3
data = dt_sc[i]
out = nxt.predict(data[np.newaxis])
plot_red_comp(dt[i+1], out[0], 2, 90, 0, 'Pred')

print(loss_norm_error_np(out, dt[i+1]))


import matplotlib.animation as manimation



pillow = manimation.writers['pillow']
metadata = dict(title='Decoder next step', artist='FCTC',
                comment='Decoder structure optimized')
writer = pillow(fps=15, metadata=metadata)


case = 8
alg = 'Decoder next'

# Calculate the simulation mse
slc = slice(*dt_sc.attrs['idx'][case])
out = nxt.predict(dt_sc[slc])
mse_sim = np.mean((out - dt[slc])**2)
mse_sim

# Create figure
# fig, ax = plt.subplots(2, figsize=(8, 8))

n_frames = np.diff(dt_sc.attrs['idx'][case])[0] - 1
idx_case = dt_sc.attrs['idx'][case][0]
# with writer.saving(fig, "writer_test.mp4", n_frames):
for i in range(n_frames):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    if i >= 0:
        out = nxt.predict(dt_sc[idx_case + i][np.newaxis])[0]
    else:
        encoded = enc.predict(out[np.newaxis])[0]
        encoded = (encoded - dt_sc.attrs['mean'])/dt_sc.attrs['std']
        out = nxt.predict(encoded[np.newaxis])[0]
    org = dt[idx_case + i + 1]
    mse = np.mean((out - org)**2)
    # Set figure title and layout
    tit = "Simulation MSE: {:.4f}  Current MSE: {:.4f}".format(mse_sim, mse)
    fig.suptitle(tit, y=1.02)
    # fig.tight_layout(pad=2)
    ax[0].pcolormesh(org[:,:, 2].T, rasterized=True)
    ax[1].pcolormesh(out[:,:, 2].T, rasterized=True)
    ax[0].set_title("Original data")
    ax[1].set_title("{} with {} dimensions".format(alg, dt_sc.shape[1]))
    plt.savefig('./frames/frame_{:03d}'.format(i))
    plt.close()