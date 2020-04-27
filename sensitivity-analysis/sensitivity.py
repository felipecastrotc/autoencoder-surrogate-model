import json

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend, optimizers, regularizers
from keras.models import Model
from SALib.sample import latin, saltelli

from utils import gen_problem, proper_type, split, slicer

# https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/


def ae_add_model(x_train, y_train, x_val, y_val, params):

    act_fn = {0: "relu", 1: "tanh", 2: "sigmoid", 3: "elu", 4: "linear"}
    optm = {0: "adam", 1: "nadam", 2: "adamax"}

    # Get the number of layers
    lyrs = ["n1_filters", "n2_filters", "n3_filters"]
    n_lyrs = (np.array(params[lyrs].tolist()) > 0).sum()
    lyrs_sz = params[lyrs].tolist()

    # Split latent size
    latent_1 = int(params["latent_layer_size"] * params["latent_layer_div"])
    latent_2 = int(params["latent_layer_size"] - latent_1)

    # Activation
    # Reduction and expansion layers
    if 'act_latent_layers' in smp_st.dtype.names:
        act = act_fn[params["act_layers"]]
    else:
        act = 'tanh'
    # Latent space activation
    if 'act_latent_layers' in smp_st.dtype.names:
        act_lt = act_fn[params["act_latent_layers"]]
    else:
        act_lt = 'linear'

    # Regularisations
    if 'l2_reg' in smp_st.dtype.names:
        l2 = regularizers.l2(l=10 ** params["l2_reg"])
    else:
        l2 = None
    if 'dropout' in smp_st.dtype.names:
        dp = params["dropout"]
    else:
        dp = 0

    # Optimisation
    if 'learning_rate' in smp_st.dtype.names:
        lr = 10 ** params["learning_rate"]
    else:
        lr = 0
    if 'optm' in smp_st.dtype.names:
        opt = optm[params["optm"]]
    else:
        opt = 'adam'

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=x_train.shape[1:])
    e = inputs
    # Encoder
    cmp = [2, 2, 5]
    enc = []
    for i in range(n_lyrs):
        e = layers.Conv2D(
            lyrs_sz[i],
            (5, 5),
            strides=cmp[i],
            activation=act,
            padding="same",
            kernel_regularizer=l2,
            name="{}_encoder".format(i + 1),
        )(e)
        # Add layers
        if i == 0:
            ed = layers.Conv2D(
                1,
                (1, 1),
                padding="same",
                kernel_regularizer=l2,
                name="l2_input".format(i),
            )(e)
        # Dropout
        if dp > 0:
            e = layers.Dropout(dp, name="{}_dropout_encoder".format(i + 1))(e)

    # Latent space
    lt1 = layers.Flatten()(e)
    lt1 = layers.Dense(
        latent_1, activation=act_lt, kernel_regularizer=l2, name="l1_latent"
    )(lt1)

    lt2 = layers.Flatten()(ed)
    lt2 = layers.Dense(
        latent_2, activation=act_lt, kernel_regularizer=l2, name="l2_latent"
    )(lt2)

    # Dencoder
    # Flat input to the decoder
    n_flat = np.prod(backend.int_shape(e)[1:])
    d = layers.Dense(
        n_flat, activation=act_lt, kernel_regularizer=l2, name="l1_dense_decoder"
    )(lt1)
    # Consider uses only one filter with convolution
    # Reshape to the output of the encoder
    d = layers.Reshape(backend.int_shape(e)[1:])(d)
    # Generate the convolutional layers
    n_lyrs = 3
    for i in range(n_lyrs):
        # Settings index
        j = -i - 1
        # Add the latent space
        if i == n_lyrs - 1:
            d1 = layers.Dense(
                5000,
                activation="linear",
                kernel_regularizer=l2,
                name="l2_dense_decoder",
            )(lt2)
            d1 = layers.Reshape(backend.int_shape(ed)[1:], name="l2_reshape_decoder")(
                d1
            )
            d1 = layers.Conv2D(
                lyrs_sz[j + 1],
                (1, 1),
                padding="same",
                name="l2_compat_decoder",
                kernel_regularizer=l2,
            )(d1)
            d = layers.Add()([d1, d])
        # Convolutional layer
        d = layers.Conv2DTranspose(
            lyrs_sz[j],
            (5, 5),
            strides=cmp[j],
            activation=act,
            padding="same",
            kernel_regularizer=l2,
            name="{}_decoder".format(i + 1),
        )(d)
        # Dropout layers
        if dp > 0:
            d = layers.Dropout(dp, name="{}_dropout_decoder".format(i + 1))(d)

    decoded = layers.Conv2DTranspose(
        1,
        (5, 5),
        activation="linear",
        padding="same",
        kernel_regularizer=l2,
        name="output_decoder",
    )(d)

    ae = Model(inputs, decoded, name="auto_encoder_add")

    if opt == "adam":
        k_optf = optimizers.Adam
    elif opt == "nadam":
        k_optf = optimizers.Nadam
    elif opt == "adamax":
        k_optf = optimizers.Adamax
    if lr > 0:
        k_opt = k_optf(learning_rate=lr)
    else:
        k_opt = k_optf()

    ae.compile(optimizer=k_opt, loss="mse", metrics=["mse"])
    # return ae

    hist = ae.fit(
        x_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch"],
        shuffle=True,
        validation_data=(x_val, y_val),
    )
    # hist = None
    return hist, ae


params = dict(
    n1_filters={"bounds": [2, 64], "type": "int"},
    n2_filters={"bounds": [2, 64], "type": "int"},
    n3_filters={"bounds": [16, 64], "type": "int"},
    act_layers={"bounds": [0, 3], "type": "int"},
    latent_layer_div={"bounds": [0.5, 0.7], "type": "float"},
    latent_layer_size={"bounds": [120, 200], "type": "int"},
    epochs={"bounds": [15, 80], "type": "int"},
    batch={"bounds": [0, 32], "type": "int"},
)

# The Saltelli sampler generates N∗(2*D+2)
n_smp = 108  # multiple of 28

# Generate the SALib dictionary
problem = gen_problem(params)

# Get the number of samples  to input at Saltelli function
n_stl = int(n_smp / (2 * problem["num_vars"] + 2))
# Generate the samples
smp_sbl = saltelli.sample(problem, n_stl, True)

# Convert to structured array
smp_st = proper_type(smp_sbl, params)

# x_sbl = smp_sbl[:, 1]
# kw = {'marker': 'o', 'alpha': 0.7, 'ls': ''}
# plt.plot(x_sbl, smp_sbl[:, 1], **kw)
# plt.plot(x_sbl, smp_sbl[:, 2], **kw)
# plt.plot(x_sbl, smp_sbl[:, 3], **kw)
# plt.plot(x_sbl, smp_sbl[:, 4], **kw)
# plt.xlabel(problem['names'][0])
# plt.legend(problem['names'][1::])
# plt.title('Samples generated using Sobol QMC')

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

# Initialize dataframe to store the data
df = pd.DataFrame(smp_st)
df["val_loss"] = np.inf
df["val_mse"] = np.inf
df["mse"] = np.inf
df["loss"] = np.inf
df["hist"] = np.nan
df["hist"] = df["hist"].apply(lambda x: {})

# Initialize log
min_val = np.inf
min_trn = np.inf
# Train autoencoders
for i, smp in enumerate(smp_st):
    hist, ae = ae_add_model(trn, trn, vld, vld, smp)
    df.loc[i, "val_loss"] = min(hist.history["val_loss"])
    df.loc[i, "val_mse"] = min(hist.history["val_mse"])
    df.loc[i, "mse"] = min(hist.history["mse"])
    df.loc[i, "loss"] = min(hist.history["loss"])
    df.loc[i, "hist"].update(hist.history)
    if df.loc[i, "val_mse"] < min_val:
        ae.save("ae_add_min_val_mse.h5")
        min_val = df.loc[i, "val_mse"]
    if df.loc[i, "mse"] < min_trn:
        ae.save("ae_add_min_trn_mse.h5")
        min_trn = df.loc[i, "mse"]
    df.to_hdf("ae_add_salib_hist_2.h5", "log_sa", "a")
    print("{:.2%}%".format(i/smp_st.shape[0]))
