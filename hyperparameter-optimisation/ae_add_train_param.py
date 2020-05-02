import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend, optimizers, regularizers
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects

import joblib
from utils import plot_red_comp, slicer, split
from utils_keras import loss_norm_error


# Train best
def get_ae(prs, shape):

    # Initialize parameters
    flt = [prs["n0_flts"], prs["n1_flts"], prs["n2_flts"]]
    k_sz = [prs["e0_kernel_size"], prs["e1_kernel_size"], prs["e2_kernel_size"]]
    l2 = [prs["e0_l2"], prs["e1_l2"], prs["e2_l2"]]
    act = [prs["e0_activation"], prs["e1_activation"], prs["e2_activation"]]

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=shape)
    e = inputs
    # Encoder
    strd = [2, 2, 5]
    # n_layers = trial.suggest_int("n_layers", 2, 3)
    n_layers = 3
    for i in range(n_layers):
        # Get values
        l2_reg = regularizers.l2(l=l2[i])
        # Set layer
        e = layers.Conv2D(
            flt[i],
            (k_sz[i], k_sz[i]),
            strides=strd[i],
            activation=act[i],
            padding="same",
            kernel_regularizer=l2_reg,
            name="{}_encoder".format(i + 1),
        )(e)
        # Add layers
        if i == 0:
            ed = layers.Conv2D(
                1,
                (1, 1),
                padding="same",
                kernel_regularizer=l2_reg,
                name="l2_input".format(i),
            )(e)
        # Dropout
        dp = 0
        if dp > 0:
            e = layers.Dropout(dp, name="{}_dropout_encoder".format(i + 1))(e)

    # Latent space
    act_lt = prs["lt_activation"]
    l2_lt = int(prs["lt_l2"])
    sz_lt = prs["lt_sz"]
    dv_lt = prs["lt_div"]

    l2_reg = regularizers.l2(l=l2_lt)
    # Dense latent sizes
    latent_1 = int(sz_lt * dv_lt)
    latent_2 = sz_lt - latent_1

    lt1 = layers.Flatten()(e)
    lt1 = layers.Dense(
        latent_1, activation=act_lt, kernel_regularizer=l2_reg, name="l1_latent"
    )(lt1)

    lt2 = layers.Flatten()(ed)
    lt2 = layers.Dense(
        latent_2, activation=act_lt, kernel_regularizer=l2_reg, name="l2_latent"
    )(lt2)

    # Dencoder
    # Flat input to the decoder
    n_flat = np.prod(backend.int_shape(e)[1:])
    d = layers.Dense(
        n_flat, activation=act_lt, kernel_regularizer=l2_reg, name="l1_dense_decoder"
    )(lt1)
    # Consider uses only one filter with convolution
    # Reshape to the output of the encoder
    d = layers.Reshape(backend.int_shape(e)[1:])(d)
    # Generate the convolutional layers
    for i in range(n_layers):
        # Settings index
        j = -i - 1
        # Set the regularizer
        l2_reg = regularizers.l2(l=l2[j])
        # Add the latent space
        if i == n_layers - 1:
            d1 = layers.Dense(
                5000,
                activation="linear",
                kernel_regularizer=l2_reg,
                name="l2_dense_decoder",
            )(lt2)
            d1 = layers.Reshape(backend.int_shape(ed)[1:], name="l2_reshape_decoder")(
                d1
            )
            d1 = layers.Conv2D(
                flt[j + 1],
                (1, 1),
                padding="same",
                name="l2_compat_decoder",
                kernel_regularizer=l2_reg,
            )(d1)
            d = layers.Add()([d1, d])
        # Convolutional layer
        d = layers.Conv2DTranspose(
            flt[j],
            (k_sz[j], k_sz[j]),
            strides=strd[j],
            activation=act[j],
            padding="same",
            kernel_regularizer=l2_reg,
            name="{}_decoder".format(i + 1),
        )(d)
        # Dropout layers
        if dp > 0:
            d = layers.Dropout(dp, name="{}_dropout_decoder".format(i + 1))(d)

    decoded = layers.Conv2DTranspose(
        x_train.shape[-1],
        (5, 5),
        activation="linear",
        padding="same",
        kernel_regularizer=l2_reg,
        name="output_decoder",
    )(d)

    ae = Model(inputs, decoded, name="auto_encoder_add")

    opt = "adam"
    if opt == "adam":
        k_optf = optimizers.Adam
    elif opt == "nadam":
        k_optf = optimizers.Nadam
    elif opt == "adamax":
        k_optf = optimizers.Adamax

    lr = prs["lr"]
    if lr > 0:
        k_opt = k_optf(learning_rate=lr)
    else:
        k_opt = k_optf()

    ae.compile(optimizer=k_opt, loss=loss_norm_error, metrics=["mse", loss_norm_error])

    return ae


DT_FL = "nn_data.h5"
DT_DST = "scaled_data"

N_TRAIN = 0.8
N_VALID = 0.1

# Open data file
f = h5py.File(DT_FL, "r")
dt = f[DT_DST]

# Split data and get slices
idxs = split(dt.shape[0], N_TRAIN, N_VALID)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

# Get data
x_train = dt[slc_trn]
x_val = dt[slc_vld]

params = {
    "batch_sz": 2.07776644561671,
    "e0_activation": "relu",
    "e0_kernel_size": 5,
    "e0_l2": 0.0007485322502228407,
    "e1_activation": "linear",
    "e1_kernel_size": 5,
    "e1_l2": 4.2264527305479816e-07,
    "e2_activation": "linear",
    "e2_kernel_size": 3,
    "e2_l2": 0.0009153072585366238,
    "lr": 0.00011636759775213867,
    "lt_activation": "elu",
    "lt_div": 0.3002328103345342,
    "lt_l2": 5.62443449268462e-07,
    "lt_sz": 124,
    "n0_flts": 78,
    "n1_flts": 108,
    "n2_flts": 100,
}
batch_size = int(params["batch_sz"])

ae = get_ae(params, x_train.shape[1:])

ae.summary()
hist = ae.fit(
    x_train,
    x_train,
    epochs=15,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, x_val),
    verbose=1,
)

history = pd.DataFrame(hist.history)
history.plot()

i = 634
i = 99
var = 2
org = dt[i]
# org = trn[i]
rec = np.squeeze(ae.predict(org[np.newaxis, :]))
# ((org[:,:,var] - rec[:, :, var])/org[:,:,var]).min()
plot_red_comp(org, rec, var, params["lt_sz"], 0, "AE")


# Loaded Model
get_custom_objects().update({"loss_norm_error": loss_norm_error})
ae = load_model("model_ae-add_6.h5")
ae.summary()

lt_sz = ae.layers[7].output.shape[1]
lt_sz += ae.layers[8].output.shape[1]

i = 634
i = 1414
var = 2
org = dt[i]
# org = trn[i]
rec = np.squeeze(ae.predict(org[np.newaxis, :]))
# ((org[:,:,var] - rec[:, :, var])/org[:,:,var]).min()
plot_red_comp(org, rec, var, lt_sz, 0, "AE")

