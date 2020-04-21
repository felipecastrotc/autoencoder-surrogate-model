import json

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras import backend
from SALib.sample import latin, saltelli


def gen_problem(params):

    n_vars = len(params)

    problem = {"num_vars": n_vars, "names": [], "bounds": []}

    for key in params.keys():
        problem["names"] += [key]
        if "type" in params[key].keys():
            bd = params[key]["bounds"]
            if params[key]["type"] == "int":
                bd[-1] += 1
            problem["bounds"] += [bd]
        else:
            problem["bounds"] += [params[key]["bounds"]]

    return problem


def proper_type(samples, params):
    # Convertion to the numpy data types
    # cvt = {"float": float, "int": int}
    cvt = {"float": "f8", "int": "i8"}
    # Create the dtypes
    # dtype = {key: cvt[params[key]["type"]] for key in params.keys()}
    dtype = [(key, cvt[params[key]["type"]]) for key in params.keys()]
    # Create a dataframe with the cases to train
    smp_st = np.array(list(zip(*smp_sbl.T)), dtype=dtype)
    # smp_df = pd.DataFrame(smp_sbl, columns=params.keys())
    # smp_df = smp_df.astype(dtype)
    return smp_st


def ae_add_model(x_train, y_train, x_val, y_val, params):

    act_fn = {0: "relu", 1: "tanh", 2: "sigmoid", 3: "elu", 4: "linear"}
    optm = {0: "adam", 1: "nadam", 2: "adamax"}

    params = smp_st[0]
    # Get the number of layers
    lyrs = ["n1_filters", "n2_filters", "n3_filters"]
    n_lyrs = (np.array(params[lyrs].tolist()) > 0).sum()
    lyrs_sz = params[lyrs].tolist()

    # Split latent size
    latent_1 = int(params["latent_layer_size"] * params["latent_layer_div"])
    latent_2 = int(params["latent_layer_size"] - latent_1)

    # Activation
    # Reduction and expansion layers
    act = act_fn[params["act_layers"]]
    # Latent space activation
    act_lt = act_fn[params["act_latent_layers"]]

    # Regularisations
    l2 = 10 ** params["l2_reg"]
    dp = params["dropout"]

    # Optimisation
    lr = 10 ** params["learning_rate"]
    opt = params["optm"]

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
            name="{}_encoder".format(i + 1),
        )(e)
        # Add layers
        if i == 0:
            ed = layers.Conv2D(1, (1, 1), padding="same", name="l2_input".format(i))(e)
        # Dropout
        if dp > 0:
            e = layers.Dropout(dp, name="{}_dropout_encoder".format(i + 1))(e)

    # Latent space
    l1 = layers.Flatten()(e)
    l1 = layers.Dense(latent_1, activation=act_lt, name="l1_latent")(l1)

    l2 = layers.Flatten()(ed)
    l2 = layers.Dense(latent_2, activation=act_lt, name="l2_latent")(l2)

    # Dencoder
    # Flat input to the decoder
    n_flat = np.prod(backend.int_shape(e)[1:])
    d = layers.Dense(n_flat, activation=act_lt, name="l1_dense_decoder")(l1)
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
            d1 = layers.Dense(5000, activation="linear", name="l2_dense_decoder")(l2)
            d1 = layers.Reshape(backend.int_shape(ed)[1:], name="l2_reshape_decoder")(
                d1
            )
            d1 = layers.Conv2D(
                lyrs_sz[j + 1], (1, 1), padding="same", name="l2_compat_decoder"
            )(d1)
            d = layers.Add()([d1, d])
        # Convolutional layer
        d = layers.Conv2DTranspose(
            lyrs_sz[j],
            (5, 5),
            strides=cmp[j],
            activation=act,
            padding="same",
            name="{}_decoder".format(i + 1),
        )(d)
        # Dropout layers
        if dp > 0:
            d = layers.Dropout(dp, name="{}_dropout_decoder".format(i + 1))(d)

    decoded = layers.Conv2DTranspose(
        1, (5, 5), activation="linear", padding="same", name="output_decoder"
    )(d)

    ae = Model(inputs, decoded, name='auto_encoder_add')

    ae.compile(optimizer=optm[params["optm"]], loss="mse", metrics=["mse"])

    # hist = ae.fit(
    #     x_train,
    #     y_train,
    #     epochs=params["epochs"],
    #     batch_size=params["batch"],
    #     shuffle=True,
    #     validation_data=(x_val, y_val),
    # )

    # return hist, ae
    return ae


params = dict(
    n1_filters={"bounds": [2, 128], "type": "int"},
    n2_filters={"bounds": [2, 128], "type": "int"},
    n3_filters={"bounds": [0, 128], "type": "int"},
    act_layers={"bounds": [0, 3], "type": "int"},
    act_latent_layers={"bounds": [0, 4], "type": "int"},
    latent_layer_div={"bounds": [0, 1], "type": "float"},
    latent_layer_size={"bounds": [10, 300], "type": "int"},
    dropout={"bounds": [0, 3], "type": "float"},
    l2_reg={"bounds": [-8, -3], "type": "float"},
    learning_rate={"bounds": [-3, 0], "type": "float"},
    optm={"bounds": [0, 3], "type": "int"},
    epochs={"bounds": [15, 300], "type": "int"},
    batch={"bounds": [0, 64], "type": "int"},
)

# The Saltelli sampler generates N∗(2*D+2)
n_smp = 112  # multiple of 28

# Generate the SALib dictionary
problem = gen_problem(params)

# Get the number of samples  to input at Saltelli function
n_stl = int(n_smp / (2 * problem["num_vars"] + 2))
# Generate the samples
smp_sbl = saltelli.sample(problem, n_stl, True)

# Convert to structured array
smp_st = proper_type(smp_sbl, params)

# Train autoencoders
x = np.zeros((100, 200, 100, 1))
for smp in smp_st:
    hist, ae = ae_add_model(x, x, x, x, smp)





# # Graph X
# x_sbl = smp_sbl[:, 0]
# # Plot graphs
# kw = {'marker': 'o', 'alpha': 0.7, 'ls': ''}
# plt.plot(x_sbl, smp_sbl[:]['n_layers'], **kw)
# plt.plot(x_sbl, smp_sbl[:]['act_layers'], **kw)
# plt.plot(x_sbl, smp_sbl[:]['l2_reg'], **kw)
# plt.xlabel(problem['names'][0])
# plt.legend(problem['names'][1::])
# plt.title('Samples generated using Sobol QMC')


# https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/
