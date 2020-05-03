import glob
import os

import h5py
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras import backend, optimizers, regularizers
from keras.models import Model

import joblib
import optuna
from optuna.integration import KerasPruningCallback
from optuna.visualization import *
from utils import slicer, split
from utils_keras import loss_norm_error

# Model name
PREFIX = "model_ae-add_{}-"
SUFFIX = "{}.h5"


def objective(trial):

    # Open data file
    f = h5py.File(DT_FL, "r")
    dt = f[DT_DST]

    # Split data and get slices
    idxs = split(dt.shape[0], N_TRAIN, N_VALID)
    slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

    # Get data
    x_train = dt[slc_trn]
    x_val = dt[slc_vld]

    # Limits and options
    # Filters
    # flt_lm = [4, 128]
    flt_lm = [[4, 128], [4, 128], [4, 128]]
    # Kernel
    k_lm = [3, 5]
    # Regularizer
    l2_lm = [1e-7, 1e-3]
    # Activation functions
    act_opts = ["relu", "elu", "tanh", "linear"]
    # Latent space cfg
    lt_sz = [5, 150]
    lt_dv = [0.3, 0.7]
    # Learning rate
    lm_lr = [1e-5, 1e-2]

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=x_train.shape[1:])
    e = inputs
    # Encoder
    flt, k_sz, act, l2 = [], [], [], []
    strd = [2, 2, 5]
    # n_layers = trial.suggest_int("n_layers", 2, 3)
    n_layers = 3
    for i in range(n_layers):
        # Get values
        flt += [trial.suggest_int("n{}_flts".format(i), flt_lm[i][0], flt_lm[i][1])]
        k_sz += [trial.suggest_categorical("e{}_kernel_size".format(i), k_lm)]
        act += [trial.suggest_categorical("e{}_activation".format(i), act_opts)]
        l2 += [trial.suggest_loguniform("e{}_l2".format(i), l2_lm[0], l2_lm[1])]
        l2_reg = regularizers.l2(l=l2[-1])
        # l2_reg = regularizers.l2(l=0)
        # Set layer
        e = layers.Conv2D(
            flt[-1],
            (k_sz[-1], k_sz[-1]),
            strides=strd[i],
            activation=act[-1],
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
    act_lt = trial.suggest_categorical("lt_activation", act_opts)
    l2_lt = int(trial.suggest_loguniform("lt_l2", l2_lm[0], l2_lm[1]))
    l2_reg = regularizers.l2(l=l2_lt)
    sz_lt = trial.suggest_int("lt_sz", lt_sz[0], lt_sz[1])
    dv_lt = trial.suggest_uniform("lt_div", lt_dv[0], lt_dv[1])
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

    lr = trial.suggest_loguniform("lr", lm_lr[0], lm_lr[1])
    if lr > 0:
        k_opt = k_optf(learning_rate=lr)
    else:
        k_opt = k_optf()

    ae.compile(optimizer=k_opt, loss=loss_norm_error, metrics=["mse", loss_norm_error])

    batch_size = int(trial.suggest_uniform("batch_sz", 2, 32))
    ae.summary()
    hist = ae.fit(
        x_train,
        x_train,
        epochs=30,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, x_val),
        callbacks=[KerasPruningCallback(trial, "val_loss_norm_error")],
        verbose=1,
    )

    txt = PREFIX + SUFFIX
    ae.save(txt.format(RUN_VERSION, trial.number))
    return hist.history["val_loss_norm_error"][-1]


def clean_models(study):
    # Get best model
    bst = study.best_trial.number
    # Rename best model
    txt = PREFIX + SUFFIX
    nw_name = PREFIX.format(RUN_VERSION)[:-1] + ".h5"
    os.rename(txt.format(RUN_VERSION, bst), nw_name)
    # Remove the other models
    rm_mdls = glob.glob(PREFIX.format(RUN_VERSION) + "*")
    for mdl in rm_mdls:
        os.remove(mdl)


def main():
    # Use Optuna to performa a hyperparameter optimisation
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    # Start the optimisation process
    study.optimize(objective, n_trials=100, timeout=1600)
    # Keep only the best model
    clean_models(study)

    # Save Optuna study
    joblib.dump(study, study_nm.format(RUN_VERSION))


if __name__ == "__main__":
    # Study naming
    study_nm = "study_add_v{}.pkl"

    # File to be used
    DT_FL = "nn_data.h5"
    # Dataset to be used
    DT_DST = "scaled_data"

    # Split train test and validation datasets
    N_TRAIN = 0.8
    N_VALID = 0.1

    # Current search run
    RUN_VERSION = 6
    main()
