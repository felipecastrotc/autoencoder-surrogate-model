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
from utils import format_data, slicer, split
from utils_keras import loss_norm_error

# Model name
PREFIX = "model_pred-lstm_{}-"
SUFFIX = "{}.h5"


def objective(trial):

    # Open data file
    f = h5py.File(DT_FL, "r")
    dt = f[DT_DST]

    # Format data for LSTM training
    x_data, y_data = format_data(dt, wd=WD, get_y=True)

    # Split data and get slices
    idxs = split(x_data.shape[0], N_TRAIN, N_VALID)
    slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)

    # Get data
    x_train = x_data[slc_trn[0]]
    y_train = y_data[slc_trn[0]]
    x_val = x_data[slc_vld[0]]
    y_val = y_data[slc_vld[0]]

    # Limits and options
    # Filters
    n_lstm = [[4, 128], [4, 128], [4, 128]]
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
    lm_lr = [1e-5, 1]

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=x_train.shape[1:])
    p = inputs
    # LSTM layers
    n_lyr_lstm = trial.suggest_int("n_lyr_lstm", 1, 3)
    for i in range(n_lyr_lstm):
        # For the current layer
        # Get number of filters
        l = trial.suggest_int("n{}_lstm".format(i), n_lstm[i][0], n_lstm[i][1])
        # Get the activation function
        act = trial.suggest_categorical("l{}_activation".format(i), act_opts)
        # Regularization value
        # l2 = trial.suggest_loguniform("l{}_l2".format(i), l2_lm[0], l2_lm[1])
        # l2_reg = regularizers.l2(l=l2)
        # Set layer
        p = layers.LSTM(
            l,
            activation=act,
            # kernel_regularizer=l2_reg,
            return_sequences=n_lyr_lstm - 1 != i,
            name="{}_lstm".format(i + 1),
        )(p)
        # Dropout
        # dp = trial.suggest_uniform("l{}_dropout".format(i), 0 ,1)
        # p = layers.Dropout(dp, name="{}_dropout_lstm".format(i + 1))(p)
        p = layers.BatchNormalization(name="{}_bnorm_lstm".format(i + 1))(p)

    # Dense layers
    n_lyr_dense = trial.suggest_int("n_lyr_dense", 0, 2)
    for i in range(n_lyr_dense):
        # For the current layer
        # Get number of filters
        l = trial.suggest_int("n{}_dense".format(i), n_lstm[i][0], n_lstm[i][1])
        # Get the activation function
        act = trial.suggest_categorical("d{}_activation".format(i), act_opts)
        # Regularization value
        # l2 = trial.suggest_loguniform("d{}_l2".format(i), l2_lm[0], l2_lm[1])
        # l2_reg = regularizers.l2(l=l2)
        # Set layer
        p = layers.Dense(
            l,
            activation=act,
            # kernel_regularizer=l2_reg,
            name="{}_dense".format(i + 1),
        )(p)
        # Dropout
        # dp = trial.suggest_uniform("d{}_dropout".format(i), 0 ,1)
        # p = layers.Dropout(dp, name="{}_dropout_dense".format(i + 1))(p)
        p = layers.BatchNormalization(name="{}_bnorm_dense".format(i + 1))(p)

    out = layers.Dense(x_data.shape[2], activation="linear")(p)

    pred = Model(inputs, out, name="auto_encoder_add")

    opt_opts = ["adam", "nadam", "adamax", "RMSprop"]
    opt = trial.suggest_categorical("optimizer", opt_opts)
    if opt == "adam":
        k_optf = optimizers.Adam
    elif opt == "nadam":
        k_optf = optimizers.Nadam
    elif opt == "adamax":
        k_optf = optimizers.Adamax
    elif opt == "RMSprop":
        k_optf = optimizers.RMSprop

    lr = trial.suggest_loguniform("lr", lm_lr[0], lm_lr[1])
    if lr > 0:
        k_opt = k_optf(learning_rate=lr)
    else:
        k_opt = k_optf()

    pred.compile(optimizer=k_opt, loss="mse", metrics=["mse", loss_norm_error])

    batch_size = int(trial.suggest_uniform("batch_sz", 2, 32))
    pred.summary()
    hist = pred.fit(
        x_train,
        y_train,
        epochs=300,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[KerasPruningCallback(trial, "val_mse")],
        verbose=1,
    )

    txt = PREFIX + SUFFIX
    pred.save(txt.format(RUN_VERSION, trial.number))
    return hist.history["val_mse"][-1]


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
    pass


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
    study_nm = "study_lstm_v{}.pkl"

    # File to be used
    DT_FL = "data_compact.h5"
    # Dataset to be used
    DT_DST = "model_ae-smp_4_scaled"

    # Split train test and validation datasets
    N_TRAIN = 0.8
    N_VALID = 0.1

    # Window size to be used to predict the next sample
    WD = 3

    # Current search run
    RUN_VERSION = 1

    main()
