#%%
import glob
import os

import h5py
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import backend, optimizers, regularizers
from keras.models import Model

import joblib
import optuna
from optuna.integration import KerasPruningCallback
from optuna.visualization import *
from utils import slicer, split
from utils_keras import loss_norm_error

#%%
PREFIX = "model_fcnn_{}-"
SUFFIX = "{}.h5"


def objective(trial):

    # Open data file
    f = h5py.File(DT_FL, "r")
    dt = f[DT_DST]

    y_data = np.empty_like(dt)
    for idx in dt.attrs['idx']:
        y_data[idx[0]:idx[1]] = np.gradient(dt[idx[0]:idx[1]], 10, axis=0)

    # Split data file
    idxs = split(dt.shape[0], N_TRAIN, N_VALID, test_last=dt.attrs['idx'])
    slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

    # Slice data
    x_train = dt[slc_trn]
    y_train = y_data[slc_trn]
    x_val = dt[slc_vld]
    y_val = y_data[slc_vld]

    # Limits and options
    epochs = 500
    # Filters
    n_n = [[30, 150], [30, 150]]
    # Regularizer
    l2_lm = [1e-7, 1e-2]
    # Activation functions
    act_opts = ["relu", "elu", "tanh", "linear"]
    # Learning rate
    lm_lr = [1e-5, 1e-1]

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=x_train.shape[1:])
    d = inputs
    # FCNN
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for i in range(n_layers):
        # For the current layer
        # Get number of filters
        n = trial.suggest_int("l{}_n_neurons".format(i), n_n[i][0], n_n[i][1])
        # Get the activation function
        act = trial.suggest_categorical("l{}_activation".format(i), act_opts)
        # Regularization value
        l2 = trial.suggest_loguniform("l{}_l2".format(i), l2_lm[0], l2_lm[1])
        l2_reg = regularizers.l2(l=l2)
        # Set layer
        d = layers.Dense(
            n,
            activation=act,
            kernel_regularizer=l2_reg,
            name="l{}_fc".format(i),
        )(d)
    dd = layers.Dense(x_train.shape[1], activation='linear')(d)

    fcnn = Model(inputs, dd, name="FCNN")

    monitor = "val_loss_norm_error"
    patience = int(epochs * 0.1)
    es = EarlyStopping(
        monitor=monitor, mode="min", patience=patience, restore_best_weights=True
    )


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

    fcnn.compile(optimizer=k_opt, loss=loss_norm_error, metrics=["mse", loss_norm_error])

    batch_size = int(trial.suggest_uniform("batch_sz", 2, 32))
    fcnn.summary()
    hist = fcnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=[KerasPruningCallback(trial, "val_loss_norm_error"), es],
        verbose=1,
    )
    
    txt = PREFIX + SUFFIX
    fcnn.save(txt.format(RUN_VERSION, trial.number))
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
    pass

#%%
DT_FL = "data_compact.h5"
DT_DST = "model_ae-smp_4_scaled"

N_TRAIN = 0.8
N_VALID = 0.1

#%%
RUN_VERSION = 1

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

study.optimize(objective, n_trials=200, timeout=2400)
clean_models(study)

joblib.dump(study, "study_fcnn_v{}.pkl".format(RUN_VERSION))


# %%
