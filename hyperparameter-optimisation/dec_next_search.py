import glob
import os

import h5py
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras import backend, optimizers, regularizers
from keras.callbacks import EarlyStopping
from keras.models import Model

import joblib
import optuna
from optuna.integration import KerasPruningCallback
from optuna.visualization import *
from utils import format_data, slicer, split
from utils_keras import loss_norm_error

# Model name
PREFIX = "model_dec-nxt_{}-"
SUFFIX = "{}.h5"


def objective(trial):

    # Open data file
    f_in = h5py.File(DT_FL_IN, "r")
    dt_in = f_in[DT_DST_IN]

    f_out = h5py.File(DT_FL_OUT, "r")
    dt_out = f_out[DT_DST_OUT]

    WD = 2
    # Dummy y_data
    x_data, y_s = format_data(dt_in, wd=WD, get_y=True)
    _, y_data = format_data(dt_out, wd=WD, get_y=True)
    x_data = np.squeeze(x_data)

    # Split data and get slices
    idxs = split(x_data.shape[0], N_TRAIN, N_VALID)
    slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)

    # Get data
    x_train = x_data[slc_trn[0]]
    y_train = y_data[slc_trn[0]]
    x_val = x_data[slc_vld[0]]
    y_val = y_data[slc_vld[0]]

    conv_shape = y_train.shape[1:3]
    # Strides cfg
    strd = [2, 2, 5, 5]

    # Limits and options
    # Filters
    flt_lm = [[4, 128], [4, 128], [4, 128]]
    d_lm = [2, 100]
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
    lm_lr = [1e-5, 1e-1]

    # Clear tensorflow session
    tf.keras.backend.clear_session()
    # Input
    inputs = layers.Input(shape=x_train.shape[1:])
    d = inputs
    # Decoder
    n_layers = trial.suggest_int("n_layers", 1, 3)
    flt = trial.suggest_int("nl_flt", d_lm[0], d_lm[1])
    # Reduction from output
    red = np.prod(strd[:n_layers])
    # Decoder first shape
    lt_shp = (np.array(conv_shape) / red).astype(int)
    # Decoder dense size
    n_flat = np.prod(lt_shp) * flt
    # Format stride list
    strd = strd[::-1][-n_layers:]
    # Latent -> Decoder layer
    # Activation
    act_lt = trial.suggest_categorical("lt_activation", act_opts)
    # Regularization
    l2_lt = int(trial.suggest_loguniform("lt_l2", l2_lm[0], l2_lm[1]))
    l2_reg = regularizers.l2(l=l2_lt)
    # Flat input to the decoder
    d = layers.Dense(
        n_flat, activation=act_lt, kernel_regularizer=l2_reg, name="l1_dense_decoder"
    )(inputs)
    # Reshape to the output of the encoder
    d = layers.Reshape(list(lt_shp) + [flt])(d)
    # Generate the convolutional layers
    for i in range(n_layers):
        # Get number of filters
        flt = trial.suggest_int("n{}_flt".format(i), flt_lm[i][0], flt_lm[i][1])
        # Get the kernel size
        k_sz = trial.suggest_categorical("d{}_kernel_size".format(i), k_lm)
        # Get the activation function
        act = trial.suggest_categorical("d{}_activation".format(i), act_opts)
        # Regularization value
        l2 = trial.suggest_loguniform("d{}_l2".format(i), l2_lm[0], l2_lm[1])
        l2_reg = regularizers.l2(l=l2)
        # Convolutional layer
        d = layers.Conv2DTranspose(
            flt,
            (k_sz, k_sz),
            strides=strd[i],
            activation=act,
            padding="same",
            kernel_regularizer=l2_reg,
            name="{}_decoder".format(i + 1),
        )(d)
        dp = 0
        # Dropout layers
        if dp > 0:
            d = layers.Dropout(dp, name="{}_dropout_decoder".format(i + 1))(d)

    decoded = layers.Conv2DTranspose(
        y_train.shape[3],
        (5, 5),
        activation="linear",
        padding="same",
        name="output_decoder",
    )(d)

    ae = Model(inputs, decoded, name="Decoder_nxt")

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

    # es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=7)

    batch_size = int(trial.suggest_uniform("batch_sz", 2, 32))
    ae.summary()
    hist = ae.fit(
        x_train,
        y_train,
        epochs=90,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
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
    pass


def main():
    # Study naming
    study_nm = "study_dec-nxt_v{}.pkl"

    # File to be used as output
    DT_FL_OUT = "nn_data.h5"
    # Dataset to be used as output
    DT_DST_OUT = "scaled_data"

    # File to be used as input
    DT_FL_IN = "data_compact.h5"
    # Dataset to be used as input
    DT_DST_IN = "model_ae-smp_4_scaled"

    # Split train test and validation datasets
    N_TRAIN = 0.8
    N_VALID = 0.1

    # Current search run
    RUN_VERSION = 1

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
    main()
