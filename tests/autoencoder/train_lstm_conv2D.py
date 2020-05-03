# %%
import h5py
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model

from utils import plot_red_comp, slicer, split, format_data

# %% [markdown]
# # Convolutional LSTM
# This code trains a simple LSTM neural network using convolutional
#  layers.The contraction and expansion of the implemented neural network used
# only convolutional layers. Therefore, it does not rely on maxpooling or
# upsampling layers. Instead, it was used strides to control the contraction
# and expansion of the neural network. Also, in the decoder part it was used a
# decovolutional process.
#
# For the latent space it was used a fully connected layer with an additional
# fully connected layer in sequence, to connect the latent space with the
# decoder convolutional layer.
#
# The neural network architecture with the activation function is stated below.

# %%
# Selecting data
dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

# The percentage for the test is implicit
n_train = 0.8
n_valid = 0.1

# Select the variable to train
# 0: Temperature - 1: Pressure - 2: Velocity - None: all
var = 2

# %%
# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

x_data, y_data = format_data(dt, wd=3, var=2, get_y=True, cont=True)

# Split data file
idxs = split(x_data.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)
# Slice data
x_train = x_data[slc_trn]
x_val = x_data[slc_vld]

slc_trn, slc_vld, slc_tst = slicer(y_data.shape, idxs)
y_train = y_data[slc_trn]
y_val = y_data[slc_vld]

# %%
# LSTM neural network settings

# Activation function
act = "tanh"  # Convolutional layers activation function
# Number of filters of each layer
flt = [20, 20, 20, 30]
# Filter size
flt_size = 5

# Training settings
opt = "adam"  # Optimizer
loss = "mse"
epochs = 60
batch_size = 16

# %%
# Build the LSTM neural network
tf.keras.backend.clear_session()
flt_tp = (flt_size, flt_size)
conv_kwargs = dict(kernel_size=flt_tp, padding="same")
# Encoder
inputs = layers.Input(shape=x_train.shape[1:])
x = layers.ConvLSTM2D(flt[0], return_sequences=True, **conv_kwargs)(inputs)
x = layers.ConvLSTM2D(flt[1], return_sequences=True, **conv_kwargs)(x)
x = layers.ConvLSTM2D(flt[2], return_sequences=False, **conv_kwargs)(x)
x = layers.Conv2D(flt[3], activation=act, **conv_kwargs)(x)
out = layers.Conv2D(x_train.shape[-1], flt_tp, activation="linear", padding="same")(x)

# Mount the LSTM
lstm = Model(inputs, out, name="LSTM neural network")

# %%
# Show the architecture
lstm.summary()

# %% [markdown]
# ## Callbacks
# Early stopping to stop training when the validation loss start to increase
# The patience term is a number of epochs to wait before stop. Also, the
# 'restore_best_weights' is used to restore the best model against the
# validation dataset. It is necessary as not always the best model against
# the validation dataset is the last neural network weights.

# %%
# Callbacks
monitor = "val_dec_pred_loss"
patience = int(epochs * 0.3)
es = EarlyStopping(
    monitor=monitor, mode="min", patience=patience, restore_best_weights=True
)

# %%
# Compile and train
lstm.compile(optimizer=opt, loss=loss)
hist = lstm.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, y_val),
    callbacks=[es],
)

# %%
# Convert the history to a Pandas dataframe
hist = pd.DataFrame(hist.history)
hist.index.name = "Epochs"

# Plot training evolution
tit = "Validation loss: {:.3f} - Training loss: {:.3f}".format(*hist.min())
hist.plot(grid=True, title=tit)

# %%
# Test the trained neural network against the test dataset
x_test = x_data[slc_tst]
y_test = y_data[slc_tst]
loss = lstm.evaluate(x_test, y_test)
print("Test dataset loss: {:.3f}".format(loss))

global_loss = lstm.evaluate(x_data, y_data)
print("Entire dataset loss: {:.3f}".format(global_loss))

# %%
# Comparing the input and output of the LSTM neural network
data_index = 634

# Slice the data
dt_in = x_data[data_index]
# Get the neural network output
dt_out= lstm.predict(dt_in[np.newaxis])
# Plot
alg = "Convolutional LSTM"
plot_red_comp(y_data[data_index], dt_out[0], 0, lt_sz, global_loss, alg)

# %%
