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
# # Convolutional Autoeconder with two decoders
# This code trains a simple autoencoder neural network using convolutional
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

x_data, y_data = format_data(dt, wd=2, var=2, get_y=True, cont=True)
x_data = x_data[:,0]

# Split data file
idxs = split(x_data.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)
# Slice data
x_train = x_data[slc_trn]
y_train = y_data[slc_trn]
x_val = x_data[slc_vld]
y_val = y_data[slc_vld]

# %%
# Autoencoder settings

# Activation function
act = "tanh"  # Convolutional layers activation function
act_lt = "tanh"  # Latent space layers activation function
# Number of filters of each layer
flt = [3, 9, 27]
# Filter size
flt_size = 5
# Strides of each layer
strd = [2, 2, 5]
# Latent space size
lt_sz = 50

# Training settings
opt = "adam"  # Optimizer
loss = "mse"
epochs = 60
batch_size = 64

# %%
# Build the autoencoder neural network
tf.keras.backend.clear_session()
flt_tp = (flt_size, flt_size)
conv_kwargs = dict(activation=act, padding="same")
# Encoder
inputs = layers.Input(shape=x_train.shape[1:])
e = layers.Conv2D(flt[0], flt_tp, strides=strd[0], **conv_kwargs)(inputs)
e = layers.Conv2D(flt[1], flt_tp, strides=strd[1], **conv_kwargs)(e)
e = layers.Conv2D(flt[2], flt_tp, strides=strd[2], **conv_kwargs)(e)

# Latent space
l = layers.Flatten()(e)
l = layers.Dense(lt_sz, activation=act_lt)(l)

# Latent to decoder same
dn_flt = flt[-1]
d_shp = (x_train.shape[1:-1] / np.prod(strd)).astype(int)
d_sz = np.prod(d_shp) * dn_flt
d = layers.Dense(d_sz, activation=act_lt)(l)
d = layers.Reshape(np.hstack((d_shp, dn_flt)))(d)
# Decoder same
d = layers.Conv2DTranspose(flt[-1], flt_tp, strides=strd[-1], **conv_kwargs)(d)
d = layers.Conv2DTranspose(flt[-2], flt_tp, strides=strd[-2], **conv_kwargs)(d)
d = layers.Conv2DTranspose(flt[-3], flt_tp, strides=strd[-3], **conv_kwargs)(d)
decoded = layers.Conv2DTranspose(
    x_train.shape[-1], flt_tp, activation="linear", padding="same", name="dec_same"
)(d)
# Decoder next step
dp = layers.Dense(d_sz, activation=act_lt)(l)
dp = layers.Reshape(np.hstack((d_shp, dn_flt)))(dp)
# Decoder same
dp = layers.Conv2DTranspose(flt[-1], flt_tp, strides=strd[-1], **conv_kwargs)(dp)
dp = layers.Conv2DTranspose(flt[-2], flt_tp, strides=strd[-2], **conv_kwargs)(dp)
dp = layers.Conv2DTranspose(flt[-3], flt_tp, strides=strd[-3], **conv_kwargs)(dp)
decoded_pred = layers.Conv2DTranspose(
    x_train.shape[-1], flt_tp, activation="linear", padding="same", name="dec_pred"
)(dp)

# Mount the autoencoder
ae = Model(inputs, [decoded, decoded_pred], name="Dual Convolutional Autoencoder")

# %%
# Show the architecture
ae.summary()

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
ae.compile(optimizer=opt, loss=loss)
hist = ae.fit(
    x_train,
    [x_train, y_train],
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, [x_val, y_val]),
    callbacks=[es],
)

# %%
# Convert the history to a Pandas dataframe
hist = pd.DataFrame(hist.history)
hist.index.name = "Epochs"

min_loss = hist[['val_loss', 'loss']].min()
# Plot training evolution
tit = "Validation loss: {:.3f} - Training loss: {:.3f}".format(*min_loss)
hist.plot(grid=True, title=tit)

# %%
# Test the trained neural network against the test dataset
x_test = x_data[slc_tst]
y_test = y_data[slc_tst]
loss = ae.evaluate(x_test, [x_test, y_test])
print("Test dataset loss - Same: {:.3f} - Next: {:.3f}".format(*loss[1:]))

global_loss = ae.evaluate(x_data, [x_data, y_data])
print("Entire dataset loss - Same: {:.3f} - Next: {:.3f}".format(*global_loss[1:]))

# %%
# Comparing the input and output of the autoencoder neural network
data_index = 634

# Slice the data
dt_in = x_data[data_index]
# Get the neural network output
dt_out, dt_out_nxt = ae.predict(dt_in[np.newaxis])
# Plot
alg = "Dual Convolutional Autoencoder"
plot_red_comp(dt_in, dt_out[0], 0, lt_sz, global_loss[1], alg)
plot_red_comp(y_data[data_index], dt_out_nxt[0], 0, lt_sz, global_loss[2], alg)

# %%
