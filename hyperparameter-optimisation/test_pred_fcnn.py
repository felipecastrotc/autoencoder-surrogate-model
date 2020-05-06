# %%
import h5py
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model
import scipy.integrate as it
from keras import regularizers, optimizers
from keras.utils.generic_utils import get_custom_objects

from utils import plot_red_comp, slicer, split, format_data
from utils_keras import loss_norm_error, loss_norm_error_np

# Set custom keras functions
get_custom_objects().update({"loss_norm_error": loss_norm_error})

# %% [markdown]
# # Convolutional Autoeconder
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

# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# %%
# Selecting data
dt_fl = "data_compact.h5"
dt_dst = "model_ae-smp_4_scaled"

# The percentage for the test is implicit
n_train = 0.8
n_valid = 0.1

# Select the variable to train
# 0: Temperature - 1: Pressure - 2: Velocity - None: all
var = None

# %%
# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

y_data = np.empty_like(dt)
for idx in dt.attrs['idx']:
    y_data[idx[0]:idx[1]] = np.gradient(dt[idx[0]:idx[1]], 10, axis=0)

# Split data file
idxs = split(dt.shape[0], n_train, n_valid, test_last=dt.attrs['idx'])
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs, var=var)

# Slice data
x_train = dt[slc_trn]
y_train = y_data[slc_trn]
x_val = dt[slc_vld]
y_val = y_data[slc_vld]

# Convert the var into a slice
if var:
    slc = slice(var, var + 1)
else:
    slc = slice(var)
# %%
# FCNN settings
# Activation function
# act = "tanh"  # Convolutional layers activation function
act = "tanh"  # Convolutional layers activation function
# Number of filters of each layer
# n_n = [48]
n_n = [200]

# Training settings
opt = "adam"  # Optimizer
loss = loss_norm_error
epochs = 1000
batch_size = 32

# %%
# Build the autoencoder neural network
tf.keras.backend.clear_session()
# FCNN
inputs = layers.Input(shape=x_train.shape[1:])
d = inputs
for n in n_n:
    # l2_reg = regularizers.l2(l=0.0023538380930393826)
    # d = layers.Dense(n, activation=act, kernel_regularizer=l2_reg)(d)
    d = layers.Dense(n, activation=act)(d)
    d = layers.Dropout(0.4)(d)
dd = layers.Dense(x_train.shape[1], activation='linear')(d)

# Mount the FCNN
fcnn = Model(inputs, dd, name="FCNN")

# %%
# Show the architecture
fcnn.summary()

# %% [markdown]
# ## Callbacks
# Early stopping to stop training when the validation loss start to increase
# The patience term is a number of epochs to wait before stop. Also, the
# 'restore_best_weights' is used to restore the best model against the
# validation dataset. It is necessary as not always the best model against
# the validation dataset is the last neural network weights.

# %%
# Callbacks
monitor = "val_loss"
patience = int(epochs * 0.1)
es = EarlyStopping(
    monitor=monitor, mode="min", patience=patience, restore_best_weights=True
)

# %%
# Compile and train
# opt = optimizers.Adam(learning_rate=0.0035115275211751095)
fcnn.compile(optimizer=opt, loss=loss)
hist = fcnn.fit(
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

#%%
fcnn = load_model('model_fcnn_2.h5')

# %%
# Test the trained neural network against the test dataset
x_test = dt[slc_tst]
y_test = y_data[slc_tst]
loss = fcnn.evaluate(x_test, y_test)[2]
print("Test dataset loss: {:.3f}".format(loss))

global_loss = fcnn.evaluate(dt, y_data)[2]
print("Entire dataset loss: {:.3f}".format(global_loss))

# %%
# Comparing the input and output of the autoencoder neural network

def func(t, y):
    return fcnn.predict(y[np.newaxis])[0]

fcnn.predict()

idxs = dt.attrs['idx']
dtt = 10

k = 0
i = 50
j = 2
ii = idxs[k, 0]
t0, tf = dtt*i , dtt*(i+j)
y0 = dt[ii+i]
yf = dt[ii+i+j]
out = it.solve_ivp(func, [t0, tf], y0)
mse = np.mean((yf-out.y[:, -1])**2)
print(mse)

y0 = dt[ii+i]
dyy = y_data[ii+i+1] 
dyf = fcnn.predict(y0[np.newaxis])
np.mean((dyy - dyf)**2)

i = 50
k = 0
j = 10
ii = idxs[k, 0]
y0 = dt[ii+i][np.newaxis]
val = []
for jj in range(j):
    aux = y0.copy()
    for m in range(100):
        aux += fcnn.predict(y0)*10/100
    y0 = aux
    print(np.mean((dt[ii+i+jj+1] - y0)**2))


x = np.linspace(0, 5, 100)
dx = np.diff(x)[0]
y0 = 10
y = x**2 + y0
dy = np.gradient(y, dx)
plt.plot(x, dy)

plt.plot(x, np.cumsum(dy)*dx + y0)
plt.plot(x, y)

val = []
for i in range(len(dy)):
    val += [dy[i]*dx + y0]
    y0 = val[-1]

plt.plot(x, val)
plt.plot(x, y)

import matplotlib.pyplot as plt

# Slice the data
dt_in = dt[data_index, :, :, slc]
# Get the neural network output
dt_out = ae.predict(dt_in[np.newaxis])
# Plot
alg = "Convolutional Autoencoder"
plot_red_comp(dt_in, dt_out[0], 0, lt_sz, global_loss, alg)

# %%
