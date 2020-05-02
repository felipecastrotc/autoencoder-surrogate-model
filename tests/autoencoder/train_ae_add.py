# %%
import h5py
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model

from utils import plot_red_comp, slicer, split

# %% [markdown]
# # VQ-VAE-2 Autoeconder
# This code trains a neural network based on the VQ-VAE-2 autoencoder available
# at [ArXiv](https://arxiv.org/pdf/1906.00446.pdf). This autoencoder was
# developed to generate diverse high-fidelity images that competes with the
# state of the art algorithms, generative adversarial networks (GAN), in
# problems to generete images from a probability distribution.
#
# The main characteristic from the VQ-VAE-2 architecture is the use of vector
# quantization and the use of multiple latent spaces. In the vector quantization,
# the latent space is quantized into a codebook of a given size. In the multiple
# latent one is obtained from a less compressed state and the other from a
# further compressed state. Then, each latent space is decoded and they are
# added when their shapes are equal.
#
# The vector quantization is particularly interesting for image generation
# as it can be used for density estimation. Thus, it can be randomly sampled
# and then generate random images. As this property has limited use for 
# surrogate models. This implementation only used the multi latent spaces 
# proposed by the paper.
#
# The contraction and expansion of the implemented neural network used
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

# Split data file
idxs = split(dt.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs, var=var)
# Slice data
x_train = dt[slc_trn][:, :, :, np.newaxis]
x_val = dt[slc_vld][:, :, :, np.newaxis]

# Convert the var into a slice
if var:
    slc = slice(var, var + 1)
else:
    slc = slice(var)
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
lt_sz = [25, 25]

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
ed = layers.Conv2D(1, (1, 1), padding="same")(e)
e = layers.Conv2D(flt[1], flt_tp, strides=strd[1], **conv_kwargs)(e)
e = layers.Conv2D(flt[2], flt_tp, strides=strd[2], **conv_kwargs)(e)

# Latent space
l1 = layers.Flatten()(e)
l1 = layers.Dense(lt_sz[0], activation=act_lt)(l1)

l2 = layers.Flatten()(ed)
l2 = layers.Dense(lt_sz[1], activation=act_lt)(l2)

# Latent to decoder
dn_flt = flt[-1]
d_shp = (x_train.shape[1:-1] / np.prod(strd)).astype(int)
d_sz = np.prod(d_shp) * dn_flt
d = layers.Dense(d_sz, activation=act_lt)(l1)
d = layers.Reshape(np.hstack((d_shp, dn_flt)))(d)
# Decoder
d = layers.Conv2DTranspose(flt[-1], flt_tp, strides=strd[-1], **conv_kwargs)(d)
d = layers.Conv2DTranspose(flt[-2], flt_tp, strides=strd[-2], **conv_kwargs)(d)
# Add latent 2
d1n_shp = (x_train.shape[1:-1] / np.array(strd[0])).astype(int)
d2 = layers.Dense(np.prod(d1n_shp), activation="linear")(l2)
d2 = layers.Reshape(np.hstack([d1n_shp, [1]]))(d2)
d2 = layers.Conv2D(flt[-2], (1, 1), padding="same")(d2)
d = layers.Add()([d2, d])
# Back to decoder
d = layers.Conv2DTranspose(flt[-3], flt_tp, strides=strd[-3], **conv_kwargs)(d)
decoded = layers.Conv2DTranspose(
    x_train.shape[-1], flt_tp, activation="linear", padding="same"
)(d)

# Mount the autoencoder
ae = Model(inputs, decoded, name="Based on VQ-VAE 2")

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
monitor = "val_loss"
patience = int(epochs * 0.3)
es = EarlyStopping(
    monitor=monitor, mode="min", patience=patience, restore_best_weights=True
)

# %%
# Compile and train
ae.compile(optimizer=opt, loss=loss)
hist = ae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_val, x_val),
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
x_test = dt[slc_tst][:, :, :, np.newaxis]
loss = ae.evaluate(x_test, x_test)
print("Test dataset loss: {:.3f}".format(loss))

global_loss = ae.evaluate(dt[:, :, :, slc], dt[:, :, :, slc])
print("Entire dataset loss: {:.3f}".format(global_loss))

# %%
# Comparing the input and output of the autoencoder neural network
data_index = 634

# Slice the data
dt_in = dt[data_index, :, :, slc]
# Get the neural network output
dt_out = ae.predict(dt_in[np.newaxis])
# Plot
alg = "VQ-VAE-2 based autoencoder"
plot_red_comp(dt_in, dt_out[0], 0, lt_sz, global_loss, alg)

# %%
