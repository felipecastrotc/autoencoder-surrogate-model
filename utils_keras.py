import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

clear_tf = tf.keras.backend.clear_session

# Matrices multiplication
# y = np.transpose(X, axes=(0, 2, 1))
# np.einsum('ijk,ikm->ijm',x,y)
# X @ np.transpose(X, axes=(0, 2, 1))

# Regularizers
def orthogonal(w):
    w = K.squeeze(w, 2)
    wt = tf.transpose(w, perm=(2, 1, 0))
    w = tf.transpose(w, perm=(2, 0, 1))
    m = tf.einsum("ijk,ikm->ijm", wt, w) - K.eye(w.shape[1])
    return 0.01 * K.sqrt(K.sum(K.square(K.abs(m))))


# Custom activation functions
def swish(x, beta=1):
    return x * tf.math.sigmoid(beta * x)


def set_swish():
    get_custom_objects().update({"swish": layers.Activation(swish)})
    pass


# Norm error loss
def loss_norm_error(y_true, y_predicted):
    return K.sqrt(
        K.sum(K.square(K.abs(y_true - y_predicted))) / K.sum(K.square(K.abs(y_true)))
    )


def loss_norm_error_np(y_true, y_predicted):
    return np.sqrt(
        np.sum(np.square(np.abs(y_true - y_predicted)))
        / np.sum(np.square(np.abs(y_true)))
    )


# Mobile net block
def mbconv(inputs, expansion, stride, filters, alpha=1, block_id=0):

    in_nfilters = backend.int_shape(inputs)[-1]
    out_nfilters = int(alpha * filters)

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    # Expand
    x = layers.Conv2D(
        int(in_nfilters * expansion),
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        activation=None,
        name="{}_expand".format(block_id),
    )(inputs)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_expand_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Activation("relu", name="{}_expand_act".format(block_id))(x)

    # Depthwise
    if stride > 1:
        depth_padding = "valid"
        x = layers.ZeroPadding2D(padding=correct_pad(x, 3))(x)
    else:
        depth_padding = "same"

    x = layers.DepthwiseConv2D(
        (3, 3),
        strides=stride,
        padding=depth_padding,
        activation=None,
        use_bias=False,
        name="{}_depth".format(block_id),
    )(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_depth_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Activation("relu", name="{}_depth_act".format(block_id))(x)

    # Project
    x = layers.Conv2D(
        out_nfilters,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        activation=None,
        name="{}_project".format(block_id),
    )(x)
    # This layers does not have an activation function, it is linear
    # x = layers.BatchNormalization(axis=channel_axis,
    #                              name='{}_project_bn'.format(block_id))(x)
    x = layers.Dropout(0.25)(x)
    if in_nfilters == out_nfilters and stride == 1:
        # Sum the output layers with the input layer
        return layers.Add(name="{}_add".format(block_id))([inputs, x])
    return x
