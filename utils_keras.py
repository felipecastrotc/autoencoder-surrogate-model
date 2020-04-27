import keras.layers as layers
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
