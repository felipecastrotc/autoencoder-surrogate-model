import h5py
import keras.layers as layers
import numpy as np
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects

from utils_keras import loss_norm_error

# Set custom keras functions
get_custom_objects().update({"loss_norm_error": loss_norm_error})


def main(f_mdl, dt_cfl, dt_fl, dt_dst, plot_arch=False):
    
    # Open dataset file
    f = h5py.File(dt_fl, "r")
    dt = f[dt_dst]

    print("Loading model...")
    # Load model
    ae = load_model(f_mdl)
    if plot_arch:
        ae.summary()

    # Split encoder and decoder
    inputs = layers.Input(shape=ae.layers[0].input_shape[1:])
    enc_lyr = inputs
    for layer in ae.layers[1:5]:
        enc_lyr = layer(enc_lyr)

    enc = Model(inputs=inputs, outputs=enc_lyr)
    if plot_arch:
        enc.summary()

    inputs = layers.Input(shape=ae.layers[5].input_shape[1:])
    dec_lyr = inputs
    for layer in ae.layers[5:]:
        dec_lyr = layer(dec_lyr)

    dec = Model(inputs=inputs, outputs=dec_lyr)
    if plot_arch:
        dec.summary()

    print("Generating latent data...")
    # Compact data
    dt_compact = enc.predict(dt)
    dt_shape = dt_compact.shape

    print("Preprocessing latent data...")
    # Compact dataset file
    f_c = h5py.File(dt_cfl, "a")

    dt_nm = f_mdl.split(".")[0]
    f_c.create_dataset(dt_nm, dt_shape, dtype=float)

    dt_c = f_c[dt_nm]
    dt_c[()] = dt_compact
    dt_c.attrs["idx"] = dt.attrs["idx"]
    dt_c.attrs["cases"] = dt.attrs["cases"]

    # Scaling data
    dt_snm = dt_nm + "_scaled"
    f_c.create_dataset(dt_snm, dt_shape, dtype=float)
    dt_sc = f_c[dt_snm]

    dt_sc.attrs["mean"] = np.mean(dt_c, axis=0)
    dt_sc.attrs["std"] = np.std(dt_c, axis=0)

    print("Storing latent data...")
    dt_sc[()] = (dt_c - dt_sc.attrs["mean"]) / dt_sc.attrs["std"]

    dt_sc.attrs["cases"] = dt_c.attrs["cases"]
    dt_sc.attrs["idx"] = dt_c.attrs["idx"]

    f_c.close()


if __name__ == "__main__":

    import sys

    # Get arguments
    args =  sys.argv[1:]

    # Model to load
    if not args:
        f_mdl = args.pop(0)
    else:
        f_mdl = "model_ae-smp_4.h5"

    # Data compated dataset
    if not args:
        dt_cfl = args.pop(0)
    else:
        dt_cfl = "data_compact.h5"

    # Trained dataset settings
    if not args:
        dt_fl = args.pop(0)
    else:
        dt_fl = "nn_data.h5"

    if not args:
        dt_dst = args.pop(0)
    else:
        dt_dst = "scaled_data"

    main(f_mdl, dt_cfl, dt_fl, dt_dst, plot_arch=False)
