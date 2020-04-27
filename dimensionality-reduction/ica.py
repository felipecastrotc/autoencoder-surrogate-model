import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA

from utils import slicer, split, plot_red_comp


def fast_ICA_fit(n_components, train, test, shape):
    # Set and fit FastICA
    fica = FastICA(n_components=n_components)
    fica.fit(train)
    # Reduce dimension
    test_reduced = fica.transform(test)
    # Recover data from the lower dimension
    test_recovered = fica.inverse_transform(test_reduced)
    # Calculate the MSE
    mse = np.mean((test_recovered - test) ** 2)
    # Reshape into a matrix
    test_recovered = test_recovered.reshape(shape)
    return fica, test_recovered, mse


dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

idxs = split(dt.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

trn = dt[slc_trn]
vld = dt[slc_vld]

# Flatten
trn_flt = trn.reshape((trn.shape[0], np.prod(trn.shape[1:])))
vld_flt = vld.reshape((vld.shape[0], np.prod(vld.shape[1:])))

# Get the number of dimensions for 95%, 98% and 99% of the variance
# explained using PCA, to compare with the Kernel PCA

# Full PCA
pca = PCA()
pca.fit(trn_flt)
cum_exp = np.cumsum(pca.explained_variance_ratio_)
# Number of dimensions to compare
n_95 = np.where(cum_exp > 0.95)[0][0] + 1
n_98 = np.where(cum_exp > 0.98)[0][0] + 1
n_99 = np.where(cum_exp > 0.99)[0][0] + 1

# Fast ICA
shape = dt[slc_vld].shape
_, vld_95rec, mse_95 = fast_ICA_fit(n_95, trn_flt, vld_flt, shape)
_, vld_98rec, mse_98 = fast_ICA_fit(n_98, trn_flt, vld_flt, shape)
_, vld_99rec, mse_99 = fast_ICA_fit(n_99, trn_flt, vld_flt, shape)

alg = "Fast ICA"
i = 50
# 95%
plot_red_comp(vld[i], vld_95rec[i], n_95, mse_95, alg)
# 98%
plot_red_comp(vld[i], vld_98rec[i], n_98, mse_98, alg)
# 99%
plot_red_comp(vld[i], vld_99rec[i], n_99, mse_99, alg)
