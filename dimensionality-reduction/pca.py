import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# from utils import slicer, split,plot_red_comp


def pca_fit(n_components, train, test, shape):
    # Set and fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(train)
    cum_exp = np.cumsum(pca.explained_variance_ratio_)
    # Reduce dimension
    test_reduced = pca.transform(test)
    # Recover data from the lower dimension
    test_recovered = pca.inverse_transform(test_reduced)
    # Calculate the MSE
    mse = np.mean((test_recovered - test) ** 2)
    # Reshape into a matrix
    test_recovered = test_recovered.reshape(shape)
    return pca, cum_exp, test_recovered, mse


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

# Full PCA
pca = PCA()
pca.fit(trn_flt)

cum_exp = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_exp)
plt.grid(True)
plt.xlabel("Number of variables")
plt.ylabel("Cummulative explained variance ratio")

idx_95 = np.where(cum_exp > 0.95)[0][0]
idx_98 = np.where(cum_exp > 0.98)[0][0]
idx_99 = np.where(cum_exp > 0.99)[0][0]

shape = dt[slc_vld].shape
_, cum_95exp, vld_95rec, mse_95 = pca_fit(idx_95 + 1, trn_flt, vld_flt, shape)
_, cum_98exp, vld_98rec, mse_98 = pca_fit(idx_98 + 1, trn_flt, vld_flt, shape)
_, cum_99exp, vld_99rec, mse_99 = pca_fit(idx_99 + 1, trn_flt, vld_flt, shape)

alg = "PCA"
i = 50
# 95%
plot_red_comp(vld[i], vld_95rec[i], idx_95 + 1, mse_95, alg)
# 98%
plot_red_comp(vld[i], vld_98rec[i], idx_98 + 1, mse_98, alg)
# 99%
plot_red_comp(vld[i], vld_99rec[i], idx_99 + 1, mse_99, alg)
