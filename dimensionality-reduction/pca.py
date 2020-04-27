import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# from utils import slicer, split, plot_red_comp,format_data, flat_2d


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

def pca_eval(pca, data):
    # Flatten data
    shape = data.shape
    data = data.reshape((1, np.prod(data.shape)))
    # Reduce dimension
    reduced = pca.transform(data)
    # Recover data from the lower dimension
    rec_data = pca.inverse_transform(reduced)
    return rec_data.reshape(shape)

dt_fl = "nn_data.h5"
dt_dst = "scaled_data"

n_train = 0.8
n_valid = 0.1

# Open data file
f = h5py.File(dt_fl, "r")
dt = f[dt_dst]

# x_data = format_data(dt, wd=100)

# idxs = split(x_data.shape[0], n_train, n_valid)
# slc_trn, slc_vld, slc_tst = slicer(x_data.shape, idxs)

# trn = x_data[slc_trn[0]][0:2]
# vld = x_data[slc_vld[0]][0:2]

# # Flatten
# trn_flt = flat_2d(trn, 1)
# vld_flt = flat_2d(vld, 1)

idxs = split(dt.shape[0], n_train, n_valid)
slc_trn, slc_vld, slc_tst = slicer(dt.shape, idxs)

trn = dt[slc_trn]
vld = dt[slc_vld]

# Flatten
trn_flt = flat_2d(trn)
vld_flt = flat_2d(vld)

# Full PCA
pca = PCA()
pca.fit(trn_flt)

cum_exp = np.cumsum(pca.explained_variance_ratio_)

idx_95 = np.where(cum_exp > 0.95)[0][0]
idx_98 = np.where(cum_exp > 0.98)[0][0]
idx_99 = np.where(cum_exp > 0.99)[0][0]

plt.plot(cum_exp)
plt.grid(True)
plt.xlabel("Number of variables")
plt.ylabel("Cumulative explained variance ratio")
plt.plot([0, cum_exp.shape[0]-1], [0.95, 0.95], '--k', lw=1.5, alpha=0.75)
plt.plot([0, cum_exp.shape[0]-1], [0.98, 0.98], '--r', lw=1.5, alpha=0.75)
plt.plot([0, cum_exp.shape[0]-1], [0.99, 0.99], '--b', lw=1.5, alpha=0.75)
plt.legend(['Cumlative variance', '95%', '98%', '99%'])

shape = vld.shape
p95, cum_95exp, vld_95rec, mse_95 = pca_fit(idx_95 + 1, trn_flt, vld_flt, shape)
p98, cum_98exp, vld_98rec, mse_98 = pca_fit(idx_98 + 1, trn_flt, vld_flt, shape)
p99, cum_99exp, vld_99rec, mse_99 = pca_fit(idx_99 + 1, trn_flt, vld_flt, shape)

# vld = np.squeeze(vld)
# vld_95rec = np.squeeze(vld_95rec)
# vld_98rec = np.squeeze(vld_98rec)
# vld_99rec = np.squeeze(vld_99rec)

alg = "PCA"
i = 50
var = 1
# 95%
plot_red_comp(vld[i], vld_95rec[i], var,idx_95 + 1, mse_95, alg)
# 98%
plot_red_comp(vld[i], vld_98rec[i], var,idx_98 + 1, mse_98, alg)
# 99%
plot_red_comp(vld[i], vld_99rec[i], var,idx_99 + 1, mse_99, alg)

# trn_rec = pca_eval(p99, trn[50])
# plot_red_comp(trn[i], trn_rec,1, idx_99 + 1, mse_99, alg)

# Norm error
((np.sum(np.abs(vld[i] - vld_95rec[i])**2))**0.5)/(np.sum(np.abs(vld[i])**2))**0.5
