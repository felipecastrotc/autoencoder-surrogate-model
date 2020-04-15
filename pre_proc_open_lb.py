import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

# cd simulations
# rm nn_data.h5

f = h5py.File('data.h5', 'r')
kys = list(f.keys())

# Get the time dimension size Check for inf and nan values
sz_tm = 0
kys_vld = []
status = False
for i, ky in enumerate(kys):
    # Determine a view from the dataset
    dt_view = f[ky][()]
    # Check for undefined values
    status = status | (np.isnan(dt_view).sum() > 0)
    status = status | (np.isneginf(dt_view).sum() > 0)
    status = status | (np.isposinf(dt_view).sum() > 0)
    # Store only valid samples
    if not status:
        kys_vld += [ky]
        sz_tm += f[ky].shape[0]
    status = False

# Get the shape of the dataset including all data
dt_shape = tuple([sz_tm] + list(f[ky].shape[1::]))

# Initialize the HDF5 with empty values
fw = h5py.File('nn_data.h5', 'w')
dt_nm = 'unscaled_data'
fw.create_dataset(dt_nm, dt_shape, dtype=float)

# Store unscaled data
idx = [0, 0]
idx_lst = []
for i, ky in enumerate(kys_vld):
    # Determine a view from the dataset
    dt_view = f[ky][()]
    # Set the indexes values
    idx[0] = idx[1]
    idx[1] = idx[0] + dt_view.shape[0]
    print(i, idx)
    fw[dt_nm][idx[0]:idx[1], :, :, :] = dt_view
    idx_lst.append(idx.copy())
    
fw[dt_nm].attrs['cases'] = kys_vld
fw[dt_nm].attrs['idx'] = idx_lst

fw.close()
f.close()
# rm nn_data.h5

f = h5py.File('nn_data.h5', 'a')
dt_uns = 'unscaled_data'
dt_scl = 'scaled_data'

# From the test matrix the temperature won't be above of:
T_lmt = np.array([0, 100]) + 273.15    # K

# Define the scaling type
proc_type = {0: 'std', 1: 'std', 2: 'std'}

# Reformating data, it is interesting to set the dimensions to be even
# In this case the dimensions were reset to be 200x100. Remove the last
# data, it is empty.
slc = [slice(None), slice(2, -1), slice(2, -1), [0, 1, 2]]
shp = f[dt_uns][tuple(slc)].shape

# Initialize dataset
f.create_dataset(dt_scl, shape=shp)
f[dt_scl].attrs['cases'] = f[dt_uns].attrs['cases']
f[dt_scl].attrs['idx'] = f[dt_uns].attrs['idx']
# Unscaled data view
dt = f[dt_uns]

idx_slc = [slice(None)]*(len(shp) - 1)
for i in slc[-1]:
    # Slice for the unscaled data
    slc_uns = tuple(slc[0:-1] + [i])
    # Slice for the scaled data
    slc_i = tuple(idx_slc + [i])
    if proc_type[i] == 'std':
        # Standardization
        mean = dt[slc_uns].mean(axis=0)
        std = dt[slc_uns].std(axis=0)
        # Standardize
        f[dt_scl][slc_i] = (dt[slc_uns][()] - mean)/(std + 1e-5)
    elif proc_type[i] == 'norm':
        # Normalize
        f[dt_scl][slc_i] = (dt[slc_uns][()] - T_lmt[0])/(T_lmt[1] - T_lmt[0])

f.close()
