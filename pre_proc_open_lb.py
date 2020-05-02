import h5py
import numpy as np


def main(dt_fl, dt_trn):

    # Dataset key names for the processed dataset
    dt_uns = "unscaled_data"
    dt_scl = "scaled_data"

    # Open the dataset file
    f = h5py.File(dt_fl, "r")
    kys = list(f.keys())

    # Get the time dimension size Check for inf and nan values
    # The dataset with any inf and nan values will be remove from
    # the analysis.
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
    dt_shape = tuple([sz_tm] + list(f[kys[-1]].shape[1::]))

    # Initialize the HDF5 with empty values
    fw = h5py.File(dt_trn, "w")
    fw.create_dataset(dt_uns, dt_shape, dtype=float)

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
        fw[dt_uns][idx[0] : idx[1], :, :, :] = dt_view
        idx_lst.append(idx.copy())

    fw[dt_uns].attrs["cases"] = kys_vld
    fw[dt_uns].attrs["idx"] = idx_lst

    fw.close()
    f.close()

    # Scalling data
    f = h5py.File(dt_trn, "a")

    # From the test matrix the temperature won't be above of:
    T_lmt = np.array([0, 100]) + 273.15  # K

    # Define the scaling type
    proc_type = {0: "std", 1: "std", 2: "std"}

    # Reformating data, it is interesting to set the dimensions to be even
    # In this case the dimensions were reset to be 200x100. Remove the last
    # data, it is empty.
    slc = [slice(None), slice(2, -1), slice(2, -1), [0, 1, 2]]
    shp = f[dt_uns][tuple(slc)].shape

    # Initialize dataset
    f.create_dataset(dt_scl, shape=shp)
    f[dt_scl].attrs["cases"] = f[dt_uns].attrs["cases"]
    f[dt_scl].attrs["idx"] = f[dt_uns].attrs["idx"]
    # Unscaled data view
    dt = f[dt_uns]

    idx_slc = [slice(None)] * (len(shp) - 1)
    for i in slc[-1]:
        # Slice for the unscaled data
        slc_uns = tuple(slc[0:-1] + [i])
        # Slice for the scaled data
        slc_i = tuple(idx_slc + [i])
        if proc_type[i] == "std":
            # Standardization
            mean = dt[slc_uns].mean(axis=0)
            std = dt[slc_uns].std(axis=0)
            # Standardize
            # It was found that in some cases the standard deviation was 0.
            # Thus, it was added a small value to prevent zero division.
            f[dt_scl][slc_i] = (dt[slc_uns][()] - mean) / (std + 1e-5)
        elif proc_type[i] == "norm":
            # Normalize
            f[dt_scl][slc_i] = (dt[slc_uns][()] - T_lmt[0]) / (T_lmt[1] - T_lmt[0])

    f.close()


if __name__ == "__main__":

    import sys

    # Get arguments
    args = sys.argv[1:]

    # Model to load
    if not args:
        dt_fl = args.pop(0)
    else:
        dt_fl = "data.h5"

    # Data compated dataset
    if not args:
        dt_trn = args.pop(0)
    else:
        dt_trn = "nn_data.h5"

    main(dt_fl, dt_trn)
