# read_vtk
import h5py
import numpy as np
import pyvista as p
import xmltodict
import matplotlib.pyplot as plt

# Read vtk files and store them in a hdf5 file

def read_vtk(save_file, case, path, filename, h=None, close=True):
    if not h:
        h = h5py.File(save_file, "w")

    # Open the .pvd file
    f = open(path + filename)
    data_paths = xmltodict.parse(f.read())
    f.close()
    # Get the list of tim steps
    data_paths = data_paths["VTKFile"]["Collection"]["DataSet"]

    for i in range(len(data_paths)):
        # Get the data step
        mesh = pv.read(path + data_paths[i]["@file"])

        # Merge the data blocks
        dt = mesh.combine()
        coord = np.array(dt.points)

        # Get unique coordinates
        unq, unq_idx = np.unique(coord, axis=0, return_index=True)
        # Sort the unique coordinates to get the step
        idx_sort = np.argsort(unq[:, 0], axis=0)
        # Get the steps inside the mesh
        stp = np.unique(np.diff(unq[idx_sort, 0]))
        stp = stp[stp > 0]
        # Get the proper round values. Prevent float representation issues
        stp_dec = np.log10(stp).round()
        stp = stp.round(int(stp_dec[np.abs(stp_dec).argmax()] * -1))
        # Get the largest stp
        stp = np.max(stp)

        # Transform the coordinates to index
        coord_idx = (coord / stp).astype(int)
        coord_idx[:, 0] = coord_idx[:, 0] - coord_idx[:, 0].min()
        coord_idx[:, 1] = coord_idx[:, 1] - coord_idx[:, 1].min()

        # Get number of dimensions
        n_dim = 0
        for name in dt.array_names:
            if len(dt[name].shape) == 1:
                n_dim += 1
            elif len(dt[name].shape) >= 1:
                n_dim += dt[name].shape[1]

        # Initialize the matrices
        dt_mtx = np.empty([coord_idx[:, 0].max() + 1, coord_idx[:, 1].max() + 1, n_dim])

        # Store the data
        dt_info = []
        for j, name in enumerate(dt.array_names):
            if len(dt[name].shape) > 1:
                # Iterate over the dimensions of the data
                for k in range(dt[name].shape[1]):
                    dt_mtx[coord_idx[:, 0], coord_idx[:, 1], j] = dt[name][:, k]
                    dt_info += [name]
            else:
                # Store the data
                dt_mtx[coord_idx[:, 0], coord_idx[:, 1], j] = dt[name]
                dt_info += [name]

        # Create the dataset
        if i == 0:
            # Determine the simulation shape
            dt_shape = tuple([len(data_paths)] + list(dt_mtx.shape))
            # Create an empty dataset with the dimension
            dt_st = h.create_dataset(case, dt_shape, dtype=float, compression="gzip")
            # Set the time step as a group property
            if len(data_paths) > 1:
                dt_st.attrs["time_step"] = int(data_paths[1]["@timestep"]) - int(
                    data_paths[0]["@timestep"]
                )
            else:
                dt_st.attrs["time_step"] = 0
            # Properties of case/n -> physical properties, step, spatial step
            dt_st.attrs["spatial_step_x"] = stp
            dt_st.attrs["spatial_step_y"] = stp
            dt_st.attrs["physical"] = dt_info
            dt_st[i, :, :, :] = dt_mtx
        else:
            dt_st[i, :, :, :] = dt_mtx

    if close:
        h.close()


# Data manipulation functions

def split(sz, n_train=0.8, n_valid=0.1, shuffle=True):
    # Split the data
    # Percentage for the test dataset
    n_test = 1 - n_train - n_valid
    # Generate an index array
    idx = np.array(range(sz))
    # Get the datasets indexes
    idx_tst = np.random.choice(idx, int(n_test * sz), replace=False)
    idx = np.setdiff1d(idx, idx_tst, assume_unique=True)

    idx_vld = np.random.choice(idx, int(n_valid * sz), replace=False)
    idx_trn = np.setdiff1d(idx, idx_vld, assume_unique=True)

    # # Shuffle the train dataset
    if shuffle:
        np.random.shuffle(idx_trn)

    return idx_trn, idx_vld, idx_tst


def slicer(shp, idxs):
    # Obtain a list of slicers to slice the data array according to the
    # selected data

    # It is assumed that the first dimension is the samples
    slc = []
    # Iterate over the datasets
    for idx in idxs:
        idx.sort()
        slc += [tuple([idx] + [slice(None)] * (len(shp) - 2) + [2])]
    return tuple(slc)


def format_data(dt, wd=20):
    # LSTM training window time
    # Get the simulation indexes
    idxs = dt.attrs["idx"]
    n_t_stp = np.min(np.diff(idxs))
    exp_fct = n_t_stp // wd
    x_data = np.empty((len(idxs) * exp_fct, wd - 1, *dt.shape[1:-1], 1))
    y_data = np.empty((len(idxs) * exp_fct, *dt.shape[1:-1], 1))
    # Fill the matrix (sample, time, x, y, z)
    for i, idx in enumerate(idxs):
        for j in range(exp_fct - 1):
            slc = slice(idx[0] + wd * j, idx[0] + wd * (j + 1) - 1)
            x_data[exp_fct * i + j, :, :, :, 0] = dt[slc, :, :, 2]
            y_data[exp_fct * i + j, :, :, 0] = dt[idx[0] + wd * (j + 1), :, :, 2]
    return x_data, y_data


# Sensitivity analysis

def gen_problem(params):

    n_vars = len(params)

    problem = {"num_vars": n_vars, "names": [], "bounds": []}

    for key in params.keys():
        problem["names"] += [key]
        if "type" in params[key].keys():
            bd = params[key]["bounds"]
            if params[key]["type"] == "int":
                bd[-1] += 1
            problem["bounds"] += [bd]
        else:
            problem["bounds"] += [params[key]["bounds"]]

    return problem


def proper_type(samples, params):
    # Convertion to the numpy data types
    # cvt = {"float": float, "int": int}
    cvt = {"float": "f8", "int": "i8"}
    # Create the dtypes
    # dtype = {key: cvt[params[key]["type"]] for key in params.keys()}
    dtype = [(key, cvt[params[key]["type"]]) for key in params.keys()]
    # Create a dataframe with the cases to train
    smp_st = np.array(list(zip(*samples.T)), dtype=dtype)
    # smp_df = pd.DataFrame(smp_sbl, columns=params.keys())
    # smp_df = smp_df.astype(dtype)
    return smp_st

# Plot

def plot_red_comp(original, reduced, n_dim, mse_global, alg='PCA'):
    # Calculate the MSE
    mse = np.mean((original - reduced) ** 2)
    # Generate the subplot figure
    fig, ax = plt.subplots(2, figsize=(8, 8))
    tit = "Global MSE: {:.4f}  Case MSE: {:.4f}".format(mse_global, mse)
    fig.suptitle(tit, y=1.02)
    fig.tight_layout(pad=2)
    ax[0].pcolormesh(original.T, rasterized=True)
    ax[1].pcolormesh(reduced.T, rasterized=True)
    ax[0].set_title("Original data")
    ax[1].set_title("{} with {} dimensions".format(alg, n_dim))
