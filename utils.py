import h5py
import matplotlib.pyplot as plt
import numpy as np

# Read vtk files and store them in a hdf5 file
def read_vtk(save_file, case, path, filename, h=None, close=True):
    """Read the .vtk file and store it in a HDF5 file

    Parameters
    ----------
    save_file : str
        HDF5 file name to store the data from the .vtk file
    case : str
        Dataset key to store the data inside the HDF5
    path : str
        Path to the .vtk file folder
    filename : str
        Name of the  vtk file
    h : h5py, optional
        File handler of the h5py library, by default None
    close : bool, optional
        Close the HDF5 file at the end of the function run, by default True
    """
    import pyvista as pv
    import xmltodict

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
        unq, _ = np.unique(coord, axis=0, return_index=True)
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
    """Split the data in three different sets.
    
    Parameters
    ----------
    sz : int
        Number of samples to be splitted
    n_train : float, optional
        Percentage for the training dataset, by default 0.8
    n_valid : float, optional
        Percentage for the validation dataset, by default 0.1
    shuffle : bool, optional
        Shuffle the data, by default True

    Returns
    -------
    numpy.ndarray
        Index of the training dataset
    numpy.ndarray
        Index of the validation dataset
    numpy.ndarray
        Index of the training test dataset
    """
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


def slicer(shp, idxs, var=None):
    """Obtain a list of slicers to slice the data array according to the
    selected data

    Parameters
    ----------
    shp : tuple
        Data shape
    idxs : iterable
        Indexes of the selected data
    var : int, optional
        Data to be selected, in case of multidimensional sample, by default None

    Returns
    -------
    slice
        Slices of the data
    """

    # It is assumed that the first dimension is the samples
    slc = []
    # Iterate over the datasets
    for idx in idxs:
        idx.sort()
        if not var:
            slc += [tuple([idx] + [slice(None)] * (len(shp) - 1))]
        else:
            slc += [tuple([idx] + [slice(None)] * (len(shp) - 2) + [var])]
    return tuple(slc)


def format_data(dt, wd=20, var=None, get_y=False, idxs=None, cont=False):
    """Format data to be compatible with LSTM layers from keras.

    It is specific for the HDF5 datasets being used when 'idxs' is not passed.
    The function add a new dimension between the 'sample' dimension and the 
    samples dimension, for instance, the data has the shape of (100, 10, 10), 
    after formatting it will have (100, 1, 10, 10).
    
    The function does not repeat values to be predicted by other cases.

    Parameters
    ----------
    dt : numpy.ndarray or h5py dataset
        Data to be formatted
    wd : int, optional
        Window to be used to predict the next sample, by default 20
    var : int, optional
        'Channel' to be kept while formating the data, by default None
    get_y : bool, optional
        Return the sample to be predicted, by default False
    idxs : np.ndarray, optional
        It is a 2D numpy array or nested list with the sample limits
        , by default None

    Returns
    -------
    numpy.ndarray
        X data
    numpy.ndarray
        Y data
    """
    # Get the simulation indexes
    if not idxs:
        idxs = dt.attrs["idx"]
    # Create a slice from var
    if not var:
        var = slice(None)
    else:
        var = slice(var, var + 1)
    slc_dims = [slice(None)] * (dt.ndim - 2) + [var]
    # Generate slices
    slc_dt = []
    slc_y = []
    for idx in idxs:
        if cont:
            # Slide at every index
            sld = -(wd - 1)
            x_idx = np.arange(*idx)
            y_idx = np.roll(x_idx, sld)[:sld]
            x_idx = x_idx[:sld]
        else:
            # Slide every window
            x_idx = np.arange(idx[0], idx[1], wd)
            y_idx = np.arange(idx[0] + wd - 1, idx[1] + wd - 1, wd)
        slc_dt += [[slice(i, j)] + slc_dims for i, j in zip(x_idx, y_idx)]
        slc_y += [[i] + slc_dims for i in y_idx]

    slc_x = np.arange(len(slc_y))

    if var.start:
        init_x = (len(slc_y), wd - 1, *dt.shape[1:-1], 1)
        init_y = (len(slc_y), *dt.shape[1:-1], 1)
    else:
        init_x = (len(slc_y), wd - 1, *dt.shape[1:])
        init_y = (len(slc_y), *dt.shape[1:])
    # Initialise array
    x_data = np.empty(init_x)
    if get_y:
        y_data = np.empty(init_y)
    # Fill the matrix (sample, time, x, y, z)
    for sdt, sx, sy in zip(slc_dt, slc_x, slc_y):
        x_data[sx] = dt[tuple(sdt)]
        if get_y:
            y_data[sx] = dt[tuple(sy)]
    if get_y:
        return x_data, y_data
    else:
        return x_data


def flat_2d(data, div=0):
    dim_1 = np.prod(data.shape[0 : div + 1], dtype=int)
    dim_2 = np.prod(data.shape[div + 1 :], dtype=int)
    return data.reshape((dim_1, dim_2))


# Sensitivity analysis
def gen_problem(params):
    """Convert parameters to be input to SALib.

    This function converts the params dictionary into a dictionary formatted to 
    be input at the SALib as mean to generate the samples.

    Parameters
    ----------
    params : dict
        Example: {"variable": {"bounds": [0, 1], "type": "int"}}
        
    Returns
    -------
    dict
        Formatted to be input into SALib sampling
    """

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
    """Convert a numpy array into a structured array.

    It converts a numpy array into a structured array to be able to
    handle different datatypes. The different datatypes is used to build
    a neural network.

    Parameters
    ----------
    samples : numpy.ndarray
        Matrix of samples
    params : dict
        Dictionary with the variables datatypes.

    Returns
    -------
    numpy.recarray
        Structured numpy array with different datatypes
    """
    # Convertion to the numpy data types
    cvt = {"float": "f8", "int": "i8"}
    # Create the dtypes
    dtype = [(key, cvt[params[key]["type"]]) for key in params.keys()]
    # Create a dataframe with the cases to train
    smp_st = np.array(list(zip(*samples.T)), dtype=dtype)
    return smp_st


# Plot
# TODO: Plot documentation
def plot_red_comp(original, reduced, var, n_dim, mse_global, alg="PCA"):
    """
    [extended_summary]

    Parameters
    ----------
    original : [type]
        [description]
    reduced : [type]
        [description]
    var : [type]
        [description]
    n_dim : [type]
        [description]
    mse_global : [type]
        [description]
    alg : str, optional
        [description], by default "PCA"
    """
    # Calculate the MSE
    original = original[:, :, var]
    reduced = reduced[:, :, var]
    mse = np.mean((original - reduced) ** 2)
    vmax = original.max()
    vmin = original.min()
    # Generate the subplot figure
    fig, ax = plt.subplots(2, figsize=(8, 8))
    tit = "Global MSE: {:.4f}  Case MSE: {:.4f}".format(mse_global, mse)
    fig.suptitle(tit, y=1.02)
    fig.tight_layout(pad=2)
    ax[0].pcolormesh(original.T, rasterized=True)
    ax[1].pcolormesh(reduced.T, vmax=vmax, vmin=vmin, rasterized=True)
    ax[0].set_title("Original data")
    ax[1].set_title("{} with {} dimensions".format(alg, n_dim))
