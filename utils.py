# read_vtk
import h5py
import xmltodict
import numpy as np
import pyvista as pv


# Matrices multiplication
# y = np.transpose(X, axes=(0, 2, 1))
# np.einsum('ijk,ikm->ijm',x,y)
# X @ np.transpose(X, axes=(0, 2, 1))


def read_vtk(save_file, case, path, filename, h=None, close=True):
    if not h:
        h = h5py.File(save_file, 'w')

    # Open the .pvd file
    f = open(path+filename)
    data_paths = xmltodict.parse(f.read())
    f.close()
    # Get the list of tim steps
    data_paths = data_paths['VTKFile']['Collection']['DataSet']

    for i in range(len(data_paths)):
        # Get the data step
        mesh = pv.read(path + data_paths[i]['@file'])

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
        stp = stp.round(int(stp_dec[np.abs(stp_dec).argmax()]*-1))
        # Get the largest stp
        stp = np.max(stp)

        # Transform the coordinates to index
        coord_idx = (coord/stp).astype(int)
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
        dt_mtx = np.empty(
            [coord_idx[:, 0].max()+1, coord_idx[:, 1].max()+1, n_dim])

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
            dt_st = h.create_dataset(case, dt_shape, dtype=float, 
                                     compression='gzip')
            # Set the time step as a group property
            if len(data_paths) > 1:
                dt_st.attrs['time_step'] = int(data_paths[1]['@timestep']) - \
                                           int(data_paths[0]['@timestep'])
            else:
                dt_st.attrs['time_step'] = 0
            # Properties of case/n -> physical properties, step, spatial step
            dt_st.attrs['spatial_step_x'] = stp
            dt_st.attrs['spatial_step_y'] = stp
            dt_st.attrs['physical'] = dt_info
            dt_st[i, :, :, :] = dt_mtx
        else:
            dt_st[i, :, :, :] = dt_mtx

    if close:
        h.close()
