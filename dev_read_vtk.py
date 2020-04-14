import h5py
import xmltodict
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

path = './dados/rayleigh-bernard_default/tmp/vtkData/'
filename = 'rayleighBenard2d.pvd'

# Open the .pvd file
f = open(path+filename)
data_paths = xmltodict.parse(f.read())
f.close()
# Get the list of tim steps
data_paths = data_paths['VTKFile']['Collection']['DataSet']

# Set the time step as a group property
if len(data_paths) > 1:
    time_step = int(data_paths[1]['@timestep']) - \
        int(data_paths[0]['@timestep'])
else:
    time_step = 0

i = 50
# Get the data step
mesh = pv.read(path + data_paths[i]['@file'])

# Plot meshes
mk = 0.01
c = ['r', 'b', 'g', 'k', 'y', 'm', 'c']
for j in range(7):
    plt.scatter(mesh[i][0].points[:, 0], mesh[i][0].points[:, 1], s=mk, c=c[i])

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

plt.pcolormesh(dt_mtx[:, :, 1].T, rasterized=True)

#
h = h5py.File('data.h5')
h['example/1'].attrs.keys()
h.close()

ph['example']['1'].value[:, :, 1]
