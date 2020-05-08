# This is an example of how to use the read_vtk function

from utils import read_vtk

# Data structure
# case/1,2,3....
# Properties of case/n -> physical properties, step, spatial step
# Properties of case -> simulation parameters, time step

# File to save
save_file = './data/data.h5'
case = 'example'

# File to read
path = './data/rayleigh-bernard_default/tmp/vtkData/'
filename = 'rayleighBenard2d.pvd'

read_vtk(save_file, case, path, filename)
