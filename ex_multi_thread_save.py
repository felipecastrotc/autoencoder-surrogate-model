import os
import h5py
import subprocess
import threading
from utils import read_vtk
from queue import Queue


def run(cmd):
    while len(args) > 0:
        # Get current arguments to simulate
        arg = args.pop(0)
        case = path_format.format(*[a for a in arg])
        path = './' + case
        # Create target Directory if don't exist
        if not os.path.exists(path):
            os.mkdir(path)
        # Create the command
        arg_str = ' '.join([str(a) for a in arg])
        cmd_arg = '{} {} {}'.format(cmd, arg_str, case)
        # Create the log files
        out = open(path + '/log.log', 'w')
        err = open('/dev/null', 'w')
        # Start process
        print(cmd_arg + ' has started!')
        p = subprocess.Popen(cmd_arg.split(' '), stdout=out, stderr=err)
        p.wait()
        print(cmd_arg + ' completed, return code: ' + str(p.returncode))
        # Storing into HDF5
        print(cmd_arg + ' storing into HDF5')
        q.put((case, path + '/vtkData/', case_name + '.pvd'))


def store_hdf5(save_file):
    # Open the file
    h = h5py.File(save_file, 'w')
    # Storing loop
    crit = False
    while not crit:
        try:
            case, path,  vtk_file = q.get(timeout=1)
            read_vtk(' ', case, path, vtk_file, h, False)
            print(case + ' Finished storing into HDF5')
        except:
            pass
        crit = (q.qsize() == 0) & stop_store
    h.close()
    print('Storing done!')


# File to save
save_file = './data.h5'
case_name = 'rayleighBenard2d'
command = './rayleighBenard2d'
n_threads = 2
# Cases to run
args = [[1e4, 0.71, 274.15, 273.15], [1e5, 0.71, 274.15, 273.15]]
# [1e6, 0.71, 274.15, 273.15], [1e7, 0.71, 274.15, 273.15]]

# Generating commands
commands = [command]*n_threads

# Path settings
# path_format = 'ra-{:.3e}_pr-{:.3e}_Thot-{:.3e}_Tcold-{:.3e}'
path_format = '{}_{}_{}_{}'

# Queue
q = Queue(100)
stop_store = False

# Generte the threads
store_thread = threading.Thread(
    target=store_hdf5, kwargs={'save_file': save_file})
proc = [threading.Thread(target=run, kwargs={'cmd': cmd}) for cmd in commands]
# Start the threads
store_thread.start()
start_threads = [p.start() for p in proc]
# Wait to finish
wait_threads = [p.join() for p in proc]
# Finishing
print('Simulations done!')
stop_store = True
print('Stoping storage thread!')
