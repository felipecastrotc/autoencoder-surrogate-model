import os
import sys
import h5py
import json
import subprocess
import threading
from utils import read_vtk
from queue import Queue


def run(cmd):
    while len(args) > 0:
        # Get current arguments to simulate
        arg = args.pop(0)
        path_format = ''.join(['{}_']*len(arg))[0:-1]
        case = path_format.format(*[a for a in arg])
        path = './' + ppth + case
        # Create target Directory if don't exist
        if not os.path.exists(path):
            os.mkdir(path)
        # Create the command
        arg_str = ' '.join([str(a) for a in arg])
        cmd_arg = '{} {} {}'.format(cmd, arg_str, ppth+case)
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
    h = h5py.File('./' + ppth + save_file, 'w')
    # Storing loop
    save_counter = 0
    crit = False
    while not crit:
        try:
            case, path,  vtk_file = q.get(timeout=1)
            read_vtk(' ', case, path, vtk_file, h, False)
            print(case + ' Finished storing into HDF5')
            save_counter += 1
            print('Completed {} of {} ({:.2f})% !'.format(save_counter, args_sz,
                                                          save_counter*100/args_sz))
        except:
            pass

        crit = (q.qsize() == 0) & stop_store
    h.close()
    print('Storing done!')


def main():
    global stop_store
    # Generating commands
    commands = [command]*n_threads

    # Generte the threads
    store_thread = threading.Thread(
        target=store_hdf5, kwargs={'save_file': save_file})
    proc = [threading.Thread(
        target=run, kwargs={'cmd': cmd}) for cmd in commands]
    # Start the threads
    store_thread.start()
    start_threads = [p.start() for p in proc]
    # Wait to finish
    wait_threads = [p.join() for p in proc]
    # Finishing
    print('Simulations done!')
    stop_store = True
    print('Stoping storage thread!')


# Global variables
stop_store = False
# Queue
q = Queue(100)
# Pre-path
ppth = 'simulations/'

if __name__ == "__main__":
    # File to save
    save_file = sys.argv[1]
    # Where to look for the vtk files
    case_name = sys.argv[2]
    # Command to run the simulation
    command = sys.argv[3]
    # Arguments to be passed to the simulator
    arg_file = sys.argv[4]
    # Number of threads
    n_threads = int(sys.argv[5])
    # Cases to run
    fa = open(arg_file, 'r')
    args = json.load(fa)
    fa.close()
    args_sz = len(args)

    if (not os.path.exists(ppth)) & (ppth != ''):
        os.mkdir(ppth)

    # Start
    main()
    pass

# Command example
# python multi_thread.py data.h5 rayleighBenard2d ./rayleighBenard2d args.json 2
