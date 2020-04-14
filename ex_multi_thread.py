import subprocess
import threading
import os


def run(cmd):
    while len(args) > 0:
        # Get current arguments to simulate
        arg = args.pop(0)
        path = path_format.format(*[a for a in arg])
        path_fld = './' + path
        # Create target Directory if don't exist
        if not os.path.exists(path_fld):
            os.mkdir(path_fld)
        # Create the command
        arg_str = ' '.join([str(a) for a in arg])
        cmd_arg = '{} {}'.format(cmd, arg_str)
        # Create the log files
        out = open(path_fld + '/log.log', 'w')
        err = open('/dev/null', 'w')
        p = subprocess.Popen(cmd.split(' '), stdout=out, stderr=err)
        p.wait()
        print(cmd_arg + ' completed, return code: ' + str(p.returncode))


n_threads = 2
command = ['./dummy.sh']*n_threads
args = [[1], [2], [3], [4], [5], [6], [7]]

# Path settings
path_format = 'ra-{}_pr-{}_Thot-{}_Tcold-{}'
path_format = 'ra-{}'

# Start the threads
proc = [threading.Thread(target=run, kwargs={'cmd': cmd}) for cmd in command]
start_threads = [p.start() for p in proc]
# Wait to finish
wait_threads = [p.join() for p in proc]
# Finishing
print("Done!")
