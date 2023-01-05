import os
import time
import numpy as np
from libensemble.executors.executor import Executor


def run_sim_f(H, persis_info, sim_specs, libE_info):
    """
    Description:
        Simulation user function for running 'train_network' application via the
        Executor. LibEnsemble's workers call this simulation function with a
        (single) row of the History array, corresponding to the input arguments
        for the training function.

    Note:
        Application execution is limited to one app per node. In order to train
        multiple networks simultaneously on one node, multiple processes per
        node are spawned where each process uses a subset of the node CPUs to
        train a unique network.
    """

    # parse hyperparameters from H['in']['alloc id']
    architecture = H['model_type'][0]
    learning_rate = H['learning_rate'][0]
    hid_dim = H['hid_dim'][0]
    epochs = H['epochs'][0]
    dataset = H['dataset'][0]
    permute = H['permute'][0]
    pad = H['pad'][0]
    orientation = H['orientation'][0]
    identifier = H['identifier'][0]
    sim_path = os.getcwd()
    ensemble_end_time = sim_specs['user']['ensemble_end_time']

    args = f'{architecture} {learning_rate} {hid_dim} {epochs} {dataset} {permute} {pad} {orientation} {identifier} {sim_path} {ensemble_end_time}'

    # Specify the Executor object created in the calling script.
    exctr = Executor.executor

    # simulation arguments
    # 10 mpi ranks in total (-n and -N)
    # 4 hardware thread per rank (-d)
    # 1 hardware thread per physical core (-j)
    # 1 OpenMP thread per mpi rank (-e)
    PROCS = 10
    PPN = 10
    d = 4
    OMP = 1
    command_args = f'-n {PROCS} -N {PPN} -d {d} -j 1 -cc depth -e OMP_NUM_THREADS={OMP}'

    # submit application to executor
    task = exctr.submit(app_name='train_network', app_args=args, num_nodes=1, extra_args=command_args)

    exctr.polling_loop(task, timeout=None, delay=1)

    time.sleep(0.2)
    output_path = os.path.join(sim_path, f'training-history-{identifier}-*****.npy')

    # generate and populate libE output history array
    H_out = np.zeros(1, dtype=sim_specs['out'])
    H_out['history_path'] = output_path

    return H_out, persis_info
