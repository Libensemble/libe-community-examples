""" Create a ParMOO generator inside of libE to solve the accelerator
optimization MOOP.

Execute via one of the following commands (where N is the number of threads,
and the optional int M in [0, 2^32-1] is a random seed):

```
mpiexec -np N python3 parmoo_accelerator_structured_solver.py [--iseed M]
python3 parmoo_accelerator_structured_solver.py --nworkers N --comms local [--iseed M]
python3 parmoo_accelerator_structured_solver.py --nworkers N --comms tcp [--iseed M]
```


The number of concurrent evaluations of the accelerator sim model will be N-1,
since the first thread is reserved for the ParMOO generator.

"""

import numpy as np
import csv
import sys
from time import time_ns
from parmoo import MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.extras.libe import parmoo_persis_gen
from libensemble.libE import libE
from libensemble.alloc_funcs.start_only_persistent \
    import only_persistent_gens as alloc_f
from libensemble.tools import parse_args
import accelerator_model

# Set the problem dimensions
n = 2
m = 4
o = 2

# Read libE specific inputs from command line
nworkers, is_manager, libE_specs, _ = parse_args()
if nworkers < 2:
    raise ValueError("Cannot run ParMOO + libE with less than 2 " +
                     "workers -- aborting...\n\n" +
                     "Note: this error could be caused by a " +
                     "failure to specify the communication mode " +
                     " (e.g., local comms or MPI)")

# Read the random seed from the command line
iseed = time_ns() % (2 ** 32) # default is to set from system clock
for i, opt in enumerate(sys.argv[1:]):
    if opt == "--iseed":
        try:
            iseed = int(sys.argv[i+2])
        except IndexError:
            raise ValueError("iseed requires an integer value")
        except ValueError:
            raise ValueError("iseed requires an integer value")
# Fix the random seed for reproducability
np.random.seed(iseed)

# Create a dummy sim func to give parmoo - libE gets the real sim func
def dummy_sim(x): return np.zeros(m)

if __name__ == "__main__":
    """ For a libE_MOOP to be run on certain OS (such as MacOS) it must be
    enclosed within an ``if __name__ == '__main__'`` clause. """

    # Create a parmoo.MOOP object, which we will pass as the libE gen_func
    moop = MOOP(TR_LBFGSB, hyperparams={})
    # Add 2 design variables from accelerator_model meta data
    for i in range(n):
        moop.addDesign({'name': accelerator_model.DES_NAMES[i],
                        'lb': accelerator_model.lb[i],
                        'ub': accelerator_model.ub[i]})
    # Add 1 simulation, with 100 pt initial search
    moop.addSimulation({'name': "sim out",
                        'm': m,
                        'sim_func': dummy_sim,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': LocalGaussRBF})
    # Add 2 objectives
    moop.addObjective({'name': "emittance",
                       'obj_func': accelerator_model.emittance})
    moop.addObjective({'name': "bunch length",
                       'obj_func': accelerator_model.bunch_length})
    # Add 5 random acquisition functions (generate batch of 5 sims per iter)
    for i in range(5):
        moop.addAcquisition({'acquisition': RandomConstraint,
                             'hyperparams': {}})

    # Create libEnsemble dictionaries
    libE_specs['final_fields'] = []
    for name in moop.des_names:
        libE_specs['final_fields'].append(name[0])
    for name in moop.sim_names:
        libE_specs['final_fields'].append(name[0])
    libE_specs['final_fields'].append('sim_name')

    # Define input/output dtypes
    x_type = [(f'{name}', 'f8') for name in accelerator_model.DES_NAMES]
    x_type.append(('sim_name', 'a10'))
    f_type = [('sim out', 'f8', 4)]
    all_types = x_type.copy()
    for ft in f_type:
        all_types.append(ft)

    # Set the input dictionaries
    sim_specs = {'sim_f': accelerator_model.accelerator_sim_model,
                 'in': [name[0] for name in x_type],
                 'out': f_type}
    gen_specs = {'gen_f': parmoo_persis_gen,
                 'persis_in': [name[0] for name in all_types],
                 'out': x_type,
                 'user': {}}
    alloc_specs = {'alloc_f': alloc_f, 'out': [('gen_informed', bool)]}

    # Initialize persistent info dictionaries
    persis_info = {}
    for i in range(nworkers + 1):
        persis_info[i] = {}
    persis_info[1]['moop'] = moop

    # Solve the accelerator callibration moop with a 200 sim budget
    exit_criteria = {'sim_max': 200, 'wallclock_max': 60}
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info, alloc_specs, libE_specs)

    # Extract solutions from persistent moop object
    soln = persis_info[1]['moop'].getPF()
    full_data = persis_info[1]['moop'].getObjectiveData()

    # Dump full data set to a CSV file
    with open(f"accelerator_libe_parmoo_results_seed_{iseed}.csv", "w") as fp:
        csv_writer = csv.writer(fp, delimiter=",")
        # Define the header
        header = accelerator_model.DES_NAMES.copy()
        header.append("emittance")
        header.append("bunch length")
        # Dump header to first row
        csv_writer.writerow(header)
        # Add each data point as another row
        for xs in full_data:
            csv_writer.writerow([xs[name] for name in header])
