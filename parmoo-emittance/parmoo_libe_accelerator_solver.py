""" Create a ParMOO solver for the accelerator optimization MOOP, that uses a
libE backend.

Execute via one of the following commands (where N is the number of threads,
and the optional int M in [0, 2^32-1] is a random seed):

```
mpiexec -np N python3 parmoo_libe_accelerator_solver.py [--iseed M]
python3 parmoo_libe_accelerator_solver.py --nworkers N --comms local [--iseed M]
python3 parmoo_libe_accelerator_solver.py --nworkers N --comms tcp [--iseed M]
```


The number of concurrent evaluations of the accelerator sim model will be N-1,
since the first thread is reserved for the ParMOO generator.

"""

import numpy as np
import csv
import sys
from time import time_ns
from parmoo.extras.libe import libE_MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint
import accelerator_model

# Set the problem dimensions
n = 2
m = 4
o = 2

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


if __name__ == "__main__":
    """ For libE to run on certain OS (such as MacOS) it must be
    enclosed within an ``if __name__ == '__main__'`` clause. """

    # Create a parmoo.MOOP object, which we will pass as the libE gen_func
    libe_moop = libE_MOOP(TR_LBFGSB, hyperparams={})
    # Add 2 design variables from accelerator_model meta data
    for i in range(n):
        libe_moop.addDesign({'name': accelerator_model.DES_NAMES[i],
                             'lb': accelerator_model.lb[i],
                             'ub': accelerator_model.ub[i]})
    # Add 1 simulation, with 100 pt initial search
    libe_moop.addSimulation({'name': "sim out",
                             'm': m,
                             'sim_func':
                             accelerator_model.accelerator_sim_model_parmoo,
                             'hyperparams': {'search_budget': 100},
                             'search': LatinHypercube,
                             'surrogate': LocalGaussRBF})
    # Add 2 objectives
    libe_moop.addObjective({'name': "emittance",
                            'obj_func': accelerator_model.emittance})
    libe_moop.addObjective({'name': "bunch length",
                            'obj_func': accelerator_model.bunch_length})
    # Add 5 random acquisition functions (generate batch of 5 sims per iter)
    for i in range(5):
        libe_moop.addAcquisition({'acquisition': RandomConstraint,
                                  'hyperparams': {}})

    # Solve with a budget of 200 sims
    libe_moop.solve(sim_max=200, wt_max=60)

    # Extract solutions from persistent libe_moop object
    soln = libe_moop.getPF()
    full_data = libe_moop.getObjectiveData()

    # Dump full data set to a CSV file
    with open(f"parmoo_libe_acc_results_seed_{iseed}.csv", "w") as fp:
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
