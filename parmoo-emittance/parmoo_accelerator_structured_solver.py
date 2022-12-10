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
from parmoo.extras.libe import libE_MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint, FixedWeights
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

if __name__ == "__main__":
    """ For a libE_MOOP to be run on certain OS (such as MacOS) it must be
    enclosed within an ``if __name__ == '__main__'`` clause. """

    # Create a libE_MOOP
    moop = MOOP(TR_LBFGSB, hyperparams={})
    # Add 2 design variables from accelerator_model meta data
    for i in range(n):
        moop.addDesign({'name': fayans_model.DES_NAMES[i],
                               'lb': accelerator_model.lb[i],
                               'ub': accelerator_model.ub[i]})
    # Add 1 simulation, with 100 pt initial search
    moop.addSimulation({'name': "sim out",
                        'm': m,
                        'sim_func': accelerator_model.accelerator_sim_model,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': LocalGaussRBF})
    # Add 2 objectives
    moop.addObjective({'name': "emittance",
                       'obj_func': accelerator_model.emittance})
    moop.addObjective({'name': "emittance",
                       'obj_func': accelerator_model.bunch_length})
    # Add 5 random acquisition functions (generate batch of 5 sims per iter)
    for i in range(9):
        moop.addAcquisition({'acquisition': RandomConstraint,
                             'hyperparams': {}})
    # Fix the random seed for reproducability
    np.random.seed(iseed)
    # Solve the Fayans EDF callibration moop with a 200 sim budget
    moop.solve(sim_max=200, wt_max=60)
    full_data = moop.getObjectiveData()
    soln = moop.getPF()

    # Dump full data set to a CSV file
    with open("accelerator_structured_results_seed_" + str(iseed) + ".csv",
              "w") as fp:
        csv_writer = csv.writer(fp, delimiter=",")
        # Define the header
        header = accelerator_model.DES_NAMES.copy()
        header.append("binding energy")
        header.append("std radii")
        header.append("other quantities")
        # Dump header to first row
        csv_writer.writerow(header)
        # Add each data point as another row
        for xs in full_data:
            csv_writer.writerow([xs[name] for name in header])


def parmoo_persis_gen(H, persis_info, gen_specs, libE_info):
    """ A persistent ParMOO generator function for libEnsemble.

    This generator function is meant to be called from within libEnsemble.

    Args:
        H (numpy structured array): The current libE history array.

        persis_info (dict): Any information that should persist after this
            generator has exited. Must contain the following field:
             * 'moop' (parmoo.MOOP)

        gen_specs (dict): A list of specifications for the generator function.

        libE_info (dict): Other information that will be used by libEnsemble.

    Returns:
        dict: The final simulation history.

        dict: The persistent information after completion of the generator.

        int: The stop tag.

    """

    from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.tools.persistent_support import PersistentSupport

    # Get moop from pers_info
    if 'moop' in persis_info:
        moop = persis_info['moop']
        if not isinstance(moop, MOOP):
            raise TypeError("persis_info['moop'] must be an instance of " +
                            "parmoo.MOOP class")
    else:
        raise KeyError("'moop' key is required in persis_info dict")
    # Setup persistent support
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    # Send batches until manager sends stop tag
    tag = None
    k = 0
    sim_count = 0
    # Iterate until the termination condition is reached
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Generate a batch by running one iteration
        x_out = moop.iterate(k)
        # Check for duplicates in simulation databases
        xbatch = []
        ibatch = []
        for (xi, i) in x_out:
            if moop.check_sim_db(xi, i) is None:
                xbatch.append(xi)
                ibatch.append(i)
        # Get the batch size and allocate the H_o structured array
        b = len(xbatch)
        H_o = np.zeros(b, dtype=gen_specs['out'])
        # Populate the H_o structured array 'x' values as appropriate
        if moop.use_names:
            for name in moop.des_names:
                for i in range(b):
                    H_o[name[0]][i] = xbatch[i][name[0]]
        else:
            H_o['x'] = np.asarray(xbatch)
        for i, namei in enumerate(ibatch):
            H_o['sim_name'][i] = namei
        # Evaluate H_o and add to the simulation database
        batch = []
        if isinstance(x_out[0][-1], str) or x_out[0][-1] >= 0:
            tag, Work, calc_in = ps.send_recv(H_o)
            if calc_in is not None:
                for s_out in calc_in:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    sim_count += 1
            else:
                new_count = 0
                for s_out in Work[sim_count:]:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    new_count += 1
                sim_count += new_count
        # Update the ParMOO databases
        moop.updateAll(k, batch)
        k += 1
    # Return the results
    persis_info['moop'] = moop
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG

        from libensemble.libE import libE
        from libensemble.alloc_funcs.start_only_persistent \
            import only_persistent_gens as alloc_f
        from libensemble.tools import parse_args

        # Create libEnsemble dictionaries
        nworkers, is_manager, libE_specs, _ = parse_args()
        if self.moop.use_names:
            libE_specs['final_fields'] = []
            for name in self.moop.des_names:
                libE_specs['final_fields'].append(name[0])
            for name in self.moop.sim_names:
                libE_specs['final_fields'].append(name[0])
            libE_specs['final_fields'].append('sim_name')
        else:
            libE_specs['final_fields'] = ['x', 'f', 'sim_name']
        # Set optional libE specs
        libE_specs['profile'] = profile

        if nworkers < 2:
            raise ValueError("Cannot run ParMOO + libE with less than 2 " +
                             "workers -- aborting...\n\n" +
                             "Note: this error could be caused by a " +
                             "failure to specify the communication mode " +
                             " (e.g., local comms or MPI)")

        # Get the max m for all SimGroups
        max_m = max(self.moop.m)

        # Set the input dictionaries
        if self.moop.use_names:
            x_type = self.moop.des_names.copy()
            x_type.append(('sim_name', 'a10'))
            f_type = self.moop.sim_names.copy()
            all_types = x_type.copy()
            for name in f_type:
                all_types.append(name)
            sim_specs = {'sim_f': self.moop_sim,
                         'in': [name[0] for name in x_type],
                         'out': f_type}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': [name[0] for name in all_types],
                         'out': x_type,
                         'user': {}}
        else:
            sim_specs = {'sim_f': self.moop_sim,
                         'in': ['x', 'sim_name'],
                         'out': [('f', float, max_m)]}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': ['x', 'sim_name', 'f'],
                         'out': [('x', float, self.moop.n),
                                 ('sim_name', int)],
                         'user': {}}

        alloc_specs = {'alloc_f': alloc_f, 'out': [('gen_informed', bool)]}

        persis_info = {}
        for i in range(nworkers + 1):
            persis_info[i] = {}
        persis_info[1]['moop'] = self.moop

        exit_criteria = {'sim_max': sim_max, 'wallclock_max': wt_max}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                    persis_info, alloc_specs, libE_specs)

        self.moop = persis_info[1]['moop']
        return
