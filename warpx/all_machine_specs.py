import os
import shutil

"""
This file is part of the suite of scripts to use LibEnsemble on top of WarpX
simulations. It contains a dictionary for machine-specific elements.
"""


local_specs = {
    "name": "local",  # Machine name
    "cores": 1,  # Number of cores per simulation
    "sim_app": shutil.which("warpx.2d"),
    "extra_args": "",  # extra arguments passed to mpirun/mpiexec at execution
    "OMP_NUM_THREADS": "2",
    "sim_max": 10,  # Maximum number of simulations
}

polaris_specs = {
    "name": "polaris",  # Machine name
    "cores": 4,  # Number of cores per simulation
    "sim_app": shutil.which("warpx.2d"),
    "extra_args": "",  # extra arguments passed to mpirun/mpiexec at execution
    "OMP_NUM_THREADS": "8",
    "sim_max": 100,  # Maximum number of simulations
}
