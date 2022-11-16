#!/bin/bash -x
#COBALT -t 01:00:00
#COBALT -n 2
#COBALT -q debug-cache-quad
#COBALT -A rnn-robustness
#COBALT -O libE-project
#COBALT --attrs filesystems=home,grand

# --- Prepare Python ---

# Obtain Conda PATH from miniconda-3/latest module
CONDA_DIR=/soft/datascience/conda/miniconda3/latest/bin

# Name of conda environment
export CONDA_ENV_NAME=my_env

# Activate conda environment
export PYTHONNOUSERSITE=1
source $CONDA_DIR/activate $CONDA_ENV_NAME

# --- Prepare libEnsemble ---

# Name of calling script
export EXE=calling_script.py

# Communication Method
export COMMS='--comms local'

# Number of workers.
export NWORKERS='--nworkers 2'

# Required for killing tasks from workers on Theta
export PMI_NO_FORK=1

# Unload Theta modules that may interfere with task monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

python $EXE $COMMS $NWORKERS > out.txt 2>&1
