# libEnsemble Community Examples
A selection of libEnsemble functions and complete workflows from the community.
Many previously built-in libEnsemble examples have been moved here
for easier discoverability.

More information about each of these can be found in the various READMEs
and the ``test_*.py`` Python calling scripts.

Each of the Optimization Generator Function examples is tested regularly
on [GitHub Actions](https://github.com/Libensemble/libe-community-examples/actions)
and has [API documentation available online](https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/).


1. #### VTMOP
   *Optimization Generator Function*

    VTMOP is a Fortran 2008 package containing a robust, portable solver and
    a flexible framework for solving MOPs. Designed for efficiency and
    scalability to an arbitrary number of objectives, VTMOP attempts to generate
    uniformly spaced points on a (possibly nonconvex) Pareto front with minimal
    cost function evaluations.

    ```
    Chang, T.H., Larson, J., Watson, L.T., and Lux, T.C.H. Managing
    computationally expensive blackbox multiobjective optimization problems
    with libEnsemble. In Proc. 2020 Spring Simulation Conference (SpringSim '20),
    Article No. 31, pp. 1–12. DOI: 10.22360/springsim.2020.hpc.001
    ```

    Originally included in libEnsemble v0.8.0. See [here](https://github.com/Libensemble/libensemble/tree/v0.8.0/libensemble/gen_funcs/vtmop_libe).

2. #### LibE-DDMD
   *Complete Workflow*

    A complete Molecular-Dynamics / Machine-Learning adaptive
    simulation loop based on [DeepDriveMD](https://deepdrivemd.github.io/).
    The simulation function runs molecular-dynamics evaluations using DeepDriveMD's
    ``run_openmm.py``, while the persistent generator function runs the remaining
    machine-learning training and model selection operations on the output.
    The generator parameterizes subsequent MD runs by selecting outlier points.
    See ``ddmd/readme.md`` for more information. Constructed by the libEnsemble team
    as a proof-of-concept with help from [the DeepDriveMD team](https://deepdrivemd.github.io/team.html).

    Originally included in libEnsemble v0.8.0. See [here](https://github.com/Libensemble/libensemble/tree/v0.8.0/libensemble/tests/scaling_tests/ddmd).

3. #### DEAP-NSGA-II
   *Optimization Generator Function*

   A persistent generator function that interfaces with the [DEAP](https://github.com/DEAP/deap),
   evolutionary algorithms as generator functions. This example demonstrates the NSGA-II multi-objective optimization
   strategy. The generator evaluates the "fitness" of current population members
   and requests their evaluation from libEnsemble's manager. The manager
   returns corresponding "fitness values" for each objective.

   Last included in libEnsemble v0.8.0. See [here](https://github.com/Libensemble/libensemble/blob/v0.8.0/libensemble/tests/regression_tests/test_deap_nsga2.py).

4. #### RNN-Robustness
   *Complete Workflow*

   A generator-less workflow used to train a selection of neural network architectures on ALCF's Theta. An initial History array of hyperparameters
   are read from ``test_training_args.npz`` and provided to libEnsemble for distribution and evaluation by worker processes. See ``rnn/description.pdf``
   for more information about the included files.

5. #### ParMOO-Emittance
   *Complete Workflow*

   A ParMOO persistent generator function is used to solve a biobjective accelerator emittance minimization problem, while exploiting problem structure.
   This directory contains sample-solvers for multiobjective optimization of particle accelerator setup by using either ParMOO inside libE or libE inside
   of ParMOO. Note that both methods produce equivalent results. See the ``parmoo-emittance/README.rst`` for more details, as well as detailed usage 
   instructions and references.

6. #### Distributed Consensus-based Optimization Methods
   *Generator and Allocator Functions*

   Four generator functions, a matching allocator function, a consensus-subroutines module, a handful of test simulation functions, and tests for each. 
   All created by Caleb Ju at ANL as Given's associate Summer 2021. See the docstrings in each module for more information.

   Last included in libEnsemble v0.9.3. See [here](https://github.com/Libensemble/libensemble/tree/v0.9.3/libensemble/gen_funcs).

7. #### Icesheet Modelling
   *Scripts for workflow - requires external data files*

   Ensembles of ice-flow simulations using GPUs.

   James Chegwidden, and Kuang Hsu Wang (ANL/UND in Summer 2022).
   External Supervisor: Prof. Anjali Sandip (UND).

8. #### heFFTe / ytopt
   *Complete Workflows*

   [heFFTe](https://github.com/icl-utk-edu/heffte) is a fast Fourier transform code
   that is used in many applications and supports heterogeneous hardware. The ``heffte`` subdirectory contains
   a simple example of libEnsemble launching heFFTe's ``speed3d_c2c`` at various predetermined configurations.

   [ytopt](https://github.com/ytopt-team/ytopt) is a Bayesian Optimization package for
   determining optimal input parameter configurations for applications or other executables. The ``ytopt_heffte`` subdirectory
   contains the ``ytopt_asktell`` generator function for providing runtime results to ytopt and prompting it for subsequent parameter
   configurations and a ``test_ytopt_heffte.py`` script for coupling ytopt to heFFTe application instances
   (launched with ``ytopt_obj.py``) via libEnsemble.

9. #### Ax multi-task
   *Optimization Generator Function*

   A generator function for multi-fidelity Bayesian optimization with a Gaussian process.

10. #### GP Dragonfly
    *Optimization Generator Function*

      Another generator function for Bayesian optimization with a Gaussian process, using `dragonfly`. Previous
      working versions of the included test required the following fork: `https://github.com/jlnav/dragonfly`,
      since upstream no longer appears under active development.

11. #### Gen Updates CNN - Data-parallel Distributed Supervised Learning
    *Complete Workflow*

      libEnsemble starts N parallel training instances on separate, distributed
      worker processes, each with an even split of the training dataset. These workers
      compute the gradient of the loss function for their portion
      of the dataset and send their gradients to a generator process. The generator process
      combines the gradients from all workers, updates the model parameters, and sends 
      the updated model back to the workers.

      Given the number of gradients/parameters that are being updated,
      [proxystore](https://docs.proxystore.dev/main/)
      is used to stream data between the processes.