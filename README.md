# libEnsemble Community Examples
A selection of libEnsemble functions and complete workflows from the community.
Many previously built-in libEnsemble examples have been moved here
for easier discoverability. See the various READMEs for more information.

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

    Originally included in libEnsemble v0.8.0. See [here](https://github.com/Libensemble/libensemble/tree/main/libensemble/gen_funcs/vtmop_libe).


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

    Originally included in libEnsemble v0.8.0. See [here](https://github.com/Libensemble/libensemble/tree/main/libensemble/tests/scaling_tests/ddmd).
