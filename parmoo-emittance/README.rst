======================================================
ParMOO+libE for Optimizing Particle Accelerator Setups
======================================================

This repository contains sample-solvers for multiobjective optimization of
particle accelerator setup by using either ParMOO inside libE or
libE inside of ParMOO.
Note that both methods produce equivalent results.

Setup and Installation
----------------------

The requirements for this directory are:
 - libensemble_ and
 - parmoo_

To use or compare against any of the sample problem, clone this directory
and install the requirements, or install the included ``REQUIREMENTS.txt``
file.

.. code-block:: bash

    python3 -m pip install -r REQUIREMENTS

Instructions and Structure
--------------------------

This particular directory contains three Python files.
 - ``accelerator_model.py`` defines a made-up simulation function that
   mimics the inputs, outputs, and general structure of a multiobjective
   emittance minimization problem;
 - ``libe_parmoo_acc_solve.py`` demonstrates how to set-up and solve
   the made-up problem from inside ``libensemble``, using ParMOO as
   a libensemble ``gen_func``; and
 - ``parmoo_libe_acc_solve.py`` demonstrates how to set-up and solve
   the made-up problem using ParMOO with a ``libE`` backend to distribute
   simulation evaluations.

If ``name.py`` is the name of the file that you want to run, then
you can execute any file in this directory using the command:

.. code-block:: bash

    python3 name.py --comms C --nworkers N [--iseed I]

where ``C`` is the communication protocol (``local`` or ``tcp``);
``N`` is the number of libE workers (i.e., number of concurrent simulation
evaluations); and
``I`` is the random seed, which can be fixed to any integer for
reproducability (when omitted, it is assigned by the system clock).

After running, the complete function-value database is saved to a file
``parmoo_libe_acc_results_seed_I.csv`` or
``libe_parmoo_acc_results_seed_I.csv``, depending on the method run
where ``I`` is as defined above.

Resources
---------

For more reading on the ParMOO library and its other options

 * visit the parmoo_GitHub_page_, or
 * view the parmoo_readthedocs_page_

Citing ParMOO
-------------

To cite the ParMOO library, use one of the following:

.. code-block:: bibtex

    @article{parmoo,
        title   = {{ParMOO}: A {P}ython library for parallel multiobjective simulation optimization},
        author  = {Chang, Tyler H. and Wild, Stefan M.},
        year    = {2022},
        journal = {Journal of Open Source Software},
        note    = {Under review, see \url{https://github.com/openjournals/joss-reviews/issues/4468}}
    }

    @techreport{parmoodocs,
        title       = {{ParMOO}: {P}ython library for parallel multiobjective simulation optimization},
        author      = {Chang, Tyler H. and Wild, Stefan M.},
        institution = {Argonne National Laboratory},
        number      = {Version 0.1.0},
        year        = {2022},
        url         = {https://parmoo.readthedocs.io/en/latest}
    }


.. _libensemble: https://github.com/libensemble/libensemble
.. _parmoo: https://github.com/parmoo/parmoo
.. _parmoo_github_page: https://github.com/parmoo/parmoo
.. _parmoo_readthedocs_page: https://parmoo.readthedocs.org
