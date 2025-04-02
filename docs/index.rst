libEnsemble Community Examples
==============================

The following is an overview of several examples included in
libEnsemble's `community repository`_. See the README in that repository for
more information.

:doc:`Main libEnsemble Documentation<main:index>`

Generators
==========

.. _vtmop-link:

vtmop
-----
.. automodule:: vtmop
  :members:

.. _deap-link:

persistent_deap_nsga2
---------------------

Required: DEAP_

.. automodule:: persistent_deap_nsga2
  :members: deap_nsga2
  :undoc-members:

.. _consensus-link:

Consensus Generators
--------------------

.. automodule:: gens.persistent_independent_optimize
  :members: independent_optimize

.. automodule:: gens.persistent_n_agent
  :members: n_agent

.. automodule:: gens.persistent_pds
  :members: opt_slide

.. automodule:: gens.persistent_prox_slide
  :members: opt_slide

.. _ytopt-link:

ytopt
-----

.. automodule:: ytopt_heffte.ytopt_asktell
  :members:

.. _ax-link:

Ax-Multitask
------------

.. automodule:: persistent_ax_multitask
  :members: persistent_gp_mt_ax_gen_f

.. _dragonfly-link:

GP Dragonfly
------------

.. automodule:: persistent_gp
  :members: persistent_gp_gen_f

.. _DEAP: https://deap.readthedocs.io/en/master/overview.html
.. _`community repository`: https://github.com/Libensemble/libe-community-examples

Simulators
==========

.. automodule:: warpx_simf
  :members: run_warpx