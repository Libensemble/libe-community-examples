import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.specs import ExitCriteria, AllocSpecs, SimSpecs, GenSpecs
from gen_updates_cnn.simulator import mnist_training_sim
from gen_updates_cnn.generator import parent_model_trainer

if __name__ == "__main__":

    ensemble = Ensemble(parse_args=True)

    ensemble.libE_specs.gen_on_manager = True

    user = {"num_networks": ensemble.nworkers}

    sim_specs = SimSpecs(sim_f=mnist_training_sim, user=user)
    gen_specs = GenSpecs(gen_f=parent_model_trainer, user=user)
    alloc_specs = AllocSpecs(alloc_f=only_persistent_gens)

    ensemble.sim_specs = sim_specs
    ensemble.gen_specs = gen_specs
    ensemble.alloc_specs = alloc_specs
    ensemble.exit_criteria = ExitCriteria(wallclock_max=10)

    ensemble.run()
    ensemble.save_output(__file__)
