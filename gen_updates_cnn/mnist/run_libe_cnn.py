
import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work
from libensemble.specs import ExitCriteria, AllocSpecs, SimSpecs, GenSpecs

if __name__ == "__main__":

    n_samples = 4
    init_history = np.zeros(n_samples, dtype=[("weights", object), ("acc", float), ("sim_id", int)])
    init_history["weights"] = [None] * n_samples
    init_history["acc"] = [0.0] * n_samples
    init_history["sim_id"] = range(n_samples)

    # Create the ensemble
    ensemble = Ensemble(parse_args=True)

    # Create the sim_specs
    sim_specs = SimSpecs()
    sim_specs.sim_f = "run_cnn"
    sim_specs.inputs = ["weights"]
    sim_specs.outputs = [("acc", float)]

    gen_specs = GenSpecs()
    gen_specs.gen_f = "eval_cnn"
    gen_specs.inputs = ["acc"]
    gen_specs.outputs = [("weights", object)]

    alloc_specs = AllocSpecs()
    alloc_specs.alloc_f = give_pregenerated_sim_work

    ensemble.sim_specs = sim_specs
    ensemble.gen_specs = gen_specs
    ensemble.alloc_specs = alloc_specs
    ensemble.exit_criteria = ExitCriteria(sim_max=4)

    ensemble.run()
    ensemble.print_stats()

