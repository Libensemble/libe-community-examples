import sys
import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport


from mnist.nn import main as run_cnn


def _run_cnn_send(sim_specs, weights=None):

    Output = np.zeros(1, dtype=sim_specs["out"])

    grad, train_loss, parameters = run_cnn(weights)  # initial
    Output["grad"] = grad
    Output["loss"] = train_loss
    Output["parameters"] = parameters
    ps.send(Output)


@persistent_input_fields(["weights"])
@output_data([("loss", float), ("grad", object, (10, 128)), ("parameters", object)])
def mnist_training_sim(H, _, sim_specs, info):

    ps = PersistentSupport(info, EVAL_SIM_TAG)
    tag = None

    _run_cnn_send(sim_specs)

    while True:
        tag, Work, calc_in = ps.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(sim_specs, Work["weights"])
