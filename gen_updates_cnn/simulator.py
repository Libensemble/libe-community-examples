import sys
import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator


from mnist.nn import main as run_cnn


def _run_cnn_send(generator, sim_specs, parameters=None):

    Output = np.zeros(1, dtype=sim_specs["out"])

    grads, params = run_cnn(parameters)  # initial
    Output["grads"] = [i.cpu().detach().numpy() for i in grads]
    Output["parameters"] = [i.cpu().detach().numpy() for i in params]
    generator.send(Output)


@input_fields(["parameters"])
@persistent_input_fields(["parameters"])
@output_data(
    [("grads", object, (8,)), ("parameters", object, (8,))]
)
def mnist_training_sim(H, _, sim_specs, info):

    generator = ToGenerator(info, EVAL_SIM_TAG)

    _run_cnn_send(generator, sim_specs)

    while True:
        tag, Work, calc_in = generator.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(generator, sim_specs, calc_in["parameters"])
