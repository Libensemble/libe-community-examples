import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator


from mnist.nn import main as run_cnn


def _run_cnn_send(generator, sim_specs, parameters=None, workerID=0):

    Output = np.zeros(1, dtype=sim_specs["out"])

    grads, params = run_cnn(parameters, workerID)
    Output["grads"] = [i.cpu().detach().numpy() for i in grads]
    Output["output_parameters"] = [i.cpu().detach().numpy() for i in params]
    generator.send(Output)


@input_fields(["input_parameters", "sim_id"])
@persistent_input_fields(["input_parameters"])
@output_data(
    [("grads", object, (8,)), ("output_parameters", object, (8,))]
)
def mnist_training_sim(H, _, sim_specs, info):

    generator = ToGenerator(info, EVAL_SIM_TAG)

    _run_cnn_send(generator, sim_specs, None, info["workerID"])

    while True:
        tag, Work, calc_in = generator.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(generator, sim_specs, calc_in["input_parameters"], calc_in["sim_id"])

    return [], {}, 0