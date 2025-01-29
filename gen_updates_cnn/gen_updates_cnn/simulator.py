import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator


from .mnist.nn import Net, main as run_cnn

def _run_cnn_send(generator, sim_specs, parameters):

    Output = np.zeros(1, dtype=sim_specs["out"])

    grads = run_cnn(parameters)
    Output["local_gradients"] = [i.cpu().detach().numpy() for i in grads]
    generator.send(Output)


@input_fields(["parameters"])
@persistent_input_fields(["parameters"])
@output_data(
    [("local_gradients", object, (8,))]
)
def mnist_training_sim(H, _, sim_specs, info):

    generator = ToGenerator(info, EVAL_SIM_TAG)

    _run_cnn_send(generator, sim_specs, H["parameters"][0])

    while True:
        tag, Work, calc_in = generator.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(generator, sim_specs, calc_in["parameters"])

    return None, {}, 0