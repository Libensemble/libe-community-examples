import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator


from .mnist.nn import Net, main as run_cnn
from .generator import _connect_to_store


def _proxify_gradients(store, grads):
    """Convert resulting gradients to proxies for parent model."""
    return [store.proxy(i.cpu().detach().numpy(), evict=True) for i in grads]


def _run_cnn_send(generator, sim_specs, store, parameters, workerID):
    """Run CNN with new parameters, send gradients to parent model."""

    Output = np.zeros(1, dtype=sim_specs["out"])

    grads = run_cnn(parameters, workerID, sim_specs["user"]["num_networks"])

    Output["local_gradients"] = _proxify_gradients(store, grads)
    generator.send(Output)


@input_fields(["parameters"])
@persistent_input_fields(["parameters"])
@output_data([("local_gradients", object, (8,))])
def mnist_training_sim(InitialData, _, sim_specs, info):
    """
    Run CNN with parameters from parent model. Send gradients to parent model.
    """

    workerID = info["workerID"]

    generator = ToGenerator(info, EVAL_SIM_TAG)
    store = _connect_to_store()

    _run_cnn_send(generator, sim_specs, store, InitialData["parameters"][0], workerID)

    while True:
        tag, _, SubsequentData = generator.recv()
        if tag in [PERSIS_STOP, STOP_TAG]:
            break

        _run_cnn_send(
            generator, sim_specs, store, SubsequentData["parameters"][0], workerID
        )

    return None, {}, 0
