import random
import sys
import numpy as np
from itertools import chain

from libensemble.message_numbers import (
    PERSIS_STOP,
    STOP_TAG,
    EVAL_GEN_TAG,
    FINISHED_PERSISTENT_GEN_TAG,
)
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

import torch
import torch.optim as optim
import torch.nn.functional as F

from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, get_store
from proxystore.proxy import Proxy, is_resolved

from .mnist.nn import Net


def _optimize(model, device, grads, optimizer):
    """Assign summed gradients from simulators to parent model. Trains parent model."""
    for param, new_grad in zip(model.parameters(), grads):
        new_grad = new_grad.to(device)
        param.grad = new_grad

    model.train()
    optimizer.step()


def _connect_to_store():
    """Connect to proxystore redis server"""
    store = Store(
        "my-store",
        RedisConnector(hostname="localhost", port=6379),
        register=True,
    )

    return get_store("my-store")


def _get_device():
    """Get device to train on"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def _proxify_parameters(store, model, N):
    """
    Convert parent model parameters to proxies for N target networks.
    Data is evicted from proxystore after consumption by simulators.
    """
    return [
        [store.proxy(i.cpu().detach().numpy(), evict=True) for i in model.parameters()]
        for _ in range(N)
    ]


def _get_optimizer(model):
    """Prepare optimizer for parent model training"""
    return optim.Adadelta(model.parameters(), lr=0.1)


def _get_summed_grads(grads):
    """Sum gradients from simulators"""
    summed_grads = [torch.zeros_like(i) for i in grads[0]]
    for grad in grads:
        for i in range(len(summed_grads)):
            summed_grads[i] += grad[i]
    return summed_grads


@persistent_input_fields(["local_gradients"])
@output_data([("parameters", object, (8,))])
def parent_model_trainer(H, _, gen_specs, libE_info):
    """
    Maintain a parent CNN that is trained using summed gradients from the
    simulators. Optimized parameters are streamed back to the simulators.
    """

    store = _connect_to_store()
    device = _get_device()

    simulators = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    N = gen_specs["user"]["num_networks"]

    model = Net().to(device)
    model.optimize = _optimize
    optimizer = _get_optimizer(model)

    while True:
        output = np.zeros(N, dtype=gen_specs["out"])
        if not initial_complete:
            output_parameters = _proxify_parameters(store, model, N)
            output["parameters"][:N] = output_parameters
            initial_complete = True
        else:
            tag, Work, calc_in = simulators.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            grad_proxies = calc_in["local_gradients"]
            grads = [[torch.from_numpy(np.array(i)) for i in j] for j in grad_proxies]
            summed_grads = _get_summed_grads(grads)
            _optimize(model, device, summed_grads, optimizer)
            output_parameters = _proxify_parameters(store, model, N)
            output["parameters"][:N] = output_parameters

        simulators.send(output)

    return [], {}, FINISHED_PERSISTENT_GEN_TAG
