import sys
import numpy as np
from itertools import chain

from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

import torch
import torch.optim as optim

from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, get_store
from proxystore.proxy import Proxy, is_resolved

from .mnist.nn import Net

# grads = GET GRADIENTS and/or WEIGHTS
# proxy = store.proxy(grads)

# https://stackoverflow.com/questions/75012448/optimizer-step-not-updating-model-weights-parameters

def _connect_to_store():
    store = Store(
        "my-store",
        RedisConnector(hostname="localhost", port=6379),
        register=True,
    )

    return get_store("my-store")

def _get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def optimize_net(params, grads):
    # GOING TO SUM ALL LOCAL GRADIENTS HERE.
    optimizer = optim.Adadelta(params, lr=1.0)
    optimizer.zero_grad()
    for i, param in enumerate(params):
        param.grad = grads[i].clone().detach()
    # DONT STEP OPTIMIZER MULTIPLE TIMES WITH SAME GRADIENTS
    optimizer.step()
    return params

def _proxify_parameters(store, model):
    return [store.proxy(i.cpu().detach().numpy(), evict=True) for i in model.parameters()]

def _sum_all_gradients(N, grads):
    return [torch.sum(grads[i]) for i in range(N)]

@persistent_input_fields(["local_gradients"])
@output_data([("summed_gradients", object, (8,)), ("initial_parameters", object, (8,))])
def network_sync(H, _, gen_specs, libE_info):
        
    store = _connect_to_store()
    device = _get_device()

    simulators = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    N = gen_specs["user"]["num_networks"]
    
    model = Net().to(device)

    while True:
        output = np.zeros(N, dtype=gen_specs["out"])
        if not initial_complete:
            output_parameters = _proxify_parameters(store, model)
            output["initial_parameters"][:N] = output_parameters * N
            initial_complete = True
        else:
            tag, Work, calc_in = simulators.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            grads = calc_in["local_gradients"][0]
            grads = [torch.from_numpy(i) for i in grads]
            output["summed_gradients"] = _sum_all_gradients(N, grads)

        simulators.send(output)
        
    return [], {}, FINISHED_PERSISTENT_GEN_TAG
