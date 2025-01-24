import sys
import numpy as np
from itertools import chain

from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

import torch
import torch.optim as optim

# https://stackoverflow.com/questions/75012448/optimizer-step-not-updating-model-weights-parameters

def _create_new_parameters(grads, params):
    optimizer = optim.Adam(params, lr=1.0)
    optimizer.zero_grad()
    for i, param in enumerate(params):
        param.grad = grads[i].clone().detach()
    [optimizer.step() for _ in range(10)]
    return params


@persistent_input_fields(["grads", "output_parameters"])
@output_data([("input_parameters", object, (8,))])
def optimize_cnn(H, _, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    tag = None

    while True:
        History = np.zeros(1, dtype=gen_specs["out"])
        if not initial_complete:
            History["input_parameters"] = 0  # will be disregarded by first persis sim anyway
            initial_complete = True
        else:
            tag, Work, calc_in = ps.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            grads = calc_in["grads"][0]
            params = calc_in["output_parameters"][0]
            params = [torch.from_numpy(i) for i in params]
            grads = [torch.from_numpy(i) for i in grads]
            History["input_parameters"] = _create_new_parameters(grads, params)

        ps.send(History)
        
    return [], {}, FINISHED_PERSISTENT_GEN_TAG
