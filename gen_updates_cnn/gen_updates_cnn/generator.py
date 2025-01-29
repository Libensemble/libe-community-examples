import sys
import numpy as np
from itertools import chain

from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

import torch
import torch.optim as optim

# https://stackoverflow.com/questions/75012448/optimizer-step-not-updating-model-weights-parameters

def optimize_net(params, grads):
    # GOING TO SUM ALL LOCAL GRADIENTS HERE.
    optimizer = optim.Adadelta(params, lr=1.0)
    optimizer.zero_grad()
    for i, param in enumerate(params):
        param.grad = grads[i].clone().detach()
    # DONT STEP OPTIMIZER MULTIPLE TIMES WITH SAME GRADIENTS
    optimizer.step()
    return params

@persistent_input_fields(["local_gradients"])
@output_data([("summed_gradients", object, (8,))])
def sum_all_grads(H, _, gen_specs, libE_info):

    Simulators = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    N = gen_specs["user"]["num_networks"]

    while True:
        SummedGrads = np.zeros(N, dtype=gen_specs["out"])
        if not initial_complete:
            SummedGrads["summed_gradients"] = 0  # will be disregarded by first persis sim anyway
            initial_complete = True
        else:
            tag, Work, calc_in = Simulators.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            grads = calc_in["local_gradients"][0]
            grads = [torch.from_numpy(i) for i in grads]
            SummedGrads["summed_gradients"] = _sum_all_gradients(N, grads)

        Simulators.send(SummedGrads)
        
    return [], {}, FINISHED_PERSISTENT_GEN_TAG
