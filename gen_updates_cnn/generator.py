import sys
import numpy as np
from itertools import chain

from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_GEN_TAG, WORKER_DONE
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport

import torch
import torch.optim as optim


def _create_new_weights(loss, grad, params):
    optimizer = optim.Adadelta(chain(params), lr=1.0)
    optimizer.zero_grad(set_to_none=True)
    [optimizer.step() for _ in range(100)]
    # now what...?
    return 0


@persistent_input_fields(["loss", "grad", "parameters"])
@output_data([("weights", object)])
def optimize_cnn(H, _, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    tag = None

    while True:
        History = np.zeros(1, dtype=gen_specs["out"])
        if not initial_complete:
            # We somehow need to send something, the sim inits itself
            History["weights"] = 0  # will be disregarded by first persis sim anyway
            initial_complete = True
        else:
            tag, Work, calc_in = ps.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            loss = calc_in["loss"]
            grad = calc_in["grad"]
            params = calc_in["parameters"][0]
            params = [torch.from_numpy(i) for i in params]
            History["weights"] = _create_new_weights(loss, grad, params)

        ps.send(History)
