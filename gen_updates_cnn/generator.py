import sys
import numpy as np

from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_GEN_TAG, WORKER_DONE
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport


def _create_new_weights(loss, grad):
    # TODO: the fun stuff starts here...
    return 0


@persistent_input_fields(["loss", "grad"])
@output_data([("weights", object)])
def optimize_cnn(H, persis_info, gen_specs, libE_info):

    ps = PersistentSupport(persis_info, EVAL_GEN_TAG)
    initial_complete = False
    tag = None

    while True:
        if not initial_complete:
            # We somehow need to send something, the sim inits itself
            History = np.zeros(1, dtype=gen_specs["out"])
            History["weights"] = 0  # will be disregarded by first persis sim anyway
            initial_complete = True
        else:
            tag, Work, calc_in = ps.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            loss = Work["loss"]
            grad = Work["grad"]
            History["weights"] = _create_new_weights(loss, grad)

        ps.send(History)
