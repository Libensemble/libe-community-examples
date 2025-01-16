import sys
import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport


from mnist.nn import main as run_cnn

@persistent_input_fields(["weights"])
@output_data([("loss", float), ("grad", object, (10, 128))])
def mnist_training_sim(H, _, sim_specs, info):

    ps = PersistentSupport(info, EVAL_SIM_TAG)
    tag = None

    Output = np.zeros(1, dtype=sim_specs['out'])

    grad, train_loss, test_loss = run_cnn()  # initial
    init_done = True
    Output["grad"] = grad
    Output["loss"] = train_loss + test_loss
    return Output