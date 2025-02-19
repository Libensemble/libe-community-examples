import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator

import torch
import torch.nn.functional as F

from .mnist.nn import Net
from .utils import _connect_to_store, _get_device, _get_datasets


def _proxify_gradients(store, grads):
    """Convert resulting gradients to proxies for parent model."""
    return [store.proxy(i.cpu().detach().numpy(), evict=True) for i in grads]


def _update_parameters(model, device, params):
    """Update sim model parameters"""
    for (param, value) in zip(model.parameters(), params):
        param.data = torch.from_numpy(np.array(value)).to(device)


@input_fields(["parameters"])
@persistent_input_fields(["parameters"])
@output_data([("local_gradients", object, (8,))])
def mnist_training_sim(InitialData, _, sim_specs, info):
    """
    Maintain a child CNN that is trained using summed gradients from an optimized
    parent model on the generator. Gradients are streamed to the generator, and
    updated parameters are streamed back to the simulators.
    """

    store = _connect_to_store(sim_specs["user"]["proxystore_hostname"])
    device = _get_device()

    generator = ToGenerator(info, EVAL_SIM_TAG)
    workerID = info["workerID"]
    num_networks = sim_specs["user"]["num_networks"]


    model = Net(InitialData["parameters"][0]).to(device)
    model.train()
    train_loader, test_loader = _get_datasets(workerID, num_networks)

    for epoch in range(1, sim_specs["user"]["max_epochs"] + 1):
        print(f"Sim {workerID}: Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward(retain_graph=True)
            print(
                f"Sim {workerID}: [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.3f}"
            )

            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

            Output = np.zeros(1, dtype=sim_specs["out"])
            Output["local_gradients"] = _proxify_gradients(store, grads)

            tag, _, calc_in = generator.send_recv(Output)
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            _update_parameters(model, device, calc_in["parameters"][0])

            model.zero_grad()

    model.eval()
    model.test_model(device, test_loader)
    return None, {}, FINISHED_PERSISTENT_SIM_TAG