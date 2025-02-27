import numpy as np
from libensemble.message_numbers import (
    PERSIS_STOP,
    STOP_TAG,
    EVAL_SIM_TAG,
    FINISHED_PERSISTENT_SIM_TAG,
)
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator

import torch
import torch.nn.functional as F

from .utils import _connect_to_store, _get_device, _get_datasets


def _proxify_gradients(store, grads):
    """Convert resulting gradients to proxies for parent model."""
    return [store.proxy(i.cpu().detach().numpy(), evict=True) for i in grads]


def _update_parameters(model, device, params):
    """Update sim model parameters"""
    for param, value in zip(model.parameters(), params):
        param.data = torch.from_numpy(np.array(value)).to(device)


def test_model(model, device, test_loader, workerID):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Sim {}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            workerID,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        ),
        flush=True,
    )


@input_fields(["parameters"])
@persistent_input_fields(["parameters"])
@output_data([("local_gradients", object, (8,))])
def model_trainer(InitialData, _, sim_specs, info):
    """
    Maintain a child CNN that is trained using summed gradients from an optimized
    parent model on the generator. Gradients are streamed to the generator, and
    updated parameters are streamed back to the simulators.
    """

    user = sim_specs["user"]

    store = _connect_to_store(user["proxystore_hostname"])
    device = _get_device(info, sim_specs)

    generator = ToGenerator(info, EVAL_SIM_TAG)
    workerID = info["workerID"]

    N = user["num_networks"]
    Net = user["model_definition"]

    model = Net().to(device)

    _update_parameters(model, device, InitialData["parameters"][0])

    train_loader, test_loader = _get_datasets(
        user["train_data"],
        user["test_data"],
        workerID,
        N,
        user["train_batch_size"],
        user["test_batch_size"],
    )

    for epoch in range(1, user["max_epochs"] + 1):
        print(f"Sim {workerID}: Starting epoch {epoch}")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward(retain_graph=True)
            print(
                f"Sim {workerID}: [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.3f}", flush=True
            )

            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

            Output = np.zeros(1, dtype=sim_specs["out"])
            Output["local_gradients"] = _proxify_gradients(store, grads)

            tag, _, calc_in = generator.send_recv(Output)
            if tag in [PERSIS_STOP, STOP_TAG]:
                print(f"Sim {workerID}: Instructed to stop. Testing model.", flush=True)
                test_model(model, device, test_loader, workerID)
                return None, {}, FINISHED_PERSISTENT_SIM_TAG

            _update_parameters(model, device, calc_in["parameters"][0])

            model.zero_grad()

        print(f"Sim {workerID}: Ending epoch {epoch}. Testing model.", flush=True)
        test_model(model, device, test_loader, workerID)

    return None, {}, FINISHED_PERSISTENT_SIM_TAG
