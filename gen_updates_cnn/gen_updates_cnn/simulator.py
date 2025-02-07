import numpy as np
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE
from libensemble.specs import output_data, input_fields, persistent_input_fields
from libensemble.tools.persistent_support import PersistentSupport as ToGenerator

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from .mnist.nn import Net
from .utils import _connect_to_store, _get_device


def _get_datasets(worker_id, num_networks):
    """Get datasets for training and testing, splitting into even chunks for each worker"""
    # TODO: refactor this

    train_kwargs = {"batch_size": 64, "shuffle": True}
    test_kwargs = {"batch_size": 250, "shuffle": True}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    local_train_dataset_size = len(dataset1) // num_networks
    local_test_dataset_size = len(dataset2) // num_networks

    if worker_id is not None:
        start_index_mult = worker_id - 1

        start_index_train = start_index_mult * local_train_dataset_size
        end_index_train = start_index_train + local_train_dataset_size

        start_index_test = start_index_mult * local_test_dataset_size
        end_index_test = start_index_test + local_test_dataset_size

        dataset1 = torch.utils.data.Subset(
            dataset1, range(start_index_train, end_index_train)
        )
        dataset2 = torch.utils.data.Subset(
            dataset2, range(start_index_test, end_index_test)
        )

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


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
    Run CNN with parameters from parent model. Send gradients to parent model.
    """

    workerID = info["workerID"]
    num_networks = sim_specs["user"]["num_networks"]

    generator = ToGenerator(info, EVAL_SIM_TAG)
    store = _connect_to_store()
    device = _get_device()

    train_loader, test_loader = _get_datasets(workerID, num_networks)

    model = Net(InitialData["parameters"][0]).to(device)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward(retain_graph=True)

        if batch_idx % 10 == 0:
            print(
                f"Sim {workerID}: [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.2f}"
            )

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

        Output = np.zeros(1, dtype=sim_specs["out"])
        Output["local_gradients"] = _proxify_gradients(store, grads)

        tag, _, calc_in = generator.send_recv(Output)
        if tag in [PERSIS_STOP, STOP_TAG]:
            model.eval()
            model.test_model(device, test_loader)
            return None, {}, WORKER_DONE

        _update_parameters(model, device, calc_in["parameters"][0])
        model.zero_grad()

    model.eval()
    model.test_model(device, test_loader)
    return None, {}, WORKER_DONE