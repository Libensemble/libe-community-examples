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

def _train(self, device, train_loader, grads, optimizer, epochs, num_networks):
    
    for param, new_grad in zip(model.parameters(), grads):
        param.grad = torch.tensor(new_grad) / num_networks
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = self(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

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

def _proxify_parameters(store, model):
    return [store.proxy(i.cpu().detach().numpy(), evict=True) for i in model.parameters()]

def _get_train_loader():
    from torchvision import datasets, transforms
    train_kwargs = {"batch_size": args.batch_size}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    return train_loader

def _get_optimizer(model):
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    return optimizer

def _get_summed_grads(grads):
    pass

@persistent_input_fields(["local_gradients"])
@output_data([("parameters", object, (8,))])
def network_sync(H, _, gen_specs, libE_info):
        
    store = _connect_to_store()
    device = _get_device()

    simulators = PersistentSupport(libE_info, EVAL_GEN_TAG)
    initial_complete = False
    N = gen_specs["user"]["num_networks"]
    
    model = Net().to(device)
    model._train = _train
    train_loader = _get_train_loader()
    optimizer = _get_optimizer(model)

    while True:
        output = np.zeros(N, dtype=gen_specs["out"])
        if not initial_complete:
            output_parameters = _proxify_parameters(store, model)
            output["parameters"][:N] = output_parameters * N
            initial_complete = True
        else:
            tag, Work, calc_in = simulators.recv()
            if tag in [PERSIS_STOP, STOP_TAG]:
                break

            grads = calc_in["local_gradients"][0]
            grads = [torch.from_numpy(i) for i in grads]
            summed_grads = _get_summed_grads(grads)
            model._train(device, train_loader, grads, optimizer, 3, N)
            output_parameters = _proxify_parameters(store, model)
            output["parameters"][:N] = output_parameters * N

        simulators.send(output)
        
    return [], {}, FINISHED_PERSISTENT_GEN_TAG
