import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.specs import ExitCriteria, AllocSpecs, SimSpecs, GenSpecs
from gen_updates_cnn.simulator import model_trainer
from gen_updates_cnn.generator import parent_model_optimizer


class Net(nn.Module):
    def __init__(self, input_parameters=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x.retain_grad()
        output = F.log_softmax(x, dim=1)
        return output


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("../data", train=False, transform=transform)

PARENT_MODEL_DEVICE = "GPU"  # "CPU"
MAX_OPTIMIZER_STEPS = 100
STREAMING_DATABASE_HOST = "localhost"
NUM_CUDA_NODES = 2

settings = {
    "model_definition": Net,
    "train_data": dataset1,
    "test_data": dataset2,
    "train_batch_size": 1000,
    "test_batch_size": 5000,
    "max_epochs": 2,
    "proxystore_hostname": STREAMING_DATABASE_HOST,
    "num_nodes": NUM_CUDA_NODES,
    "parent_model_device": PARENT_MODEL_DEVICE,
}

if __name__ == "__main__":

    ensemble = Ensemble(parse_args=True)

    settings["num_networks"] = ensemble.nworkers

    ensemble.libE_specs.gen_on_manager = True

    sim_specs = SimSpecs(sim_f=model_trainer, user=settings)
    gen_specs = GenSpecs(gen_f=parent_model_optimizer, user=settings)
    alloc_specs = AllocSpecs(alloc_f=only_persistent_gens)

    ensemble.sim_specs = sim_specs
    ensemble.gen_specs = gen_specs
    ensemble.alloc_specs = alloc_specs
    ensemble.exit_criteria = ExitCriteria(sim_max=MAX_OPTIMIZER_STEPS)

    ensemble.run()
    if ensemble.is_manager:
        ensemble.save_output(__file__)
    sys.exit(0)
