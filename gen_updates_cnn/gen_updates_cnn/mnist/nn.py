import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys

import numpy as np


class Net(nn.Module):
    def __init__(self, input_parameters=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        if input_parameters is not None:
            for param, value in zip(self.parameters(), input_parameters):
                param.data = torch.from_numpy(np.array(value))

        self.total_train_loss = 0

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

    def train_model(self, args, device, train_loader, epoch, worker_id):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = self(data)
            loss = F.nll_loss(output, target)
            if batch_idx % args.log_interval == 0:
                print(
                    "SIMULATOR {}: Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        worker_id,
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                if args.dry_run:
                    break
            loss.backward(retain_graph=True)
            
            # send:recv
            
            # if input_parameters is not None:
            #     for param, value in zip(self.parameters(), input_parameters):
            #         param.data = torch.from_numpy(np.array(value))
            
            break  # we want new parameters from the gen after each batch
            # update new parameters based on optimized parameters from gen
        grads = torch.autograd.grad(loss, self.parameters(), retain_graph=True)
        return grads

    def test_model(self, device, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.enable_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nSIMULATOR Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        , flush=True)


def main(parameters=None, worker_id=None, num_networks=1):
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=250,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args([])  # avoid conflict with libensemble args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    seed = worker_id if worker_id is not None else args.seed

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size, "shuffle": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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

        dataset1 = torch.utils.data.Subset(dataset1, range(start_index_train, end_index_train))
        dataset2 = torch.utils.data.Subset(dataset2, range(start_index_test, end_index_test))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if parameters is None:
        model = Net().to(device)
    else:
        model = Net(parameters).to(device)

    for epoch in range(1, args.epochs + 1):
        grads = model.train_model(args, device, train_loader, epoch, worker_id)
        model.test_model(device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return grads


if __name__ == "__main__":
    main()
