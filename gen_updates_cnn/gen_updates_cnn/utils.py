import torch
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, get_store
from proxystore.proxy import Proxy, is_resolved
from torchvision import datasets, transforms

def _connect_to_store(hostname):
    """Connect to proxystore redis server"""
    store = Store(
        "my-store",
        RedisConnector(hostname=hostname, port=6379),
        register=True,
    )

    return get_store("my-store")


def _get_device(info, is_generator=False):
    """Get device to train on"""
    if torch.cuda.is_available():
        worker_id = int(info["worker_id"])
        if is_generator:  # use GPU 1 for generator
            device_id = 1
        else:  # use GPU 1 for simulator, GPU N for simulator N
            device_id = worker_id
        device = torch.device("cuda:" + str(device_id))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def _get_datasets(worker_id, num_networks):
    """Get datasets for training and testing, splitting into even chunks for each worker"""
    # TODO: refactor this

    train_kwargs = {"batch_size": 1000, "shuffle": True}
    test_kwargs = {"batch_size": 5000, "shuffle": True}

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