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


def _get_device(info, specs, is_generator=False):
    """Get device to train on"""
    if torch.cuda.is_available():
        num_nodes = specs["user"]["num_nodes"]
        worker_id = int(info["workerID"])
        if is_generator:  # use GPU 1 for generator
            device_id = 1
        else:  # use GPU N for simulator N
            device_id = (worker_id % num_nodes) + 1
        device = torch.device("cuda:" + str(device_id))
    elif torch.backends.mps.is_available():
        if not is_generator:  # use CPU for simulator
            device = torch.device("mps")
        elif specs["user"]["parent_model_device"] == "CPU":  # use CPU for generator
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if is_generator and specs["user"]["parent_model_device"] == "CPU":  # overwrite device if specified
        device = torch.device("cpu")
    print("I'm a generator" if is_generator else "I'm a simulator", "and I'm on", device)
    return device


def _get_datasets(
    train_dataset,
    test_dataset,
    worker_id,
    num_networks,
    train_batch_size,
    test_batch_size,
):
    """Get datasets for training and testing, splitting into even chunks for each worker"""

    local_train_dataset_size = len(train_dataset) // num_networks
    local_test_dataset_size = len(test_dataset) // num_networks

    if worker_id is not None:
        start_index_mult = worker_id - 1

        start_index_train = start_index_mult * local_train_dataset_size
        end_index_train = start_index_train + local_train_dataset_size

        start_index_test = start_index_mult * local_test_dataset_size
        end_index_test = start_index_test + local_test_dataset_size

        dataset1 = torch.utils.data.Subset(
            train_dataset, range(start_index_train, end_index_train)
        )
        dataset2 = torch.utils.data.Subset(
            test_dataset, range(start_index_test, end_index_test)
        )

    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=test_batch_size, shuffle=True
    )

    return train_loader, test_loader
