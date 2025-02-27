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
        worker_id = int(info["workerID"])
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
