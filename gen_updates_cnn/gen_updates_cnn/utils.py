import torch
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, get_store
from proxystore.proxy import Proxy, is_resolved


def _connect_to_store():
    """Connect to proxystore redis server"""
    store = Store(
        "my-store",
        RedisConnector(hostname="localhost", port=6379),
        register=True,
    )

    return get_store("my-store")


def _get_device():
    """Get device to train on"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
