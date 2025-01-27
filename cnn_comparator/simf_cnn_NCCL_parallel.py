import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from libensemble.message_numbers import PERSIS_STOP, STOP_TAG, EVAL_SIM_TAG, WORKER_DONE

# Define a Convolutional Neural Network
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 32 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 64 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2)  # 2x2 pooling
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Flattened features to 128 units
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # First convolution + activation
        x = torch.relu(self.conv2(x))  # Second convolution + activation
        x = self.pool(x)  # Pooling
        x = x.view(-1, 64 * 12 * 12)  # Flatten the features
        x = torch.relu(self.fc1(x))  # Fully connected layer
        return self.fc2(x)  # Output layer


def train(rank, group_size, epochs, batch_size, use_gpu):
    # Redirect stdout to a log file for each rank
    log_file = f"rank_{rank}_output.log"
    with open(log_file, "w") as f:
        os.dup2(f.fileno(), 1)  # Redirect stdout
        os.dup2(f.fileno(), 2)  # Redirect stderr
    # Initialize process group
    dist.init_process_group("nccl")

    # Toggle device based on use_gpu flag
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if use_gpu else "cpu")

    # Prepare dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # Initialize model and optimizer
    model = MNISTCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # Synchronize gradients across processes
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= group_size  # SH averaging?

            # SH TODO - consider inheriting dist (torch.distributed)
            # and overriding dist.all_reduce() to do libE stuff
            all_grads = []
            for param in model.parameters():
                all_grads.append(param.grad.detach().cpu().numpy().ravel())

            # libE lines
            H_o = np.zeros(1, dtype=sim_specs["out"])
            H_o["grads"] = np.concatenate(all_grads)
            tag, Work, calc_in = ps.send_recv(H_o)
            # SH TODO do we want final step - keep going till the sim is done!!!
            while tag not in [STOP_TAG, PERSIS_STOP]:
                offset = 0
                for param in model.parameters():
                    grad_shape = param.grad.shape
                    grad_size = param.grad.numel()

                    # Extract the corresponding section and reshape
                    param.grad.data.copy_(
                        torch.tensor(calc_in[offset:offset + grad_size].reshape(grad_shape),
                                    device=param.grad.device)
                    )
                    offset += grad_size

                # Now run optimzier on combined data
                optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}", flush=True)

    # Clean up process group
    dist.destroy_process_group()


def persis_cnn(H, persis_info, sim_specs, libE_info):

# def main():
    # Use environment variables set by torchrun
    # rank = int(os.environ["RANK"])
    # group_size = int(os.environ["WORLD_SIZE"])
    group_size = 4
    epochs = 3
    batch_size = 32
    # use_gpu = torch.cuda.is_available()
    use_gpu = True

    ps = PersistentSupport(libE_info, EVAL_SIM_TAG)
    processes = []

    # Run parallel on this node
    for rank in range(group_size):
        p = Process(target=train, args=(rank, group_size, epochs, batch_size, use_gpu, ps))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
