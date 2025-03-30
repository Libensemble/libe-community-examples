import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

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

def train(rank, world_size, epochs, batch_size, use_gpu):
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

    # Initialize model, wrap with DDP, and optimizer
    model = MNISTCNN().to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()]) if use_gpu else DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}", flush=True)

    # Clean up process group
    dist.destroy_process_group()

def main():
    # Use environment variables set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    epochs = 3
    batch_size = 32
    use_gpu = torch.cuda.is_available()

    train(rank, world_size, epochs, batch_size, use_gpu)

if __name__ == "__main__":
    main()
