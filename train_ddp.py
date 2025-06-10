import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet50, ResNet50_Weights
import yaml
import wandb
import os
import time

# Load configuration from YAML file
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def setup_distributed(rank, world_size):
    """Initialize process group for distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Set an available port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training process."""
    dist.destroy_process_group()


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=config['model']['num_classes']):
        super(ModifiedResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify first conv layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def train(rank, world_size):
    """Train the model using Distributed Data Parallel (DDP) with mixed precision."""
    setup_distributed(rank, world_size)

    # Device configuration
    device = torch.device(f"cuda:{rank}")

    # Hyperparameters
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size'] // world_size  # Adjust batch size per GPU
    initial_lr = config['training']['initial_lr']

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'],
        train=True,
        transform=transform,
        download=True
    )

    # Create sampler for distributed training
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    # Initialize model and move to GPU
    model = ModifiedResNet50().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensure different shuffling for each epoch
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f'Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Save model only on rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), "best_fashion_mnist_resnet.pth")

    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # Number of available GPUs

    start_time = time.time()

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total training time: {elapsed_time:.2f} seconds")