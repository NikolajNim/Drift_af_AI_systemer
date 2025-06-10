import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import yaml
import time
import os # Import os for checking CUDA availability

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=config['model']['num_classes']):
        super(ModifiedResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify first conv layer to accept grayscale images
        # Ensure it's 1 input channel for FashionMNIST
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


def train():
    # Device configuration
    # Ensure CUDA is available for AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: CUDA not available. Mixed precision will not be used.")
        use_amp = False
    else:
        use_amp = config['training'].get('use_amp', False) # Get from config, default to False
        if use_amp:
            print("Using Automatic Mixed Precision (AMP).")
        else:
            print("AMP is disabled in config. Training without mixed precision.")


    # Hyperparameters
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    initial_lr = config['training']['initial_lr']

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    # Ensure dataset path exists or handle download
    if not os.path.exists(config['data']['dataset_path']):
        print(f"Dataset path '{config['data']['dataset_path']}' not found. Attempting to download.")
    dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'],
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model and move to device
    model = ModifiedResNet50().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Initialize GradScaler for mixed precision training
    # Only initialize if AMP is to be used
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                # Autocast context manager for mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Scale the loss and call backward()
                scaler.scale(loss).backward()
                # Unscale gradients and call optimizer.step()
                scaler.step(optimizer)
                # Update the scale for the next iteration
                scaler.update()
            else:
                # Regular training without mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "best_fashion_mnist_resnet.pth")


if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
