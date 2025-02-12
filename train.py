import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from ptflops import get_model_complexity_info

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet50, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify final fc layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Data preprocessing
# Returns what type of transformation should be used on the data
def get_tranformation():
    print('Choose transformation for the data')
    print('1 for standard tranformation | 2 for transformation with data augmentation')
    while True:
        choice = input('Write choice here: ').strip()

        if choice == '1':
            print('Standard transformation chosen')
            tranform_name = 'standard'
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            return tranform_name, transform
        elif choice == '2':
            print('Tranformation with data augmentation chosen')
            tranform_name = 'data_aug'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomApply([transforms.RandomRotation(20)], p=0.2),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            return tranform_name, transform
        else:
            print('Wrong input, try again')


def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    initial_lr = 0.0001  # Lower learning rate for fine-tuning
    
    # Get user-selected transformation
    transform_name, transform = get_tranformation()

    # Load the Fashion MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    # Create validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize the model
    model = ModifiedResNet50().to(device)

    # Calculate FLOPS and model parameters
    macs, params = get_model_complexity_info(model, (1, 224, 224), print_per_layer_stat=False, as_strings=False)
    flops = 2 * macs
    print(f'Model Parameters: {params} | FLOPS: {flops}')

    summary_input = (1, 1, 224, 244)
    summary(model, summary_input)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for different layers
    # Higher learning rate for new layers, lower for pretrained layers
    params_new = [p for n, p in model.named_parameters() if 'fc' in n]
    params_pretrained = [p for n, p in model.named_parameters() if 'fc' not in n]
    
    optimizer = optim.Adam([
        {'params': params_pretrained, 'lr': initial_lr},
        {'params': params_new, 'lr': initial_lr * 10}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        # Update learning rate
        scheduler.step(val_epoch_accuracy)
        
        # Save best model
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            torch.save(model.state_dict(), f'best_fashion_mnist_resnet_{transform_name}.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%')
        print('-' * 50)
    
    

    # Plot training metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_{transform_name}.png')
    plt.close()

if __name__ == '__main__':
    train_model()
