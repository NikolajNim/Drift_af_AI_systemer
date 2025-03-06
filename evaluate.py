import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from train import ModifiedResNet50
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def evaluate_model():
    # Initialize wandb
    wandb.init(project="fashion-mnist-resnet", job_type="evaluation", config=config)

    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'],
        train=False,
        transform=transform,
        download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    # Initialize the model
    model = ModifiedResNet50().to(device)

    # Load the trained model
    model.load_state_dict(torch.load('final_fashion_mnist_resnet.pth'))
    model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Variables to compute metrics
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy and loss
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")

    # Log results to wandb
    wandb.log({
        'test_accuracy': accuracy,
        'test_loss': avg_loss
    })

    # Save the results to a file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")

if __name__ == '__main__':
    evaluate_model()