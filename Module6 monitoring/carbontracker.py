import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from carbontracker.tracker import CarbonTracker

# --- Configuration ---
# Hardcoded configuration for simplicity, inspired by config.yaml
config = {
    'model': {
        'num_classes': 10,
        'dropout_rate': 0.5,
    },
    'training': {
        'num_epochs': 2,  # Reduced for a quick test
        'batch_size': 64,
        'initial_lr': 0.001,
    },
    'data': {
        'dataset_path': './data',
        'train_val_split': 0.8,
        'num_workers': 2,
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# --- Model Definition ---
# Replicating the ModifiedResNet50 from train.py
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=config['model']['num_classes']):
        super(ModifiedResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# --- Training with Carbon Tracking ---
def train_with_carbon_tracking():
    print("--- Training with CarbonTracker ---")

    # Initialize CarbonTracker for training
    tracker = CarbonTracker(epochs=config['training']['num_epochs'], monitor_epochs=config['training']['num_epochs'])

    # Device configuration
    device = torch.device(config['device'])

    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset_full = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'], train=True, transform=transform, download=True
    )
    train_size = int(config['data']['train_val_split'] * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['data']['num_workers']
    )

    # Model, loss, and optimizer
    model = ModifiedResNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['initial_lr'])

    # Training loop wrapped by the tracker
    for epoch in range(config['training']['num_epochs']):
        tracker.epoch_start()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        tracker.epoch_end()

    # Save model for evaluation
    torch.save(model.state_dict(), 'monitoring_model.pth')
    print("Training finished.")


# --- Inference with Carbon Tracking ---
def evaluate_with_carbon_tracking():
    print("\n--- Inference with CarbonTracker ---")

    # Initialize a new tracker for inference. Let's simulate 1000 inference requests.
    n_inferences = 1000
    tracker = CarbonTracker(epochs=n_inferences, monitor_epochs=n_inferences, verbose=2)

    # Device and model
    device = torch.device(config['device'])
    model = ModifiedResNet50().to(device)
    model.load_state_dict(torch.load('monitoring_model.pth'))
    model.eval()

    # Data loader for test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'], train=False, transform=transform, download=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)  # Batch size 1 for single inference

    # Simulate inference requests
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            if i >= n_inferences:
                break
            tracker.epoch_start()
            image = image.to(device)
            _ = model(image)
            tracker.epoch_end()

    print("Inference simulation finished.")


# --- Cost Prediction ---
def predict_costs():
    print("\n--- Cost and Carbon Prediction ---")
    # This is a simplified estimation.
    # Carbontracker logs results to `carbontracker.log`. Let's assume we read it.
    # For this example, let's use some placeholder values based on typical runs.
    # These would be replaced by parsing the carbontracker output.

    # Placeholder values (replace with actuals from carbontracker.log)
    training_energy_kwh = 0.05  # in kWh
    training_co2_g = 20  # in grams
    inference_energy_kwh_per_request = 0.00001
    inference_co2_g_per_request = 0.004

    # Assumptions
    electricity_cost_usd_per_kwh = 0.15
    requests_per_day = 10000

    # Yearly Cost Calculation
    training_cost = training_energy_kwh * electricity_cost_usd_per_kwh

    yearly_inference_requests = requests_per_day * 365
    yearly_inference_energy_kwh = yearly_inference_requests * inference_energy_kwh_per_request
    yearly_inference_cost = yearly_inference_energy_kwh * electricity_cost_usd_per_kwh

    yearly_inference_co2_kg = (yearly_inference_requests * inference_co2_g_per_request) / 1000

    print(f"Estimated cost for one training run: ${training_cost:.2f}")
    print(f"Estimated yearly inference cost: ${yearly_inference_cost:.2f}")
    print(f"Estimated yearly CO2 emissions from inference: {yearly_inference_co2_kg:.2f} kg")


if __name__ == '__main__':
    train_with_carbon_tracking()
    evaluate_with_carbon_tracking()
    predict_costs()
