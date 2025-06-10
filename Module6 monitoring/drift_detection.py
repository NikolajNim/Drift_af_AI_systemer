import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm
import time

# --- Configuration ---
config = {
    'training': {
        'num_epochs': 2,
        'batch_size': 128,
        'lr': 0.001,
    },
    'data': {
        'dataset_path': './data',
        'num_workers': 2,
    },
    'drift': {
        'noise_level': 0.5,
        'p_value_threshold': 0.05,
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# --- Model Definition ---
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# --- Feature Extractor ---
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # Remove the final classification layer
        self.features = nn.Sequential(*list(original_model.resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# --- Step 1: Train Baseline Model ---
def train_baseline_model():
    print("--- Training baseline model ---")
    device = torch.device(config['device'])
    model = ModifiedResNet50().to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'], train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['data']['num_workers']
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    model.train()
    total_batches = len(train_loader)

    for epoch in range(config['training']['num_epochs']):
        epoch_start_time = time.time()
        running_loss = 0.0

        # Create progress bar for current epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["num_epochs"]}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar with current loss
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Batch': f'{batch_idx + 1}/{total_batches}'
            })

        epoch_time = time.time() - epoch_start_time
        final_avg_loss = running_loss / total_batches

        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average Loss: {final_avg_loss:.4f}")
        print("-" * 50)

    torch.save(model.state_dict(), 'baseline_model.pth')
    print("Baseline model training finished.")
    return model


# --- Step 2: Extract Features ---
def get_features(model, data_loader, desc="Extracting features"):
    device = torch.device(config['device'])
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()

    all_features = []

    # Add progress bar for feature extraction
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc=desc):
            images = images.to(device)
            features = feature_extractor(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# --- Step 3: Drift Detection ---
def detect_drift(ref_features, new_features):
    print("\n--- Performing drift detection ---")
    num_features = ref_features.shape[1]
    drift_detected = False

    # Add progress bar for drift detection
    for i in tqdm(range(num_features), desc="Checking feature dimensions for drift"):
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(ref_features[:, i], new_features[:, i])

        if p_value < config['drift']['p_value_threshold']:
            print(f"\nDrift detected in feature dimension {i + 1} (p-value: {p_value:.4f})")
            drift_detected = True
            break  # Stop at first sign of drift for efficiency

    if not drift_detected:
        print("\nNo significant data drift detected.")

    return drift_detected


if __name__ == '__main__':
    print(f"Using device: {config['device']}")
    print(f"Training configuration: {config['training']}")
    print("=" * 60)

    # 1. Train model
    baseline_model = train_baseline_model()

    # 2. Create reference and drifted datasets
    print("\n--- Setting up datasets ---")
    base_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    drift_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * config['drift']['noise_level']),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ref_dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'], train=False, transform=base_transform, download=True
    )
    drifted_dataset = torchvision.datasets.FashionMNIST(
        root=config['data']['dataset_path'], train=False, transform=drift_transform, download=True
    )

    ref_loader = DataLoader(ref_dataset, batch_size=config['training']['batch_size'])
    drifted_loader = DataLoader(drifted_dataset, batch_size=config['training']['batch_size'])

    print(f"Reference dataset size: {len(ref_dataset)}")
    print(f"Drifted dataset size: {len(drifted_dataset)}")

    # 3. Extract features
    print("\n--- Feature Extraction Phase ---")
    reference_features = get_features(baseline_model, ref_loader, "Extracting reference features")
    new_features = get_features(baseline_model, drifted_loader, "Extracting drifted features")

    print(f"Reference features shape: {reference_features.shape}")
    print(f"New features shape: {new_features.shape}")

    # 4. Detect drift
    detect_drift(reference_features, new_features)

    print("\n--- Process completed ---")