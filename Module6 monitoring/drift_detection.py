from flask import Flask, Response
from prometheus_client import Gauge, Histogram, generate_latest
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from scipy.stats import ks_2samp
import time
import threading

# --- App & Metrics Definition ---
app = Flask(__name__)

PIPELINE_DURATION = Histogram('drift_pipeline_duration_seconds', 'Duration of the full drift detection pipeline')
DRIFT_DETECTED_GAUGE = Gauge('drift_detected', 'A binary gauge indicating if drift was detected (1=yes, 0=no)')
P_VALUE_GAUGE = Gauge('drift_p_value', 'The p-value from the KS-test for the feature that triggered drift detection')

# --- Configuration ---
config = {
    'training': {
        'num_epochs': 1,
        'batch_size': 128,
        'lr': 0.001,
    },
    'data': {
        'dataset_path': './data',
        'num_workers': 2,
    },
    'drift': {
        'noise_level': 0.01,
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
    for epoch in range(config['training']['num_epochs']):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Baseline model training finished.")
    return model


# --- Step 2: Extract Features ---
def get_features(model, data_loader):
    device = torch.device(config['device'])
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()

    all_features = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            features = feature_extractor(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# --- Step 3: Drift Detection ---
def detect_drift(ref_features, new_features):
    print("\n--- Performing drift detection ---")
    num_features = ref_features.shape[1]
    drift_detected = False
    p_value_for_drift = 1.0 # Default p-value if no drift is found

    for i in range(num_features):
        ks_stat, p_value = ks_2samp(ref_features[:, i], new_features[:, i])

        if p_value < config['drift']['p_value_threshold']:
            print(f"Drift detected in feature dimension {i+1} (p-value: {p_value:.4f})")
            drift_detected = True
            p_value_for_drift = p_value
            break

    if not drift_detected:
        print("No significant data drift detected.")

    return drift_detected, p_value_for_drift


def run_pipeline_background():
    """This function runs the pipeline and is meant to be called in a background thread."""
    print("--- Starting background drift detection pipeline ---")
    start_time = time.time()
    
    # Run the main pipeline logic
    run_pipeline()
    
    duration = time.time() - start_time
    PIPELINE_DURATION.observe(duration)
    print(f"--- Background pipeline finished in {duration:.2f}s ---")


def run_pipeline():
    """The main function that runs the entire pipeline."""
    # 1. Train model
    baseline_model = train_baseline_model()
    
    # 2. Create reference and drifted datasets
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
    
    # 3. Extract features
    print("\n--- Extracting features from reference data ---")
    reference_features = get_features(baseline_model, ref_loader)
    
    print("\n--- Extracting features from drifted data ---")
    new_features = get_features(baseline_model, drifted_loader)
    
    # 4. Detect drift
    drift_detected, p_value = detect_drift(reference_features, new_features)
    
    # 5. Update Prometheus metrics
    DRIFT_DETECTED_GAUGE.set(1 if drift_detected else 0)
    P_VALUE_GAUGE.set(p_value)


# --- Flask Routes ---
@app.route('/')
def index():
    return "Drift Detection App is running. Use /run to start the pipeline and /metrics to see results."

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain; version=0.0.4; charset=utf-8')

@app.route('/run')
def run_drift_detection_endpoint():
    # Run the pipeline in a background thread to avoid HTTP timeouts
    thread = threading.Thread(target=run_pipeline_background)
    thread.start()
    return "Drift detection pipeline started in the background. Check Docker logs for progress and Grafana for results."

if __name__ == '__main__':
    # Set initial gauge values
    DRIFT_DETECTED_GAUGE.set(0)
    P_VALUE_GAUGE.set(1)
    app.run(host='0.0.0.0', port=5001)