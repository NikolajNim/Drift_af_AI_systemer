import torch
import torchvision
from torchvision import transforms
import torch_tensorrt
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import yaml

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load your trained model (replace with your trained ResNet50)
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=config['model']['num_classes']):
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
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# Load the trained model (assuming you've already trained and saved it)
model = ModifiedResNet50(num_classes=config['model']['num_classes'])
model.load_state_dict(torch.load('final_fashion_mnist_resnet.pth'))
model.eval()

# Define the dataset and DataLoader for Fashion MNIST
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ResNet50
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust normalization as needed
])

testing_dataset = torchvision.datasets.FashionMNIST(
    root=config['data']['dataset_path'],
    train=False,
    transform=transform,
    download=True
)

testing_dataloader = DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=1
)

# Set up calibrator
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",  # Path to store calibration data
    use_cache=False,  # Set to True to reuse the cache if you have a cached file
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,  # Calibration algorithm
    device=torch.device("cuda:0"),  # Use GPU for calibration
)

# Compile the model with TensorRT (post-training quantization)
trt_mod = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],  # Input size adjusted to 224x224 for ResNet50
    enabled_precisions={torch.float, torch.half, torch.int8},  # Use FP16 and INT8 quantization
    calibrator=calibrator,  # Use the calibrator
    device={
        "device_type": torch_tensorrt.DeviceType.GPU,  # Use GPU for inference
        "gpu_id": 0,  # GPU ID (if multiple GPUs, adjust accordingly)
        "dla_core": 0,  # If using a Deep Learning Accelerator core
        "allow_gpu_fallback": False,  # Disable GPU fallback
        "disable_tf32": False,  # Disable TF32 if necessary
    }
)

# Save the quantized model
torch.save(trt_mod.state_dict(), "quantized_fashion_mnist_trt_model.pth")

print("Quantized model saved as 'quantized_fashion_mnist_trt_model.pth'")
