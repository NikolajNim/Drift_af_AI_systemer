import torch
from train import ModifiedResNet50  # Import your custom model
import torch.quantization

# Create the model
model = ModifiedResNet50()

# Load the trained weights
state_dict = torch.load("final_fashion_mnist_resnet.pth",weights_only=False, map_location="cpu")
model.load_state_dict(state_dict)  # Should load perfectly if trained with ModifiedResNet50
model.eval()

# Apply dynamic quantization (only quantizes Linear layers, conv layers remain float)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the entire quantized model
torch.save(model_quantized, "final_fashion_mnist_resnet.pth")