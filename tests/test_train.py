import torch
import pytest
from train import ModifiedResNet50  # Import your model class
import yaml

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Expected settings
BATCH_SIZE = config["training"]["batch_size"]
NUM_CLASSES = config["model"]["num_classes"]
INPUT_SHAPE = (1, 224, 224)  # Single grayscale image input

@pytest.fixture
def model():
    """Fixture to initialize the model before tests."""
    return ModifiedResNet50(num_classes=NUM_CLASSES)

def test_model_forward_pass(model):
    """Test if the model can process a forward pass."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, *INPUT_SHAPE)  # Batch of test images
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor), "Output is not a tensor"
    assert output.shape == torch.Size([BATCH_SIZE, NUM_CLASSES]), f"Unexpected output shape: {output.shape}"

