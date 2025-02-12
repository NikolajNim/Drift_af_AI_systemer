import torch
import pytest
from train import train_model

# Define expected dimensions
BATCH_SIZE = 32
NUM_CLASSES = 10

@pytest.mark.parametrize("batch_size, num_classes", [(BATCH_SIZE, NUM_CLASSES)])
def test_output_dimensions(batch_size, num_classes):
    model, output = train_model(batch_size=batch_size, num_classes=num_classes)

    # Check that output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor"

    # Check output shape
    expected_shape = torch.Size([batch_size, num_classes])
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"

    print("Test passed: Output dimensions are correct!")

