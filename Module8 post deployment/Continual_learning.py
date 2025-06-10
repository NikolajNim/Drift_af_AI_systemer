import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Task 1: Initial Training on MNIST digits 0-4

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter for digits 0-4
train_indices = [i for i, label in enumerate(train_dataset.targets) if label in range(5)]
test_indices = [i for i, label in enumerate(test_dataset.targets) if label in range(5)]

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("--- Task 1: Initial Training on digits 0-4 (PyTorch) ---")

# Train the model
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Initial training accuracy on digits 0-4: {accuracy:.4f}')

# --- Task 2: Naïve Sequential Learning on digits 5-9 ---
print("\n--- Task 2: Naïve Sequential Learning on digits 5-9 ---")

# Prepare data for the second task (digits 5-9)
train_indices_new = [i for i, label in enumerate(train_dataset.targets) if label in range(5, 10)]
test_indices_new = [i for i, label in enumerate(test_dataset.targets) if label in range(5, 10)]

train_subset_new = Subset(train_dataset, train_indices_new)
test_subset_new = Subset(test_dataset, test_indices_new)

train_loader_new = DataLoader(train_subset_new, batch_size=64, shuffle=True)
test_loader_new = DataLoader(test_subset_new, batch_size=64, shuffle=False)

# Function to evaluate the model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Track performance on the old task (digits 0-4)
old_task_accuracies = []
initial_old_task_accuracy = evaluate_model(model, test_loader)
old_task_accuracies.append(initial_old_task_accuracy)
print(f"Accuracy on digits 0-4 before new training: {initial_old_task_accuracy:.4f}")


# Train on the new task and evaluate on the old task after each epoch
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader_new:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate on the old task
    model.eval()
    accuracy_old_task = evaluate_model(model, test_loader)
    old_task_accuracies.append(accuracy_old_task)
    print(f"Epoch {epoch+1}/{epochs}, New Task Loss: {running_loss/len(train_loader_new):.4f}, Accuracy on 0-4: {accuracy_old_task:.4f}")


# Final evaluation
accuracy_old = evaluate_model(model, test_loader)
accuracy_new = evaluate_model(model, test_loader_new)

print(f"\nFinal accuracy on old digits (0-4): {accuracy_old:.4f}")
print(f"Final accuracy on new digits (5-9): {accuracy_new:.4f}")

print("\nPerformance drop on digits 0-4 (catastrophic forgetting):")
for i, acc in enumerate(old_task_accuracies):
    if i == 0:
        print(f"Before new training: {acc:.4f}")
    else:
        print(f"After epoch {i}: {acc:.4f}")

# --- Task 3: Experience Replay ---
print("\n--- Task 3: Experience Replay Solution ---")

# Re-initialize the model to start fresh from Task 1
model_er = Net()
criterion_er = nn.CrossEntropyLoss()
optimizer_er = optim.Adam(model_er.parameters())

print("\nRe-training model for Experience Replay on digits 0-4...")
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer_er.zero_grad()
        outputs = model_er(images)
        loss = criterion_er(outputs, labels)
        loss.backward()
        optimizer_er.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

initial_accuracy_er = evaluate_model(model_er, test_loader)
print(f"Initial accuracy on digits 0-4: {initial_accuracy_er:.4f}")


# Create a memory buffer with a subset of old task data
buffer_size = 200
replay_indices = torch.randperm(len(train_subset))[:buffer_size]
replay_buffer_dataset = Subset(train_subset.dataset, [train_subset.indices[i] for i in replay_indices])

# Combine new task data with replay buffer
combined_dataset = torch.utils.data.ConcatDataset([train_subset_new, replay_buffer_dataset])
train_loader_combined = DataLoader(combined_dataset, batch_size=64, shuffle=True)


print("\nTraining on digits 5-9 with Experience Replay...")
# Train on the new task with experience replay
epochs = 5
for epoch in range(epochs):
    model_er.train()
    running_loss = 0.0
    for images, labels in train_loader_combined:
        optimizer_er.zero_grad()
        outputs = model_er(images)
        loss = criterion_er(outputs, labels)
        loss.backward()
        optimizer_er.step()
        running_loss += loss.item()
    
    model_er.eval()
    accuracy_old_task_er = evaluate_model(model_er, test_loader)
    print(f"Epoch {epoch+1}/{epochs}, Combined Loss: {running_loss/len(train_loader_combined):.4f}, Accuracy on 0-4: {accuracy_old_task_er:.4f}")


# Final evaluation
accuracy_old_er = evaluate_model(model_er, test_loader)
accuracy_new_er = evaluate_model(model_er, test_loader_new)

print(f"\nFinal accuracy on old digits (0-4) with Experience Replay: {accuracy_old_er:.4f}")
print(f"Final accuracy on new digits (5-9) with Experience Replay: {accuracy_new_er:.4f}")
