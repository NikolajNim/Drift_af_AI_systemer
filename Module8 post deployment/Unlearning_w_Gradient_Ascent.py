import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Task 1: Train a Classifier on Full MNIST ---

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

print("--- Task 1: Training a Classifier on Full MNIST (0-9) ---")

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

# Function to evaluate the model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f'\nBaseline accuracy on all digits (0-9): {accuracy:.4f}')


# --- Task 2: Targeted Unlearning of a specific digit (e.g., "7") ---
print("\n--- Task 2: Targeted Unlearning using Gradient Ascent ---")

# Define the class to forget
forget_class = 7

# Create a DataLoader for the forget class
forget_indices = [i for i, label in enumerate(train_dataset.targets) if label == forget_class]
forget_subset = torch.utils.data.Subset(train_dataset, forget_indices)
forget_loader = DataLoader(forget_subset, batch_size=64, shuffle=True)

# Separate loaders for evaluating forget and retain classes
retain_indices_test = [i for i, label in enumerate(test_dataset.targets) if label != forget_class]
forget_indices_test = [i for i, label in enumerate(test_dataset.targets) if label == forget_class]
retain_subset_test = torch.utils.data.Subset(test_dataset, retain_indices_test)
forget_subset_test = torch.utils.data.Subset(test_dataset, forget_indices_test)
retain_loader_test = DataLoader(retain_subset_test, batch_size=64, shuffle=False)
forget_loader_test = DataLoader(forget_subset_test, batch_size=64, shuffle=False)

print(f"Accuracy on class {forget_class} before unlearning: {evaluate_model(model, forget_loader_test):.4f}")
print(f"Accuracy on other classes before unlearning: {evaluate_model(model, retain_loader_test):.4f}")


# Perform gradient ascent on the forget class
unlearn_epochs = 3
model.train()
for epoch in range(unlearn_epochs):
    running_loss = 0.0
    for images, labels in forget_loader:
        optimizer.zero_grad()
        outputs = model(images)
        # We want to maximize the loss for the forget class
        loss = -criterion(outputs, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Unlearning Epoch {epoch+1}/{unlearn_epochs}, Ascent Loss: {-running_loss/len(forget_loader):.4f}")


# Evaluate the model after unlearning
print("\n--- Evaluation After Unlearning ---")
accuracy_forget = evaluate_model(model, forget_loader_test)
accuracy_retain = evaluate_model(model, retain_loader_test)
accuracy_overall = evaluate_model(model, test_loader)

print(f"Accuracy on forgotten class ({forget_class}): {accuracy_forget:.4f}")
print(f"Accuracy on retained classes: {accuracy_retain:.4f}")
print(f"Overall accuracy after unlearning: {accuracy_overall:.4f}")

# --- Task 3: Unlearning with Retain/Forget Balancing ---
print("\n--- Task 3: Unlearning with Retain/Forget Balancing ---")

# Re-initialize and train a fresh model for this task
model_balanced = Net()
criterion_balanced = nn.CrossEntropyLoss()
optimizer_balanced = optim.Adam(model_balanced.parameters())

print("\nRe-training a fresh model...")
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer_balanced.zero_grad()
        outputs = model_balanced(images)
        loss = criterion_balanced(outputs, labels)
        loss.backward()
        optimizer_balanced.step()

print(f"Baseline accuracy: {evaluate_model(model_balanced, test_loader):.4f}")


# Create a DataLoader for the retain class training data
retain_indices_train = [i for i, label in enumerate(train_dataset.targets) if label != forget_class]
retain_subset_train = torch.utils.data.Subset(train_dataset, retain_indices_train)
retain_loader_train = DataLoader(retain_subset_train, batch_size=64, shuffle=True)

# Perform balanced unlearning
print("\nPerforming balanced unlearning...")
unlearn_epochs = 5
model_balanced.train()
for epoch in range(unlearn_epochs):
    # Use iterators to handle datasets of different lengths
    retain_iter = iter(retain_loader_train)
    forget_iter = iter(forget_loader)
    
    # Loop until the smaller loader is exhausted
    for i in range(len(forget_loader)):
        try:
            # Get retain data and perform a standard update
            retain_images, retain_labels = next(retain_iter)
            optimizer_balanced.zero_grad()
            retain_outputs = model_balanced(retain_images)
            loss_retain = criterion_balanced(retain_outputs, retain_labels)
            loss_retain.backward()
            optimizer_balanced.step()

            # Get forget data and perform a gradient ascent update
            forget_images, forget_labels = next(forget_iter)
            optimizer_balanced.zero_grad()
            forget_outputs = model_balanced(forget_images)
            loss_forget = -criterion_balanced(forget_outputs, forget_labels)
            loss_forget.backward()
            optimizer_balanced.step()

        except StopIteration:
            # This handles the case where one loader is exhausted before the other
            break
            
    # Evaluate at the end of each epoch
    acc_forget = evaluate_model(model_balanced, forget_loader_test)
    acc_retain = evaluate_model(model_balanced, retain_loader_test)
    print(f"Unlearning Epoch {epoch+1}/{unlearn_epochs} | Forget Acc: {acc_forget:.4f} | Retain Acc: {acc_retain:.4f}")

# Final evaluation after balanced unlearning
print("\n--- Evaluation After Balanced Unlearning ---")
accuracy_forget_balanced = evaluate_model(model_balanced, forget_loader_test)
accuracy_retain_balanced = evaluate_model(model_balanced, retain_loader_test)
accuracy_overall_balanced = evaluate_model(model_balanced, test_loader)

print(f"Accuracy on forgotten class ({forget_class}): {accuracy_forget_balanced:.4f}")
print(f"Accuracy on retained classes: {accuracy_retain_balanced:.4f}")
print(f"Overall accuracy after unlearning: {accuracy_overall_balanced:.4f}")
