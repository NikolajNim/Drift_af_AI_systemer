import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import ModifiedResNet50
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def get_output_name():
    print('Choose trained model')
    print('1 for standard trained | 2 for augmented data traind')
    while True:
        choice = input('Write choice here: ').strip()

        if choice == '1':
            return 'standard'
        elif choice == '2':
            return 'data_aug'
        else:
            print('Wrong input, try again')

def test_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the test dataset
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Class labels
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Load the trained model
    model = ModifiedResNet50().to(device)
    output_name = get_output_name()
    model.load_state_dict(torch.load(f'Results_dif/best_fashion_mnist_resnet_{output_name}1.pth'))
    model.eval()
    
    # Test the model
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')
    
    # Print detailed classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{output_name}.png')
    plt.close()

if __name__ == '__main__':
    test_model()
