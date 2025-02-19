import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Step 1: Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Define Transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Load Validation Data
val_data = ImageFolder(root="C:/Users/Dell/ml_project/data/train", transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Step 4: Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(val_data.classes))  # Adjust output layer
model.load_state_dict(torch.load("C:/Users/Dell/ml_project/model.pth"))  # Load saved model
model.to(device)
model.eval()

# Step 5: Evaluate Accuracy
def evaluate():
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in tqdm(val_loader, total=len(val_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f" Model Accuracy: {accuracy:.2f}%")

# Step 6: Run Evaluation
if __name__ == "__main__":
    print(" Evaluating model...")
    evaluate()
