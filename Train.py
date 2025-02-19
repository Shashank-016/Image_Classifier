import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Step 1: Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Load dataset (Assuming all images are inside 'data/train')
dataset = datasets.ImageFolder(root="C:/Users/Dell/ml_project/data/train", transform=transform)

# Step 4: Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Step 5: Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Step 6: Initialize Model (ResNet18)
model = models.resnet18(weights=None)  # No pretrained weights
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # Adjust output layer for class count
model = model.to(device)

# Step 7: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Training Loop
def train():
    model.train()  # Set model to training mode
    for epoch in range(5):  # Number of epochs
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/5], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0
        print(f" Finished Epoch {epoch + 1}")

    # Step 9: Save Model
    torch.save(model.state_dict(), "C:/Users/Dell/ml_project/model.pth")
    print(" Model saved as 'model.pth'")

# Step 10: Run Training
if __name__ == "__main__":
    print(" Training started...")
    train()
