# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Step 2: Data Preprocessing

# Load the images and labels into tensors
image_list = []
label_list = []

for position in range(9):  # Assuming 9 positions
    for index in range(1000):  # Assuming 1000 images per position
        img_path = f"dataset/right_eye_{index}_{position}.png"
        if os.path.exists(img_path):
            transform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),  # Resize all images to 64x64
                    transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
                ]
            )
            img = Image.open(img_path).convert("L")  # Converting to grayscale
            img = transform(img)  # Apply the transformations
            image_list.append(img)
            label_list.append(position)

# Convert lists to tensors
images = torch.stack(image_list)
labels = torch.tensor(label_list, dtype=torch.long)

# Create a dataset and dataloaders
dataset = TensorDataset(images, labels)

# Train-Test Split: 80% for training and 20% for testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3 and Step 4: Feature Extraction and Model Selection


# Define a Convolutional Neural Network (CNN) with Batch Normalization and Leaky ReLU
class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 9)  # 9 classes

    def forward(self, x):
        x = nn.LeakyReLU(0.01)(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.LeakyReLU(0.01)(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 64 * 14 * 14)
        x = nn.LeakyReLU(0.01)(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model():
    # Initialize the model, loss function, and optimizer
    model = EyeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(10):  # 10 epochs
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

    # Save the model for future use
    torch.save(model.state_dict(), "eye_position_model.pth")


if __name__ == "__main__":
    train_model()
