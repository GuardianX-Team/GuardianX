# sign_training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ✅ Check GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ✅ Data transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load dataset
dataset_dir = "sign_dataset"  # <-- Put your dataset here
train_data = datasets.ImageFolder(dataset_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# ✅ Get number of classes
num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ✅ Model (simple resnet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ✅ Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training loop
epochs = 5  # increase to 15–20 for better accuracy
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# ✅ Save model
torch.save(model.state_dict(), "sign_model.pth")
print("✅ Model saved as sign_model.pth")
# Save class labels
with open("class_labels.txt", "w") as f:
    for cls in train_data.classes:
        f.write(cls + "\n")
print("✅ Class labels saved to class_labels.txt")
