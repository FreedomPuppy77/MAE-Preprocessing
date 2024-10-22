import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom dataset for loading images and VA labels
class VADataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        # Collect image paths and corresponding label paths
        for folder in os.listdir(self.img_dir):
            img_folder = os.path.join(self.img_dir, folder)
            label_file = os.path.join(self.label_dir, folder + '.txt')

            if os.path.isdir(img_folder) and os.path.exists(label_file):
                img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')])
                labels = np.loadtxt(label_file, delimiter=',')
                self.samples.extend([(img_files[i], labels[i]) for i in range(len(img_files))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Training function
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_path, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation phase
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}] Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"Best model saved with loss {best_loss:.4f}")

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    # Paths and parameters
    train_img_dir = './train'
    val_img_dir = './val'
    label_dir = './labels'
    weight_path = './best_model.pth'
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    img_size = 224

    # Define device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets and data loaders
    train_dataset = VADataset(train_img_dir, label_dir, transform=transform)
    val_dataset = VADataset(val_img_dir, label_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    from model_VA import MAEEncoderForVA  # Assuming you saved the model as model_VA.py
    model = MAEEncoderForVA()

    # Load pretrained weights
    pretrained_weights = './mae_pretrained.pth'  # Path to your pretrained MAE weights
    model.load_pretrained_weights(pretrained_weights)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_path, device)
