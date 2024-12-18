import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from model import UNet
from PIL import Image
import numpy as np


class LiverDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # Define mapping for RGB values to class indices
        self.rgb_to_class = {
            (0, 0, 0): 0,        # Background -> Class 0
            (0, 0, 255): 1,      # Portal vein (blue) -> Class 1
            (0, 255, 0): 2,      # Central vein (green) -> Class 2
            (255, 0, 0): 3,      # Artery (red) -> Class 3
            (128, 0, 128): 4     # Bile duct (purple) -> Class 4
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        # Convert mask to single-channel tensor (class indices)
        mask = np.array(mask)
        mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for rgb, class_idx in self.rgb_to_class.items():
            mask_class[(mask == rgb).all(axis=-1)] = class_idx

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask_class, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask


def load_data(data_dir, batch_size=4, val_split=0.2):
    image_paths = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images"))])
    mask_paths = sorted([os.path.join(data_dir, "masks", f) for f in os.listdir(os.path.join(data_dir, "masks"))])

    dataset = LiverDataset(image_paths, mask_paths)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def dice_coefficient_per_class(pred, target, n_classes, smooth=1):
    
    pred = torch.softmax(pred, dim=1) 
    pred = torch.argmax(pred, dim=1) 

    dice_scores = []
    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Per-sample intersection
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Per-sample union

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean().item())  # Average over batch

    return dice_scores


def evaluate(model, dataloader, device, n_classes):
    
    model.eval()  # Set model to evaluation mode
    class_dice_scores = {c: [] for c in range(n_classes)}

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            dice_scores = dice_coefficient_per_class(outputs, masks, n_classes)

            for c, dice in enumerate(dice_scores):
                class_dice_scores[c].append(dice)

    avg_dice_scores = {c: sum(scores) / len(scores) for c, scores in class_dice_scores.items()}
    return avg_dice_scores


def train(model, train_loader, val_loader, device, epochs=20, lr=1e-4, n_classes=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    print(f"Training on {device}...")

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        class_losses = {c: 0.0 for c in range(n_classes)}

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate Dice and loss per class
            for c in range(n_classes):
                mask_c = (masks == c).float()
                loss_c = nn.functional.cross_entropy(outputs[:, c, :, :], mask_c, reduction="mean")
                class_losses[c] += loss_c.item()

        # Average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_class_losses = {f"loss_class_{c}": class_losses[c] / len(train_loader) for c in range(n_classes)}
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")
        print(f"Class Losses: {avg_class_losses}")

        # Validation Metrics
        val_dice_scores = evaluate(model, val_loader, device, n_classes)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Dice Coefficient per Class: {val_dice_scores}")

    print("Training Finished.")


def main():
    # Paths
    data_dir = "C:/Users/mehul/P2-P10-Project/data"  
    model_save_path = "./unet_liver_segmentation.pth"

    # Hyperparameters
    batch_size = 2
    epochs = 50
    learning_rate = 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    train_loader, val_loader = load_data(data_dir, batch_size=batch_size)

    # Model
    model = UNet(n_channels=3, n_classes=5)

    # Train
    train(model, train_loader, val_loader, device, epochs=epochs, lr=learning_rate, n_classes=5)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
