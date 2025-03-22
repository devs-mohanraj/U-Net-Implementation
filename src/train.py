import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb

from dataset import MedicalDataset
from unet import UNet
from utils import dice_score, save_model

# Initialize Weights & Biases
wandb.init(project="unet-isic-segmentation", config={
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 25,
    "architecture": "U-Net",
    "dataset": "ISIC 2018",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}, mode="online")  # Use "offline" if logging locally

# Configuration
BATCH_SIZE = wandb.config.batch_size
LEARNING_RATE = wandb.config.learning_rate
EPOCHS = wandb.config.epochs
DEVICE = torch.device(wandb.config.device)

# Paths
DATASET_PATH = "/teamspace/studios/this_studio/data/isic_dataset"
MODEL_SAVE_PATH = "/teamspace/studios/this_studio/output/model_path"

# Datasets and DataLoaders
train_dataset = MedicalDataset(
    image_dir=f"{DATASET_PATH}/images/ISIC2018_Task1-2_Training_Input",
    mask_dir=f"{DATASET_PATH}/masks/ISIC2018_Task1_Training_GroundTruth"
)
val_dataset = MedicalDataset(
    image_dir=f"{DATASET_PATH}/images/ISIC2018_Task1-2_Training_Input",
    mask_dir=f"{DATASET_PATH}/masks/ISIC2018_Task1_Training_GroundTruth"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

wandb.watch(model, log="all", log_freq=10)  # Log gradients and parameters

def train():
    best_val_score = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log per batch
            wandb.log({
                "Batch Train Loss": loss.item(),
                "Batch": batch_idx + 1,
                "Epoch": epoch + 1
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                val_dice += dice_score(masks, preds)
        
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Dice Score: {avg_val_dice:.4f}")
        
        # Log Metrics to wandb
        wandb.log({
            "Train Loss": avg_train_loss,
            "Validation Dice Score": avg_val_dice,
            "Epoch": epoch + 1
        })
        
        # Save best model
        if avg_val_dice > best_val_score:
            best_val_score = avg_val_dice
            save_model(model, "/teamspace/studios/this_studio/output/model_path/unet.pth")
    
    wandb.finish()

if __name__ == "__main__":
    train()
