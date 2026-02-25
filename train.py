import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SRModel
from dataset_loader import get_dataloader
import time
import os
import math

def calculate_psnr(img1, img2):
    # img1 and img2 are tensors in range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def train():
    print("Initializing training script...", flush=True)
    # Setup Device
    print("Checking for GPU...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Hyperparameters
    batch_size = 4 # Reduced to free up VRAM for inference
    patch_size = 64 # Size in LR
    lr = 1e-4
    epochs = 15
    model_save_path = "model.pth"

    # Dataset Paths
    print("Defining dataset paths...", flush=True)
    train_lr_dir = r"dataset/DIV2K_train_LR_unknown/X2"
    train_hr_dir = r"dataset/DIV2K_train_HR/DIV2K_train_HR"
    valid_lr_dir = r"dataset/DIV2K_valid_LR_unknown/X2"
    valid_hr_dir = r"dataset/DIV2K_valid_HR"

    # Check if paths exist
    for d in [train_lr_dir, train_hr_dir, valid_lr_dir, valid_hr_dir]:
        if not os.path.exists(d):
            print(f"Warning: Directory {d} does not exist: {d}", flush=True)
        else:
            print(f"Found directory: {d}", flush=True)

    # Data Loaders
    train_loader = get_dataloader(train_lr_dir, train_hr_dir, patch_size=patch_size, batch_size=batch_size, is_train=True)
    valid_loader = get_dataloader(valid_lr_dir, valid_hr_dir, patch_size=patch_size, batch_size=batch_size, is_train=False)

    # Model, Loss, Optimizer
    model = SRModel(upscale_factor=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting Training...", flush=True)
    best_psnr = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            # Keep VRAM clean for other processes (inference)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_loss += loss.item()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)
            
            # Save intermediate model for quick testing during hackathon
            if (i+1) % 5 == 0:
                torch.save(model.state_dict(), "model_new.pth")
                print(f"Intermediate model saved to model_new.pth", flush=True)

        # Validation
        model.eval()
        avg_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                outputs = model(lr_imgs)
                # Clamp outputs for PSNR calculation
                outputs = torch.clamp(outputs, 0, 1)
                
                # Accumulate PSNR for the whole batch
                batch_psnr = 0.0
                for b in range(lr_imgs.size(0)):
                    batch_psnr += calculate_psnr(outputs[b], hr_imgs[b])
                avg_psnr += batch_psnr / lr_imgs.size(0)

        avg_psnr /= len(valid_loader)
        duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] Done. Avg Loss: {epoch_loss/len(train_loader):.4f}, Val PSNR: {avg_psnr:.2f} dB, Time: {duration:.2f}s", flush=True)

        # Save Best Model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with PSNR: {best_psnr:.2f}", flush=True)

    print("Training Finished.", flush=True)

if __name__ == "__main__":
    train()
