import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=64, upscale_factor=2, transform=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        
        # List all LR files
        self.lr_filenames = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.lr_filenames)
    
    def __getitem__(self, idx):
        lr_name = self.lr_filenames[idx]
        # Match HR filename: 0001x2.png -> 0001.png
        # Some versions might have different naming, let's be flexible
        hr_name = lr_name.replace('x2', '') 
        
        lr_path = os.path.join(self.lr_dir, lr_name)
        hr_path = os.path.join(self.hr_dir, hr_name)
        
        # Check if HR exists, if not, try to find it by ID
        if not os.path.exists(hr_path):
            # Try to match the first 4 characters
            img_id = lr_name[:4]
            hr_files = os.listdir(self.hr_dir)
            for f in hr_files:
                if f.startswith(img_id) and f.endswith('.png'):
                    hr_path = os.path.join(self.hr_dir, f)
                    break

        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        
        if lr_img is None or hr_img is None:
            # Skip corrupted or missing
            return self.__getitem__((idx + 1) % len(self))

        # Convert to RGB
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Random Crop
            h, w, _ = lr_img.shape
            th, tw = self.patch_size, self.patch_size
            
            if h > th and w > tw:
                x = random.randint(0, w - tw)
                y = random.randint(0, h - th)
                
                lr_patch = lr_img[y:y+th, x:x+tw, :]
                hr_patch = hr_img[y*self.upscale_factor:(y+th)*self.upscale_factor, 
                                  x*self.upscale_factor:(x+tw)*self.upscale_factor, :]
            else:
                # Fallback if image is too small (shouldn't happen with DIV2K)
                lr_patch = cv2.resize(lr_img, (th, tw))
                hr_patch = cv2.resize(hr_img, (th*self.upscale_factor, tw*self.upscale_factor))
                
            # Random Horizontal Flip
            if random.random() > 0.5:
                lr_patch = cv2.flip(lr_patch, 1)
                hr_patch = cv2.flip(hr_patch, 1)
                
            # Random Vertical Flip
            if random.random() > 0.5:
                lr_patch = cv2.flip(lr_patch, 0)
                hr_patch = cv2.flip(hr_patch, 0)
        else:
            # For validation, we might want to just center crop or use the whole image
            # But full images vary in size, so let's stick to a fixed large crop for batching
            h, w, _ = lr_img.shape
            th, tw = self.patch_size, self.patch_size
            x = (w - tw) // 2
            y = (h - th) // 2
            lr_patch = lr_img[y:y+th, x:x+tw, :]
            hr_patch = hr_img[y*self.upscale_factor:(y+th)*self.upscale_factor, 
                              x*self.upscale_factor:(x+tw)*self.upscale_factor, :]

        # Normalize to [0, 1] and convert to Tensor (C, H, W)
        lr_patch = torch.from_numpy(lr_patch.transpose(2, 0, 1)).float() / 255.0
        hr_patch = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float() / 255.0
        
        return lr_patch, hr_patch

def get_dataloader(lr_dir, hr_dir, patch_size=64, batch_size=8, is_train=True):
    dataset = DIV2KDataset(lr_dir, hr_dir, patch_size=patch_size, transform=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
