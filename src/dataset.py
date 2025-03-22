import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Filter out non-image files
        valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if os.path.splitext(f)[1].lower() in valid_exts])

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            return None  # Return None for invalid samples

        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to binary tensor
        mask = torch.tensor(np.array(mask) > 0, dtype=torch.float32)

        return image, mask


