import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.float()
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)



def load_model_file(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Set to evaluation mode
    return model


def save_model(model, path="outputs/model_weights/unet.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at {path}")

def load_model(model, path="outputs/model_weights/unet.pth", device="cuda"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image.unsqueeze(0)

def postprocess_mask(mask):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask
