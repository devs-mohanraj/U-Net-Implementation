import os
import requests
from tqdm import tqdm
import zipfile

# Define the URLs for the images and masks
images_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip'
masks_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip'

# Define the output directory
output_dir = './isic_dataset'
images_zip_path = os.path.join(output_dir, 'images.zip')
masks_zip_path = os.path.join(output_dir, 'masks.zip')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

# Download the images and masks
print("Downloading images...")
download_file(images_url, images_zip_path)
print("Downloading masks...")
download_file(masks_url, masks_zip_path)

# Unzip the downloaded files

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

print("Unzipping images...")
unzip_file(images_zip_path, os.path.join(output_dir, 'images'))
print("Unzipping masks...")
unzip_file(masks_zip_path, os.path.join(output_dir, 'masks'))

print("Download and extraction complete.")