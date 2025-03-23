# U-Net Implementation for Image Segmentation

## 1. Introduction
U-Net is a convolutional neural network (CNN) architecture designed for image segmentation tasks. It is widely used in applications such as medical image analysis and satellite image processing. This project implements U-Net for **Medical Image Segmentation** using **PyTorch**.

## 2. Project Overview
This project aims to develop a deep learning model based on the U-Net architecture for **Medical Image Segmentation**. The model is trained on **ISIC** and evaluated using Dice Coefficient metrics.

## 3. Model Architecture
**U-Net** consists of an encoder-decoder structure with skip connections:
- **Encoder**: Feature extraction using convolutional layers
- **Bottleneck**: Bridge between encoder and decoder
- **Decoder**: Upsampling and reconstruction of segmented images
- **Skip Connections**: Help retain spatial information



![U-Net Architecture](https://www.researchgate.net/publication/361357383/figure/fig2/AS:1168145503006721@1655518999463/Architecture-of-U-Net-with-dense-block.png)


## 4. Dataset
- **Dataset Name**: [**Dataset**](https://challenge.isic-archive.com/data/#2018)
- **Number of Images And Masks**: 16072
- **Classes**: 2
- **Preprocessing**: 
  - Resizing images to [256X256]
  - Normalization

## 5. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/devs-mohanraj/U-Net-.git
cd U-Net-
pip install -r requirements.txt
```

## 6. Training & Evaluation
To train the model:
```bash
python train.py 
```
To evaluate the model:

<span style="color:lightblue;">**Use inference.ipynb for visualization**</span>

## 7. Results
Performance metrics:

- **Dice Coefficient**: **88.051** %

Example segmentation results:


## 8. Validation and loss
![**The Validation Dice Score**](/teamspace/studios/this_studio/output/outputs/W&B Chart 3_23_2025, 1_31_42 AM.png)
![Validation Dice Score](/teamspace/studios/this_studio/output/outputs/W&B Chart 3_23_2025, 1_31_42 AM.png)
### the Validation Dice score
![Validation Dice Score](data/images_for_readme/W&B Chart 3_23_2025, 1_32_44 AM.png)
## 9. Directory Structure
```
U-Net-Project/
├── data/                  # Dataset folder
├── models/                # Trained models
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter Notebooks (if applicable)
├── results/               # Output segmented images
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── train.py               # Training script
├── predict.py             # Prediction script
└── utils.py               # Helper functions
```

## 10. References
- **U-Net Paper**: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
