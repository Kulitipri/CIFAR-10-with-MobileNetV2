# 🚀 Advanced CIFAR-10 Image Classification with MobileNetV2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Accuracy-~88%25-brightgreen)

## 📖 Overview

This project implements a high-performance Deep Learning model to classify images from the **CIFAR-10 dataset** (10 classes: Airplane, Car, Bird, Cat, etc.). 

Instead of building a simple CNN from scratch, this project leverages **Transfer Learning** using **MobileNetV2** (pre-trained on ImageNet). It employs advanced techniques like **Image Upscaling** and **Fine-Tuning** to achieve a Test Accuracy of **>88%**, significantly outperforming standard baseline models.

## 🧠 Key Technical Features

* **Transfer Learning:** Utilizes `MobileNetV2` as a powerful feature extractor.
* **Resolution Upscaling:** Although CIFAR-10 images are 32x32, this model internally resizes them to **96x96** to allow MobileNetV2 to detect more intricate features.
* **Two-Stage Training Strategy:**
    1.  **Warm-up:** Training the classifier head while the base model is frozen.
    2.  **Fine-tuning:** Unfreezing the top 40 layers of MobileNetV2 with a very low learning rate (`1e-5`) to adapt to the specific dataset.
* **Regularization:** Heavy use of `Data Augmentation` (Rotation, Zoom, Flip) and `Dropout` to prevent overfitting.
* **Callbacks:** Implements `ModelCheckpoint` and `EarlyStopping` for optimal training efficiency.

## 📊 Model Performance

| Metric | Result |
| :--- | :--- |
| **Test Accuracy** | **88.12%** |
| **Model Architecture** | MobileNetV2 (Pre-trained) + Custom Dense Head |
| **Input Size** | 32x32 (Upscaled to 96x96) |

*(Detailed confusion matrix and classification report are generated after training)*

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Kulitipri/CIFAR-10-with-MobileNetV2]
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Train the Model
To train the model from scratch (including both warm-up and fine-tuning phases):

```bash
python cifar-10_with_DL.py
