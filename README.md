# Vision Transformer for MNIST Classification

This project implements a Vision Transformer (ViT) from scratch to classify the MNIST dataset, which contains handwritten digits. The model was trained to predict digits from 0 to 9, and its performance was optimized with data augmentation techniques to address classification issues.

## Problem Overview

During training, the model initially struggled with classifying the digit "8" as it was misclassified as "3". 
![Screenshot from 2025-01-10 14-49-50](https://github.com/user-attachments/assets/9498e664-5d5c-49d4-a716-bd1a78e0cf6b)

This issue was resolved by applying dropout and data augmentation techniques to enhance the model's generalization capability.
![image](https://github.com/user-attachments/assets/39ac3098-0166-4890-8c8d-7c3ab8b2bac7)


## Key Features

- **Model Architecture**: Vision Transformer (ViT) built from scratch.
- **Dataset**: MNIST dataset, consisting of 28x28 grayscale images of handwritten digits (0-9).
- **Data Augmentation**: 
  - RandomRotation
  - RandomAffine
  - RandomResizedCrop

These augmentations were applied to improve the model's robustness to transformations.

## Performance

- **Precision**: 1.0
- **Recall**: 1.0
- **F1 Score**: 1.0

The model achieved perfect scores on all evaluation metrics after implementing data augmentation.

## Installation

To get started, clone the repository:

```bash
git clone https://github.com/your-username/vision-transformer-mnist.git](https://github.com/JagadeeshKrishnaSNunna/ViT.git
cd ViT
pip install -r requirements.txt
# make changes in the config in Train.py for archetecture
# change dataset in data.py for your dataset
python3 Train.py
