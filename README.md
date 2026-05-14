# MobileViT-XXS: PyTorch Implementation & Edge-AI Use Case

## Overview
This repository contains a from-scratch PyTorch implementation of **MobileViT (XXS variant)**, a lightweight, hybrid architecture that bridges the gap between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). 

Standard ViTs are highly accurate and capture global context but are often computationally prohibitive for edge devices due to their $O(N^{2}d)$ attention complexity. This implementation utilizes MobileViT's core mechanism—**Unfold, Transform, Fold**—to swap out heavy local matrix multiplications for multi-headed self-attention directly within structured spatial tensors. This allows the model to retain CNN spatial inductive biases while gaining ViT global context.

### Proof of Concept: Real-Time Drowsiness Detection (ViT-DDD)
To demonstrate the architecture's capability on edge devices, this repository includes a **Drowsy Driver Detection (DDD)** pipeline. Drowsy driving is a critical safety issue, and legacy CNNs often fail when local features are occluded by sunglasses or harsh lighting. By leveraging the global facial geometry through the MobileViT blocks, the model accurately correlates head tilt, jaw tension, and overall geometry, achieving robust real-time performance suitable for microcomputers like the Raspberry Pi.

## Repository Structure

### Core Architecture
* **`model.py`**: The heart of the repository. Contains the custom PyTorch implementation of `MobileViT_XXS`, including:
  * `MV2Block`: MobileNetV2 inverted residual blocks for efficient local feature extraction.
  * `TransformerEncoder`: Standard multi-head self-attention and feed-forward networks.
  * `MobileViTBlock`: The fusion block that unfolds feature maps into patches, applies global transformer attention, and folds them back into spatial tensors.

### Training & Data
* **`dataset.py`**: Handles loading, transformation, and 80/20 train-test splitting of image data. Includes heavy data augmentation (cropping, rotation, color jitter) to improve generalization.
* **`train.py`**: The training loop. Features a utility function to automatically download and transfer official ImageNet pre-trained weights from `timm` to the custom architecture to speed up convergence.

### Demonstration
* **`webcam.py`**: A live inference script using OpenCV. It utilizes Haar Cascades to isolate the Region of Interest (ROI), feeds the localized sequence to the MobileViT model, and triggers a warning logic if a threshold of drowsy frames is exceeded.

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
