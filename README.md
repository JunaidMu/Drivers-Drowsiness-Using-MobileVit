# MobileViT-XXS: PyTorch Implementation & Edge-AI Use Case

## Overview
This repository contains a **from-scratch PyTorch implementation of MobileViT (XXS variant)**, a lightweight hybrid architecture that bridges the gap between **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**.

Standard ViTs are highly accurate and capture global context but are often computationally prohibitive for edge devices due to their **O(N²d)** attention complexity. This implementation utilizes MobileViT's core mechanism—**Unfold → Transform → Fold**—to swap out heavy local matrix multiplications for multi-headed self-attention directly within structured spatial tensors.

This allows the model to retain CNN spatial inductive biases while gaining ViT global context.

---

## Proof of Concept: Real-Time Drowsiness Detection (ViT-DDD)

To demonstrate the architecture's capability on edge devices, this repository includes a **Drowsy Driver Detection (DDD)** pipeline.

Drowsy driving is a critical safety issue, and legacy CNNs often fail when local features are occluded by sunglasses or harsh lighting. By leveraging global facial geometry through MobileViT blocks, the model accurately correlates **head tilt, jaw tension, and overall facial structure**, achieving robust real-time performance suitable for microcomputers like the **Raspberry Pi**.

---

## Repository Structure

### Core Architecture

#### `model.py`
The heart of the repository. Contains the custom PyTorch implementation of **MobileViT_XXS**, including:

- **MV2Block** — MobileNetV2 inverted residual blocks for efficient local feature extraction
- **TransformerEncoder** — Standard multi-head self-attention and feed-forward networks
- **MobileViTBlock** — The fusion block that unfolds feature maps into patches, applies global transformer attention, and folds them back into spatial tensors

### Training & Data

#### `dataset.py`
Handles loading, transformation, and **80/20 train-test splitting** of image data.

Includes heavy data augmentation such as:

- Random cropping
- Rotation
- Color jittering

These augmentations improve generalization and robustness.

#### `train.py`
The main training loop.

Features a utility function to automatically download and transfer official **ImageNet pre-trained weights** from `timm` to the custom architecture to speed up convergence.

### Demonstration

#### `webcam.py`
A live inference script using **OpenCV**.

It utilizes **Haar Cascades** to isolate the facial **Region of Interest (ROI)**, feeds the localized input to the MobileViT model, and triggers a warning when a threshold of drowsy frames is exceeded.

---

# Installation & Setup

## 1. Create a Virtual Environment (Recommended)

Before installing dependencies, it is highly recommended to create a virtual environment to keep your workspace isolated and clean.

```bash
# Create the virtual environment
python -m venv venv
```

### Activate the environment

**Windows**

```bash
.\venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

## 2. Install Dependencies

Once your virtual environment is active, install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

This project uses the **Driver Drowsiness Dataset (DDD)**.

### Download the dataset

Download the dataset (~3GB) from Kaggle:

**Driver Drowsiness Dataset (DDD)**

---

### Organize the dataset

Extract the files and ensure the root directory is named exactly:

```plaintext
dataset
```

Inside the `dataset` folder, organize the images into **exactly two subdirectories**:

- `Drowsy`
- `NonDrowsy`

Your folder structure must look like this:

```plaintext
./dataset
├── Drowsy/
│   ├── img1.jpg
│   └── ...
└── NonDrowsy/
    ├── img1.jpg
    └── ...
```

---

# Usage

## 1. Training the Model

To train the model from scratch on the DDD dataset:

### Step 1
Ensure your dataset folder is structured as described above.

### Step 2
Open `train.py` and adjust the `EPOCHS` variable if needed.

Recommended:

- **10–15 epochs** for good convergence
- Shorter runs on partial data may produce unreliable results

### Step 3
Run the training script:

```bash
python train.py
```

The best model weights will automatically be saved as:

```plaintext
best_mobilevit_drowsiness.pth
```

---

## 2. Running the Demo (Live Inference)

After training, run the webcam script to test the model in real time:

```bash
python webcam.py
```

Press **`q`** to terminate the video stream.
