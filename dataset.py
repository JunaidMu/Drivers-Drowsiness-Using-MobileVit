import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
from torch.utils.data import DataLoader, Subset

def get_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((260, 260)), 
        transforms.RandomCrop((256, 256)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
        transforms.RandomGrayscale(p=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

    # over here i am doing the splitting
    num_samples = len(full_train_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, full_train_dataset.classes

if __name__ == "__main__":
    DATA_DIR = "./dataset" 
    
    try:
        train_dl, test_dl, classes = get_dataloaders(DATA_DIR, batch_size=8)
        print(f"Success! Found classes: {classes}")
        
        images, labels = next(iter(train_dl))
        print(f"Batch Image Shape: {images.shape}")
        print(f"Batch Label Shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")