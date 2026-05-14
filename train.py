import torch
import torch.nn as nn
import torch.optim as optim
from model import MobileViT_XXS
from dataset import get_dataloaders
import time
import timm

# training settings
BATCH_SIZE = 16  
EPOCHS = 15
LEARNING_RATE = 1e-4
DATA_DIR = "./dataset"

def load_pretrained_weights(custom_model):
    print("\nDownloading official ImageNet weights")
    
    # grab the official pretrained model
    official_model = timm.create_model('mobilevit_xxs', pretrained=True)
    
    official_state_dict = official_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # get both sets of weights
    official_values = list(official_state_dict.values())
    custom_keys = list(custom_state_dict.keys())

    print("Transferring knowledge")
    transferred_layers = 0
    skipped_layers = 0

    # match layers by order and shape
    for i, custom_key in enumerate(custom_keys):
        if i < len(official_values):
            official_tensor = official_values[i]
            custom_tensor = custom_state_dict[custom_key]

            # only copy if dimensions line up
            if official_tensor.shape == custom_tensor.shape:
                custom_state_dict[custom_key] = official_tensor
                transferred_layers += 1
            else:
                skipped_layers += 1

    custom_model.load_state_dict(custom_state_dict)
    return custom_model

def train_model():
    # pick gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # load data and build model
    train_loader, test_loader, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Classes: {classes}")
    
    model = MobileViT_XXS(num_classes=len(classes))
    
    # load pretrained weights first
    model = load_pretrained_weights(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_accuracy = 0.0

    # main training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 15)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # reset gradients
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backprop step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # quick progress update
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = 100 * correct_train / total_train
        
        # validation time
        model.eval()
        correct_test = 0
        total_test = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        val_acc = 100 * correct_test / total_test
        epoch_time = time.time() - start_time

        # epoch summary
        print(f"Epoch {epoch+1} Summary ({epoch_time:.2f}s):")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # save best checkpoint
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "best_mobilevit_drowsiness.pth")
            print("--> Best model saved!")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()