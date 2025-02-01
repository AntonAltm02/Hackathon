import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model_effnetv2 import get_efficientnet_v2_1_channel

# Image pre-processing transformations
image_size = (256, 256)
crop_size = (224, 224)  # ImageNet standard

transform_set = [
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
]

transformations_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.RandomEqualize(),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.2),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomApply(transform_set, p=0.3),
    transforms.CenterCrop(size=crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transformations = transforms.Compose([
    transforms.CenterCrop(size=crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Custom Tensor Dataset class
class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get image and label
        image = self.images[idx]
        label = self.labels[idx]
        # Convert 1-channel tensor to PIL Image so that transformations can be applied
        image = transforms.functional.to_pil_image(image.unsqueeze(0))
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "data")

    cv10 = np.load(os.path.join(data_path, "DDSM/cv10_data/cv10_data.npy"))
    test10 = np.load(os.path.join(data_path, "DDSM/test10_data/test10_data.npy"))
    cv10_labels = np.load(os.path.join(data_path, "DDSM/cv10_labels.npy"))
    test10_labels = np.load(os.path.join(data_path, "DDSM/test10_labels.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = torch.tensor(cv10.squeeze(-1), dtype=torch.float).to(device)
    labels = torch.tensor(cv10_labels, dtype=torch.float).to(device)

    # Split the dataset for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=5)

    batch_size = 8

    # Construct datasets and dataloaders
    dataset_train = CustomTensorDataset(X_train, Y_train, transform=transformations_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = get_efficientnet_v2_1_channel(model_name="efficientnet_v2_s", pretrained=None, nclass=1)
    model = model.to(device)
    print('Model created')

    # Using CrossEntropyLoss; note that target is expected as long type for class indices.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0

        # Training loop: accumulate loss * batch_size for correct averaging later
        for X_batch, Y_batch in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch.long())
            loss.backward()
            optimizer.step()

            # Multiply the average batch loss by the number of samples in the batch
            running_loss_train += loss.item() * X_batch.size(0)

        # Calculate average training loss over all samples in the training set
        avg_train_loss = running_loss_train / len(dataset_train)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        running_loss_val = 0.0
        all_preds = []
        all_labels = []

        # Create validation dataset and dataloader using X_test and Y_test
        dataset_test = CustomTensorDataset(X_test, Y_test, transform=transformations)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for X_batch, Y_batch in dataloader_test:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                outputs = model(X_batch)
                loss_val = criterion(outputs, Y_batch.long())

                # Accumulate weighted loss for validation
                running_loss_val += loss_val.item() * X_batch.size(0)

                # Obtain predictions and store ground truth labels from the current batch
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(Y_batch.cpu().numpy())

        # Compute average validation loss over all validation samples
        avg_val_loss = running_loss_val / len(dataset_test)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')

    print("Training complete!")
