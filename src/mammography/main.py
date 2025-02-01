import pandas as pd
# from process import get_images
# import efficientNet
from tqdm import tqdm
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_effnetv2 import get_efficientnet_v2
from model_effnetv2 import get_efficientnet_v2_1_channel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

image_size = (256, 256)
crop_size = (224, 224)  # imagenet standards
transform_set = [
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    #  transforms.RandomVerticalFlip(p=0.1),
]
transformations_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(degrees=(0, 30)),
    # transforms.ColorJitter(),
    # transforms.RandomPosterize(bits=2),
    transforms.RandomEqualize(),
    # transforms.RandomCrop(size=image_size),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.2),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomApply(transform_set, p=0.3),
    transforms.CenterCrop(size=crop_size),
    # CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

transformations = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.CenterCrop(size=crop_size),
    # CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert tensor to PIL Image for transformations
        image = transforms.functional.to_pil_image(image.unsqueeze(0))

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir + "/../../data")

    """
    dicom_info = pd.read_csv(data_path + "/CBIS-DDSM/dicom_info.csv")
    image_dir = data_path + "/CBIS-DDSM/jpeg"

    print(dicom_info.head())
    print(dicom_info.info())

    full_mammogram_images = get_images(img_type="full mammogram images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)
    ROI_images = get_images(img_type="ROI mask images", image_dir=image_dir,
                                       show_img=False, dicom_info=dicom_info)
    """

    cv10 = np.load(data_path + "/DDSM/cv10_data/cv10_data.npy")
    test10 = np.load(data_path + "/DDSM/test10_data/test10_data.npy")
    cv10_labels = np.load(data_path + "/DDSM/cv10_labels.npy")
    test10_labels = np.load(data_path + "/DDSM/test10_labels.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = torch.tensor(cv10.squeeze(-1), dtype=torch.float).to(device)
    labels = torch.tensor(cv10_labels, dtype=torch.float).to(device)
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=5)

    batch_size = 8
    dataset_train = CustomTensorDataset(X_train, Y_train, transform=transformations_train)
    # dataset_train = TensorDataset(X_train, Y_train)  # Wrap tensor in a dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = get_efficientnet_v2_1_channel(model_name="efficientnet_v2_s", pretrained=None, nclass=1)
    model = model.to(device)
    print('model created')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        for X_batch, Y_batch in tqdm(dataloader_train):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch.long())
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}')

        # print("Training complete!")

        model.eval()
        dataset_test = CustomTensorDataset(X_test, Y_test, transform=transformations)  # Wrap tensor in a dataset
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, Y_batch in dataloader_test:
                # if torch.cuda.is_available():
                #     X_batch, Y_batch = X_batch.to(), Y_batch.to()

                # Forward pass
                outputs = model(X_batch.unsqueeze(1))
                loss2 = criterion(outputs, Y_batch.long())
                total_loss += loss2.item()

                # Convert outputs to predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {total_loss:.4f}')

    print("Stop here")
