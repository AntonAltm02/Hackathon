import pandas as pd
from process import get_images
import efficientNet
from tqdm import tqdm
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_effnetv2 import get_efficientnet_v2
from src.mammography.model_effnetv2 import get_efficientnet_v2_1_channel

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

    features = torch.tensor(cv10.squeeze(-1), dtype=torch.float)
    labels = torch.tensor(cv10_labels, dtype=torch.float)
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=5)

    batch_size = 32
    dataset_train = TensorDataset(X_train, Y_train)  # Wrap tensor in a dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = get_efficientnet_v2_1_channel(model_name="efficientnet_v2_s", pretrained=None, nclass=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        for X_batch, Y_batch in tqdm(dataloader_train):
            optimizer.zero_grad()
            preds = model(X_batch.unsqueeze(1))
            loss = criterion(preds, Y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")

    model.eval()
    dataset_test = TensorDataset(X_test, Y_test)  # Wrap tensor in a dataset
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader_test:
            if torch.cuda.is_available():
                X_batch, Y_batch = X_batch.to(), Y_batch.to()

            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Stop here")
