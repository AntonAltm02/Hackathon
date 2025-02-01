import copy
from functools import partial
from collections import OrderedDict
import torch
from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper import train_model, SimpleCNN, validate_model, CLAHE
import torch.nn as nn
import torch.nn.functional as F
from submit import TestDataset
import torch.optim as optim
from model_effnetv2 import get_efficientnet_v2, get_efficientnet_v2_1_channel
BATCH_SIZE = 64
image_size = (256, 256)
crop_size = (224, 224) #imagenet standards
transform_set = [ 
             transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            #  transforms.RandomVerticalFlip(p=0.1),
    ]
transformations_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(degrees=(0,30)),
        transforms.ColorJitter(),
        transforms.RandomPosterize(bits=2),
        transforms.RandomEqualize(),
        # transforms.RandomCrop(size=image_size),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.2),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomApply(transform_set, p=0.3),
        transforms.CenterCrop(size=crop_size),
        CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(size=crop_size),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
train_dataset = datasets.ImageFolder(root='data/train', transform=transformations_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers = 2)

img_channel = 1
num_classes = 1000
model = get_efficientnet_v2(model_name = "efficientnet_v2_s", pretrained= None, nclass= num_classes)

continue_step = 0
if continue_step > 0:
    model.load_state_dict(torch.load(f'effecientnetv2/effnetv2B_1channel_epoch{continue_step}.ckpt'))
val_dataset = datasets.ImageFolder(root='data/val', transform=transformations)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.to(device)
    print('Cuda available')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

num_epochs = 200-continue_step
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, labels in train_loader:
        i += 1
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
    
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+continue_step}/{num_epochs+continue_step}], Step [{i}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), f'effecientnetv2/effnetv2B_1channel_epoch{epoch+continue_step}.ckpt')
    model.eval()
    validate_model(model, val_loader, criterion)


torch.save(model.state_dict(), 'effnetv2B_1channel.ckpt')
