import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import pandas as pd
import random



class BirdDroneNet(nn.Module):
    def __init__(self, input_channels=3, input_size=224):
        super(BirdDroneNet, self).__init__()

        self.act_function = F.relu
        self.dropout = nn.Dropout(p=0.3)
        layers = [input_channels, 6, 16, 120, 84]

        self.conv1 = nn.Conv2d(layers[0], layers[1], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(layers[1])

        self.conv2 = nn.Conv2d(layers[1], layers[2], kernel_size=5)
        self.bn2 = nn.BatchNorm2d(layers[2])

        # Calculate output size after conv/pool: ((input_size - 0) // 2 - 4) // 2
        conv1_out = input_size // 2  # after first pool
        conv2_out = (conv1_out - 4) // 2  # after conv2 (kernel=5, no padding) and pool
        self.flat_features = conv2_out * conv2_out * layers[2]

        self.fc1 = nn.Linear(self.flat_features, layers[3])
        self.bn_fc1 = nn.BatchNorm1d(layers[3])

        self.fc2 = nn.Linear(layers[3], layers[4])
        self.bn_fc2 = nn.BatchNorm1d(layers[4])

        self.fc3 = nn.Linear(layers[4], 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_function(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_function(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act_function(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.act_function(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)  # Probability output for binary classification

        return x.squeeze(1)  # Ensure shape [batch_size]
    
class MLP(nn.Module):
    def __init__(self, layers, sigma_w, scaling=False):
        super(MLP, self).__init__()
        self.act_function = torch.relu
        self.scaling = scaling
        self.sigma_w = sigma_w
        self.layers = nn.ModuleList()

        for l_in, l_out in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(l_in, l_out))

        self.nb_layers = len(self.layers)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                l.weight.data.normal_()
                l.weight.data.mul_(self.sigma_w)

                if self.scaling:
                    l.weight.data.div_(torch.sqrt(torch.tensor(l.weight.size(1), dtype=torch.float32)))

                l.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        for l in self.layers[:-1]:
            x = l(x)
            x = self.act_function(x)

        x = self.layers[-1](x)
        x = torch.sigmoid(x)  # Using sigmoid for binary classification

        return x

class BirdDroneResNet50(nn.Module):
    def __init__(self):
        super(BirdDroneResNet50, self).__init__()

        # Load pretrained ResNet-50
        base_model = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout(p=0.3)

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # outputs [B, 2048, 1, 1]

        # New classifier layer for binary output
        self.classifier = nn.Linear(2048, 1)  # One output unit for binary classification

    def forward(self, x):
        x = self.backbone(x)         # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten to [B, 2048]
        x = self.classifier(x)       # [B, 1]
        x = torch.sigmoid(x)         # Sigmoid for binary classification
        return x.squeeze(1)          # [B]

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {'Drone': 0, 'Bird': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        label = self.label_map[self.df.loc[idx, 'label']]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

resnet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Dataset wrapper for feature extraction
class ImageFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {'Drone': 0, 'Bird': 1}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        label = self.label_map[self.df.loc[idx, 'label']]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


#train_dataset = ImageDataset(df_train, transform=transform)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_dataset = ImageDataset(df_test, transform=transform)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# train_feat_dataset = ImageFeatureDataset(df_train, resnet_transform)
# test_feat_dataset = ImageFeatureDataset(df_test, resnet_transform)
# train_feat_loader = DataLoader(train_feat_dataset, batch_size=32, shuffle=False)
# test_feat_loader = DataLoader(test_feat_dataset, batch_size=32, shuffle=False)
