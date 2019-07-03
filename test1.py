from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import numpy as np
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

data_transforms = {'train': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_dir = '/home/jfeinst/Projects/bionoi_files/resnet-test/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print('Size of training dataset: ' + str(dataset_sizes['train']))
print('Size of training dataset: ' + str(dataset_sizes['val']))
print('Number of classes: ' + str(len(class_names)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: ' + str(device))

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
feature_size = features.shape[1]*features.shape[2]*features.shape[3]

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

for param in model.parameters():
    param.requires_grad = False

feature_size = 256
fc = nn.Sequential(nn.Linear(feature_size, 256),
                   nn.ReLU(),
                   nn.Dropout(0.4),
                   nn.Linear(256, len(class_names)),
                   nn.LogSoftmax(dim=1))

model.classifier = fc

total_params = sum(p.numel() for p in model.parameters())
print('Total parameters: ' + str(total_params))
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print('Training parameters: ' + str(total_trainable_params))

if torch.cuda.device_count() > 1:
    print("Using " + str(torch.cuda.device_count()) + " GPUs...")
    model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
