from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from utils import train_model_no_validation, list_plot, initialize_model

def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-epoch',
                        default=30,
                        type=int,
                        required=False,
                        help='number of epochs to train')

    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')

    parser.add_argument('-model_file',
                        default='./log/resnet34.pt',
                        required=False,
                        help='file to save the model')

    parser.add_argument('-batch_size',
                        default=256,
                        type=int,
                        required=False,
                        help='the batch size, normally 2^n.')

    parser.add_argument('-feature_extract',
                        default=True,
                        type=bool,
                        required=False)


    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    num_epochs = args.epoch
    data_dir = args.data_dir
    model_file = args.model_file
    batch_size = args.batch_size
    feature_extract = args.feature_extract

data_dir = "/Users/jofeinstein/Documents/bionoi-project/bionoi-test/voronoi_diagrams/resnet-test/fold0/train/"
num_epochs = 2
model_file = './log/resnet34noval.pt'

# Data augmentation and normalization
data_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


print("Initializing Dataset and Dataloader...")
# Create training and validation datasets
image_dataset = datasets.ImageFolder(data_dir, data_transform)
# Create training and validation dataloaders
dataloader = torch.utils.data.DataLoader(image_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)


class_names = image_dataset.classes
num_classes = len(class_names)

print('Size of training dataset: ' + str((len(image_dataset))) + '    Number of classes: ' + str(num_classes))


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: ' + str(device))


# Initialize the model for this run
model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)


# Send the model to GPU
model_ft = model_ft.to(device)


if torch.cuda.device_count() > 1:
    print("Using " + str(torch.cuda.device_count()) + " GPUs...")
    model_ft = nn.DataParallel(model_ft)


# Gather the parameters to be optimized/updated in this run.
params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)


total_params = sum(p.numel() for p in model_ft.parameters())
total_trainable_params = sum(
    p.numel() for p in model_ft.parameters() if p.requires_grad)
print('Total parameters: ' + str(total_params) + '    Training parameters: ' + str(total_trainable_params) + '\n')


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()


# Train and evaluate
trained_model_ft, train_acc_history, train_loss_history = train_model_no_validation(model_ft,
                                                                                    dataloader,
                                                                                    criterion,
                                                                                    optimizer_ft,
                                                                                    num_epochs=num_epochs)

# Plot accuracy and loss
list_plot(train_acc_history, 'Training_Accuracy')
list_plot(train_loss_history, 'Training_Loss')


# save the model
torch.save(trained_model_ft.state_dict(), model_file)