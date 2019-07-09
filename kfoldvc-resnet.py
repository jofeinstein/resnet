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
from utils import train_model, list_plot, initialize_model


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
                        default='./log/bionoi_autoencoder_conv.pt',
                        required=False,
                        help='file to save the model')

    parser.add_argument('-batch_size',
                        default=256,
                        type=int,
                        required=False,
                        help='the batch size, normally 2^n.')

    parser.add_argument('-fold',
                        default=5,
                        type=int,
                        required=False,
                        help='number of folds to cross validate with')

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    num_epochs = args.epoch
    data_dir = args.data_dir
    model_file = args.model_file
    batch_size = args.batch_size
    fold = args.fold

data_dir = "/home/jfeinst/Projects/bionoi_files/resnet-test"
model_name = "resnet34"
batch_size = 8
num_epochs = 2
feature_extract = True
model_file = 'resnet34test'



data_transforms = {'train': transforms.Compose([transforms.Resize(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.Resize(256),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


fold_lst = ('fold0', 'fold1', 'fold2', 'fold3', 'fold4')
final_val_acc_history = []
final_val_loss_history = []
final_train_acc_history = []
final_train_loss_history = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: ' + str(device) + '\n')
if torch.cuda.device_count() > 1:
    print("Using " + str(torch.cuda.device_count()) + " GPUs...")


for i in range(fold):
    print('Fold {}'.format(i+1))
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, fold_lst[i], x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print('Size of training dataset: ' + str(image_datasets['train']))
    print('Size of training dataset: ' + str(image_datasets['val']))
    print('Number of classes: ' + str(len(class_names)))

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)

    # Gather the parameters to be optimized/updated in this run.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    total_params = sum(p.numel() for p in model_ft.parameters())
    total_trainable_params = sum(
        p.numel() for p in model_ft.parameters() if p.requires_grad)
    print('Total parameters: ' + str(total_params) + 'Training parameters: ' + str(total_trainable_params))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    trained_model_ft, val_acc_history, val_loss_history, train_acc_history, train_loss_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    torch.save(trained_model_ft.state_dict(), model_file + fold_lst[i] + '.pt')

    final_val_acc_history.append(val_acc_history)
    final_val_loss_history.append(val_loss_history)
    final_train_acc_history.append(train_acc_history)
    final_train_loss_history.append(train_loss_history)

list_plot(final_train_acc_history, 'Training_Accuracy')
list_plot(final_train_loss_history, 'Training_Loss')
list_plot(final_val_acc_history, 'Validation_Accuracy')
list_plot(final_val_loss_history, 'Validation_loss')