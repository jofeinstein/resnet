'''

Transfer learning on ResNet34.
Iterates through k-folds
Expects data to be a .tar.gz file with the following structure
Resets only the last fully connected layer

tar
│
└───fold0
│   │
│   └───train
│   │   │   file111.png
│   │   │   ...
│   │
│   └───val
│       │   file112.png
│       │   ...
│
└───fold1
    │   ...

'''
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
from utils import train_model, list_plot_multi, initialize_model
import tarfile
import shutil


def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-epoch',
                        default=30,
                        type=int,
                        required=False,
                        help='number of epochs to train')

    parser.add_argument('-data_dir',
                        default='/var/scratch/jfeins1',
                        required=False,
                        help='directory of training images')

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

    parser.add_argument('-feature_extract',
                        default=True,
                        type=bool,
                        required=False)

    parser.add_argument('-tar_dir',
                        default='/work/jfeins1/resnet-binary.tar.gz',
                        required=False,
                        help='directory of tarfile')

    parser.add_argument('-tar_extract_path',
                        default='/var/scratch/jfeins1/',
                        required=False,
                        help='directory to extract tarfile to')

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    num_epochs = args.epoch
    data_dir = args.data_dir
    batch_size = args.batch_size
    fold = args.fold
    feature_extract = args.feature_extract
    tar_dir = args.tar_dir
    tar_extract_path = args.tar_extract_path


tar_name = tar_dir.split('/')[-1].split('.')[0]

# Data transformations - normalize values are resnet standard
data_transforms = {'train': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


fold_lst = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
final_val_acc_history = []
final_val_loss_history = []
final_train_acc_history = []
final_train_loss_history = []


# Determine whether to use GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: ' + str(device) + '\n')
if torch.cuda.device_count() > 1:
    print("Using " + str(torch.cuda.device_count()) + " GPUs...")


for i in range(fold):
    print('--Fold {}--'.format(i+1))
    print('Extracting tarball...')


    # Untar tarball containing data
    with tarfile.open(tar_dir) as tar:
        subdir_and_files = [tarinfo for tarinfo in tar.getmembers() if
                            tarinfo.name.startswith(tar_name + '/' + fold_lst[i])]
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, members=subdir_and_files, path=tar_extract_path)


    # Forming the dataset and dataloader
    image_datasets = {x: datasets.ImageFolder(os.path.join(tar_extract_path, tar_name, fold_lst[i], x),
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4) for x in ['train', 'val']}


    class_names = image_datasets['train'].classes
    num_classes = len(class_names)


    print('Size of training dataset: ' + str((len(image_datasets['train']))) + '    Size of validation dataset: ' +
          str(len(image_datasets['val'])) + '    Number of classes: ' + str(num_classes))


    # Initialize the model
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)

    # Gather and show the parameters to be optimized/updated in this run.
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print(name)

    # Print the number of parameters being trained
    total_params = sum(p.numel() for p in model_ft.parameters())
    total_trainable_params = sum(
        p.numel() for p in model_ft.parameters() if p.requires_grad)
    print('Total parameters: ' + str(total_params) + '    Training parameters: ' + str(total_trainable_params) + '\n')

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update,
                              lr=0.001,
                              betas=(0.9, 0.999),
                              eps=1e-10,
                              weight_decay=0.0001,
                              amsgrad=False)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    trained_model_ft, val_acc_history, val_loss_history, \
    train_acc_history, train_loss_history = train_model(model_ft,
                                                        dataloaders_dict,
                                                        criterion,
                                                        optimizer_ft,
                                                        num_epochs=num_epochs,
                                                        learning_rate_scheduler=None)

    # Save the model
    torch.save(trained_model_ft.state_dict(), './log/resnet34' + fold_lst[i] + '.pt')

    # Save the accuracy and loss history
    final_val_acc_history.append(val_acc_history)
    final_val_loss_history.append(val_loss_history)
    final_train_acc_history.append(train_acc_history)
    final_train_loss_history.append(train_loss_history)

    print('Validation accuracy: ' + str(val_acc_history))
    print('Validation loss: ' + str(val_loss_history))
    print('Training accuracy: ' + str(train_acc_history))
    print('Training loss: ' + str(train_loss_history))

    # Delete directory to make room for next fold
    shutil.rmtree(tar_extract_path + tar_name)

# Plot the accuracy and loss
print('Plotting data...')
list_plot_multi(final_train_acc_history, 'Training_Accuracy')
list_plot_multi(final_train_loss_history, 'Training_Loss')
list_plot_multi(final_val_acc_history, 'Validation_Accuracy')
list_plot_multi(final_val_loss_history, 'Validation_loss')