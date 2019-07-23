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
from utils import train_model_inception, list_plot_multi, initialize_model, set_parameter_requires_grad, list_plot
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
                        default='/var/scratch/jfeins1/moretransfer/',
                        required=False,
                        help='directory to extract tarfile to')

    parser.add_argument('-fold_num',
                        default=0,
                        type=int,
                        required=False,
                        help='the fold index')

    parser.add_argument('-learning_rate',
                        default=0.001,
                        type=float,
                        required=False,
                        help='learning rate')

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
    fold_num = args.fold_num
    lr = args.learning_rate

tar_dir = '/home/jfeinst/Desktop/voronoi_diagrams/test.tar.gz'
tar_extract_path = '/home/jfeinst/Desktop/'
tar_name = tar_dir.split('/')[-1].split('.')[0]
num_epochs = 2

# Data transformations - normalize values are resnet standard
data_transforms = {'train': transforms.Compose([transforms.Resize(299),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.Resize(299),
                                              transforms.ToTensor(),
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

print('--Fold {}--'.format(fold_num + 1))

print('Extracting tarball...')

# Untar tarball containing data
with tarfile.open(tar_dir) as tar:
    subdir_and_files = [tarinfo for tarinfo in tar.getmembers() if
                        tarinfo.name.startswith(tar_name + '/' + fold_lst[fold_num])]
    tar.extractall(members=subdir_and_files, path=tar_extract_path)

# Forming the dataset and dataloader
image_datasets = {x: datasets.ImageFolder(os.path.join(tar_extract_path, tar_name, fold_lst[fold_num], x),
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
model_ft = models.inception_v3(pretrained=True)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(768, num_classes)
model_ft.fc = nn.Linear(2048, num_classes)


child_counter = 0
for child in model_ft.children():
    if child_counter < 16:
        print("child ",child_counter," was frozen")
        for param in child.parameters():
            param.requires_grad = False
    else:
        print("child ",child_counter," was not frozen")
    child_counter += 1

# Send the model to GPU
model_ft = model_ft.to(device)

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

# Gather the parameters to be optimized/updated in this run.
for name, param in model_ft.named_parameters():
    if param.requires_grad == True:
        print(name)

# Print the number of parameters being trained
total_params = sum(p.numel() for p in model_ft.parameters())
total_trainable_params = sum(
    p.numel() for p in model_ft.parameters() if p.requires_grad)
print('Total parameters: ' + str(total_params) + '    Training parameters: ' + str(total_trainable_params) + '\n')

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),
                          lr=lr,
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0.0,
                          amsgrad=False)

scheduler = torch.optim.lr_scheduler.StepLR(step_size=5, optimizer=optimizer_ft, gamma=0.1)

# Setup the loss fxn
weights = torch.tensor([1.0, 5.0]).to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=weights, reduction='mean')
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model_inception(model_ft,
                                       dataloaders_dict,
                                       criterion_weighted,
                                       optimizer_ft,
                                       num_epochs=num_epochs,
                                       learning_rate_scheduler=scheduler,
                                       is_inception=True,
                                       num_classes=num_classes)

# Save the model
torch.save(model_ft.state_dict(), './log/inception' + fold_lst[fold_num] + '.pt')

print('History: ' + str(hist))

print('Plotting data...')
list_plot(hist, 'Val Acc')

'''
# Save the accuracy and loss history
print('Saving history...')
final_val_acc_history.append(val_acc_history)
final_val_loss_history.append(val_loss_history)
final_train_acc_history.append(train_acc_history)
final_train_loss_history.append(train_loss_history)
'''

# Delete directory to make room for next fold
shutil.rmtree(tar_extract_path + tar_name)


'''
# Plot the accuracy and loss
print('Plotting data...')
list_plot_multi(final_train_acc_history, 'Training_Accuracy')
list_plot_multi(final_train_loss_history, 'Training_Loss')
list_plot_multi(final_val_acc_history, 'Validation_Accuracy')
list_plot_multi(final_val_loss_history, 'Validation_loss')
'''