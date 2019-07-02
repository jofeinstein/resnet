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
from utils import train_model

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

    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    num_epochs = args.epoch
    data_dir = args.data_dir
    model_file = args.model_file
    batch_size = args.batch_size
    normalize = args.normalize


# Image normalization/ transformations
if normalize == True:
    print('normalizing data:')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
else:
    transform = transforms.Compose([transforms.ToTensor()])

# Datasets from directories

datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,shuffle=True, num_workers=32) for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)