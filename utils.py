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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer,  learning_rate_scheduler=None, num_epochs=25, num_classes=2):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('Beginning training...')

    for epoch in range(num_epochs):


        time_taken = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if learning_rate_scheduler != None:
                    learning_rate_scheduler.step()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('Epoch {}/{}'.format(epoch + 1, num_epochs) + '    {} - Loss: {:.4f}'.format(phase, epoch_loss) + '  Acc: {:.4f}'.format(epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(float(epoch_acc))
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(float(epoch_acc))
                train_loss_history.append(epoch_loss)

            if phase == 'val':
                confusion_matrix = torch.zeros(num_classes, num_classes)
                with torch.no_grad():
                    for i, (inputs, classes) in enumerate(dataloaders['val']):
                        inputs = inputs.to(device)
                        classes = classes.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        for t, p in zip(classes.view(-1), preds.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                print('Confusion matrix:')
                print(confusion_matrix)

        time_used = time.time() - time_taken
        print('Time elapsed: {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
          + '      Best val Acc: {:4f}'.format(best_acc))
    print('-' * 15 + '\n')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history


def train_model_no_validation(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('Beginning training...')

    for epoch in range(num_epochs):
        # Each epoch has only a training phase

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1) + '   Loss: {:.4f}'.format(epoch_loss)
              + '  Acc: {:.4f}'.format(epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
          + '      Best val Acc: {:4f}'.format(best_acc) + '\n')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, train_loss_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize variables
    model_ft = None
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 256

    return model_ft, input_size


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def list_plot_multi(lst, title):
    fig = plt.figure()
    for i in range(len(lst)):
        plt.plot(lst[i])
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title(title)
    # plt.legend()
    plt.draw()
    fig.savefig('./log/' + title + '.png', dpi=500)


def list_plot(lst, title):
    fig = plt.figure()
    plt.plot(lst)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title(title)
    # plt.legend()
    plt.draw()
    fig.savefig('./log/' + title + '.png', dpi=500)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')