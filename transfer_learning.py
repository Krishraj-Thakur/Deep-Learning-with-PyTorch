# ImageFolder
# Scheduler
# Transfer Learning

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

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly crops and resizes images to 224x224
        transforms.RandomHorizontalFlip(), # Flips images horizontally for augmentation
        transforms.ToTensor(),             # Converts images to PyTorch tensors
        transforms.Normalize(mean, std)           # Normalizes image pixels using mean & std
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),           # Resizes images to 256x256
        transforms.CenterCrop(224),          # Crops the center 224x224 area
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),  # datasets.ImageFolder automatically assigns class labels based on folder names in data/hymenoptera_data/train and data/hymenoptera_data/val
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes   # The classes are stored in class_names, which gets printed out:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train 
                with torch.set_grad_enabled(phase == 'train'): # Calculates loss and gradients only in training mode.
                    outputs = model(inputs)                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()        # Performs backpropagation and updates weights
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.
# ✅ Here, ALL layers (including convolutional ones) are trainable.
# ✅ This means the model will continue learning, updating even the convolutional filters.
# ✅ This is fine-tuning, where the entire model is adapted to the new task.

model = models.resnet18(pretrained=True)  # Load pretrained ResNet
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)  # Replace FC layer with 2 outputs (ants vs. bees)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  # All parameters trainable

# scheduler 

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # this means that every 7 epochs our learning rate is multiplied by 0.1

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)

#######

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False   # Freeze all convolutional layers 
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)  # Replace FC layer with 2 outputs (ants vs. bees)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)   # Only FC layer is trainable

# scheduler 

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # this means that every 7 epochs our learning rate is multiplied by 0.1

model = train_model(model, criterion, optimizer, step_lr_scheduler , num_epochs=20)