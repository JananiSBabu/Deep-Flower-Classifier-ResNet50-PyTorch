import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import numpy as np
from PIL import Image
import json
from collections import OrderedDict
import argparse

# Collect arguments from cmd line and parse them
ap = argparse.ArgumentParser(description="This file is used to train a Deep learning network and save the checkpoint",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("data_directory", metavar="data_directory", help="Location where data is stored", type=str)
ap.add_argument("--save_dir", help="Location to save the results", default='', type=str)
ap.add_argument("--arch", help="Specify the pre-trained deep learning architecture to train on", default="resnet50",
                type=str)
ap.add_argument("--learning_rate", help="Learning rate for optimizer", default=0.003, type=float)
ap.add_argument("--hidden_units", help="number of hidden units for training", default=512, type=int)
ap.add_argument("--epochs", help="Number of epochs for training", default=1, type=int)
ap.add_argument("--gpu", help="Use gpu for training", default="cpu", type=str)
args = vars(ap.parse_args())

print("Datadir : ", args["data_directory"])

# consume user inputs
data_dir = args["data_directory"] #'~/.pytorch/flower_data'
save_dir = args["save_dir"]
arch = args["arch"]
learning_rate = args["learning_rate"]
hidden_units = args["hidden_units"]
epochs = args["epochs"]
device = args["gpu"]


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(p=0.6),
                                       transforms.RandomVerticalFlip(p=0.4),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# TODO: Build and train your network
num_output_classes = 102

# setup to pick up GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # TODO: remove this later

# get pre-trained model
model = models.resnet50(pretrained=True)
# print(model)

# freeze parameters - to prevent gradients and backprop
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048, 500),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(500, num_output_classes),
                           nn.LogSoftmax(dim=1)
                           )
model.fc = classifier

# Initialization
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# move model to cuda if available
model.to(device)

traintime = time.time()

epochs = 3
training_loss = 0

train_losses, valid_losses = [], []

for epoch in range(epochs):
    model.train()
    for images, labels in trainloader:

        # move the variables to GPU
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0

        # turn of dropouts
        model.eval()

        with torch.no_grad():

            for images, labels in validloader:
                # move the variables to GPU
                images, labels = images.to(device), labels.to(device)

                start = time.time()

                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_prob, top_class = ps.topk(1, dim=1)
                targets = labels.view(*top_class.shape)
                isEqual = top_class == targets
                accuracy += torch.mean(isEqual.type(torch.FloatTensor))

            train_losses.append(training_loss / len(trainloader))
            valid_losses.append(test_loss / len(validloader))

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {training_loss / len(trainloader):.3f}.. "
                  f"validation loss: {test_loss / len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(validloader):.3f}")
            print(f"Device = {device}; Time per batch: {(time.time() - start) / len(validloader):.3f} seconds")

            training_loss = 0

            # switch back to training
            model.train()

print(f" \nTotal Time : {(time.time() - traintime) / 60:.3f} minutes")

# Save the checkpoint
checkpoint = {'pretrained_model': models.resnet50(pretrained=True),
              'input_size': 2048,
              'output_size': 102,
              'hidden_layer': 500,
              'model_state_dict': model.state_dict(),
              'optim_state_dict': optimizer.state_dict(),
              'class_to_idx': train_data.class_to_idx}

chpt_file = 'trained_model_chpt.pth'
if save_dir:
    save_file_path = save_dir + chpt_file
else:
    save_file_path = chpt_file

torch.save(checkpoint, save_file_path)
