import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import argparse
from classifier_network import train, construct_model

# Collect arguments from cmd line and parse them
ap = argparse.ArgumentParser(description="This file is used to train a Deep learning network and save the checkpoint",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("data_directory", metavar="data_directory", help="Location where data is stored", type=str)
ap.add_argument("--save_dir", help="Location to save the results", default='', type=str)
ap.add_argument("--arch", help="Specify the pre-trained deep learning architecture to train on", default="resnet50",
                type=str)
ap.add_argument("--learning_rate", help="Learning rate for optimizer", default=0.003, type=float)
ap.add_argument("--hidden_units", help="number of hidden units for training", nargs='*', default=[], type=int)
ap.add_argument("--epochs", help="Number of epochs for training", default=15, type=int)
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
device = torch.device('cuda' if args['gpu'] == 'gpu' and torch.cuda.is_available() else 'cpu')
print("Device selected  :  ", device)

# hidden_units = [1024, 512, 256]
print("hidden_units : ", hidden_units)
print("arch : ", arch, type(arch))

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

# Using the image datasets and the transforms, define the data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Build and train your network
num_output_classes = len(train_data.class_to_idx)
drop_prob = 0.2

# Construct a model from pretrained network and add a custom classifier
model, input_size, optimizer = construct_model(arch, hidden_units, num_output_classes, drop_prob, learning_rate)

print("back in train.py")

# Initialization
criterion = nn.NLLLoss()

# Do the training
model, train_losses, valid_losses = train(model, trainloader, criterion, optimizer, device, validloader, epochs)
print("\n \n Training complete.. \n ")

# Save the checkpoint
checkpoint = {'pretrained_model': arch,
              'input_size': input_size,
              'output_size': num_output_classes,
              'hidden_units': hidden_units,
              'model_state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'drop_prob': drop_prob,
              'criterion': criterion,
              'optimizer': optimizer,
              'optim_state_dict': optimizer.state_dict(),
              'train_losses': train_losses,
              'valid_losses': valid_losses}

chpt_file = 'trained_model_chpt.pth'
if save_dir:
    save_file_path = save_dir + chpt_file
else:
    save_file_path = chpt_file

torch.save(checkpoint, save_file_path)
