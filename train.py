import numpy as np
import torch.nn as nn
import argparse

# Collect arguments from cmd line and parse them
ap = argparse.ArgumentParser(description = "This file is used to train a Deep learning network and save the checkpoint",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("data_directory", metavar="data_directory", help="Location where data is stored", type=str)
ap.add_argument("--save_dir", help="Location to save the results", type=str)
ap.add_argument("--arch", help="Specify the pre-trained deep learning architecture to train on", default="resnet50", type=str)
ap.add_argument("--learning_rate", help="Learning rate for optimizer", default=0.001, type=float)
ap.add_argument("--hidden_units", help="number of hidden units for training", default=512, type=int)
ap.add_argument("--epochs", help="Number of epochs for training", default=1, type=int)
ap.add_argument("--gpu", help="Use gpu for training", default="cpu", type=str)
args = vars(ap.parse_args())

print("Default for learning rate : ", args["learning_rate"])