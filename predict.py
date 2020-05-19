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
import image_processing_utils

# Collect arguments from cmd line and parse them
ap = argparse.ArgumentParser(description = "This file loads a pretrained model checkpoint and predict a flower name "
                                           "from the image and displays the probabilities",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("image_path", metavar="image_path", help="Location where test image is stored", type=str)
ap.add_argument("checkpoint_file", metavar="checkpoint_file", help="filename of the saved model checkpoint", type=str)
ap.add_argument("--top_k", help="Returns top most likely classes", default="5", type=int)
ap.add_argument("--category_names", help="file for mapping of categories to real names", type=str)
ap.add_argument("--gpu", help="Use gpu for inference", default="cpu", type=str)
args = vars(ap.parse_args())

print("image_path : ", args["image_path"])


############################################
#               Functions
############################################

def load_Checkpoint(filename):
    ''' function that loads a checkpoint and rebuilds the model
    '''
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(filename, map_location='cpu')

    # get pre-trained model
    # model = checkpoint['pretrained_model']
    model = models.resnet50(pretrained=True)

    # freeze parameters - to prevent gradients and backprop
    for param in model.parameters():
        param.requires_grad = False

    # TODO - this function should load for any architecture

    classifier = nn.Sequential(nn.Linear(2048, 500),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(500, num_output_classes),
                               nn.LogSoftmax(dim=1)
                               )
    model.fc = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, topk=10):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)

    image_ndarray = image_processing_utils.process_image(image)

    # create a torch tensor of type float32
    image_torch = torch.from_numpy(image_ndarray).float()

    # reshape to incorporate batch size
    batch_img = torch.unsqueeze(image_torch, 0)

    logps = model(batch_img)
    ps = torch.exp(logps)
    print("Max ps : ", ps.max())
    top_prob, top_idx = ps.topk(topk, dim=1)

    print("top idx :", top_idx)

    # index to class mapping
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    # convert torch to numpy
    top_idx = top_idx[0].numpy()
    top_class = [idx_to_class[entry] for entry in top_idx]

    return top_prob, top_class

############################################
#               End of Functions
############################################

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

num_output_classes = len(cat_to_name)
criterion = nn.NLLLoss()
new_model = load_Checkpoint('trained_model_chpt.pth')
print("Loading checkpoint complete")

test_dir = r'C:\Users\janan\.pytorch\flower_data\test'
# test prediction
with torch.no_grad():
    image_path = test_dir + r'\1\image_06743.jpg'  # pink primrose
    # image_path = test_dir+ r'\2\image_05125.jpg' # hard-leaved pocket orchid
    # image_path = test_dir+ r'\76\image_02550.jpg' # morning glory
    # image_path = test_dir+ r'\17\image_03830.jpg' # purple coneflower
    # image_path = test_dir+ r'\74\image_01307.jpg' # rose
    # image_path = test_dir+ r'\78\image_01903.jpg' # lotus lotus
    # image_path = test_dir+ r'\18\image_04272.jpg' # peruvian lily * 4th
    # image_path = test_dir+ r'\15\image_06351.jpg' # yellow iris * 5th

    label_class = [2]  # this should be target label

    savedmodel = new_model
    # savedmodel = modelSaved
    topk = 10

    top_prob, top_class = predict(image_path, savedmodel, topk=10)

    image = Image.open(image_path)
    image_ndarray = image_processing_utils.process_image(image)
    image_torch = torch.from_numpy(image_ndarray)
    # imshow(image_torch)

    print(top_prob)
    print(top_class)

    image_processing_utils.view_classify(image_torch, top_prob, top_class, topk, cat_to_name, label_class)





