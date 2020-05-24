import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_prob=0.2):
        super().__init__()

        # create input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # add all the hidden layers
        layers = hidden_layers[:]
        layers_sizes = zip(layers[:-1], layers[1:])
        self.hidden_layers.extend([nn.Linear(i, o) for i, o in layers_sizes])

        # add dropouts
        self.dropouts = nn.Dropout(p=drop_prob)

        # Create output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        # add activation functions and dropouts to each layer
        for layer in self.hidden_layers:
            x = nn.ReLU(layer(x))
            x = self.dropouts(x)

        # dropouts excluded for output layer
        x = self.output_layer(x)
        x = nn.LogSoftmax(x, dim=1)
        return x


def load_pretrained_models(modelname="resnet50"):
    input_size = 0
    if modelname == "resnet50":
        model = models.resnet50(pretrained=True)
    elif modelname == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    # extract input nodes of the pre-trained classifier
    # This will be used when the creating custom classifier
    classifier_layer = list(model.children())[-1]
    input_size = classifier_layer.in_features

    return model, input_size
