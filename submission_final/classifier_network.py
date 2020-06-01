import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time
import torch.optim as optim


class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_prob=0.2):
        super().__init__()

        # create input layer
        print("hidden_layers : ", hidden_layers)
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
            x = F.relu(layer(x))
            x = self.dropouts(x)

        # dropouts excluded for output layer
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x


def validate(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0

    for images, labels in validloader:
        # move the variables to GPU
        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)
        loss = criterion(logps, labels)
        valid_loss += loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_prob, top_class = ps.topk(1, dim=1)
        targets = labels.view(*top_class.shape)
        is_equal = top_class == targets
        accuracy += torch.mean(is_equal.type(torch.FloatTensor))

    return valid_loss, accuracy


def train(model, trainloader, criterion, optimizer, device, validloader, epochs=1, print_every=40):
    traintime = time.time()
    steps = 0
    training_loss = 0
    train_losses, valid_losses = [], []

    # move model to cuda if available
    print("device** : ", device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1

            # move the variables to GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every == 0:
                # Perform validation as the model is trained for a certain number of inputs from a batch

                # turn of dropouts
                model.eval()

                # turn off gradient computation for validation
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, validloader, criterion, device)

                train_losses.append(training_loss / len(trainloader))
                valid_losses.append(valid_loss / len(validloader))

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {training_loss / len(trainloader):.3f}.. "
                      f"validation loss: {valid_loss / len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validloader):.3f}")

                training_loss = 0

                # switch back to training
                model.train()

    print(f" \nTotal Time : {(time.time() - traintime) / 60:.3f} minutes")

    return model, train_losses, valid_losses


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


def construct_model(arch, hidden_units, num_output_classes, drop_prob=0.2, learning_rate=0.03):
    # load the pre-trained model
    model, input_size = load_pretrained_models(modelname=arch)

    # find the name of last layer in model
    l = []
    [l.append(name) for name, param in model.named_parameters()]
    last_layer = l[-1]

    # freeze parameters - to prevent gradients and back prop
    for param in model.parameters():
        param.requires_grad = False

    # Create the classifier

    classifier = ClassifierNetwork(input_size, num_output_classes, hidden_units, drop_prob=drop_prob)

    print("classifier object created")

    # add classifier to the pre-trained network
    if "classifier" in last_layer:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif "fc" in last_layer:
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return model, input_size, optimizer
