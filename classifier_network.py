import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time
import torch.optim as optim


class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_prob=0.2):
        """
            Creates a feed-forward classifier for a specified number of hidden layers

            Args:
                self: this object
                input_size: number of nodes in input layer
                output_size: number of nodes in output layer
                hidden_layers: list of number of nodes for each hidden layer
                drop_prob : drop probability for training
        """
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
        """
            performs a forward pass of the input image through the network

            Args:
                self: this object
                x: input image
            Returns:
                x: image after going through a forward pass
        """
        # add activation functions and dropouts to each layer
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropouts(x)

        # dropouts excluded for output layer
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x


def validate(model, valid_loader, criterion, device):
    """
        Performs validation on the model using data specified in valid_loader

        Args:
            model: fully trained model
            valid_loader: data loader that loads the validation data
            criterion: criterion for defining the loss
            device: execution device
        Returns:
            valid_loss: validation loss
            accuracy: classification accuracy
    """
    valid_loss = 0
    accuracy = 0

    for images, labels in valid_loader:
        # move the variables to GPU, if requested
        images, labels = images.to(device), labels.to(device)

        # perform a forward pass on validation image
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


def train(model, train_loader, criterion, optimizer, device, valid_loader, epochs=1, print_every=40):
    """
        Trains the model using data specified in train_loader and performs validation as we train

        Args:
            model: pre-trained model to be used for transfer learning
            train_loader: data loader that loads teh training data
            criterion: criterion for defining the loss
            optimizer: optimizer for the classifier
            device: execution device
            valid_loader: data loader that loads the validation data
            epochs: number of epochs
            print_every: interval for performing validation and printing results
        Returns:
            model: fully trained model
            train_loss: training loss
            valid_loss: validation loss
    """
    traintime = time.time()
    steps = 0
    training_loss = 0
    train_losses, valid_losses = [], []

    # move model to cuda if available
    print("device** : ", device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
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
                    valid_loss, accuracy = validate(model, valid_loader, criterion, device)

                train_losses.append(training_loss / len(train_loader))
                valid_losses.append(valid_loss / len(valid_loader))

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {training_loss / len(train_loader):.3f}.. "
                      f"validation loss: {valid_loss / len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(valid_loader):.3f}")

                training_loss = 0

                # switch back to training
                model.train()

    print(f" \nTotal Time : {(time.time() - traintime) / 60:.3f} minutes")

    return model, train_losses, valid_losses


def load_pretrained_models(model_name="resnet50"):
    """
        Build the pre-trained model for the specified deep learning architecture

        Args:
            model_name: pre-trained model architecture name
        Returns:
            model: loaded trained model
            input_size: input size for creating the classifier
    """
    input_size = 0
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    # extract input nodes of the pre-trained classifier
    # This will be used when the creating custom classifier
    classifier_layer = list(model.children())[-1]
    input_size = classifier_layer.in_features

    return model, input_size


def construct_model(arch, hidden_units, num_output_classes, drop_prob=0.2, learning_rate=0.03):
    """
        function rebuilds the model from scratch using pre-trained architecture
        and custom classifier using transfer learning

        Args:
            arch: pre-trained model architecture to use for transfer learning
            hidden_units: list of number of hidden units
            num_output_classes: number of output classes
            drop_prob: drop probability for training (default = 0.2)
            learning_rate: learning rate for training (default = 0.03)
        Returns:
            model: loaded trained model from checkpoint filename
            input_size: input size for creating the classifier
            optimizer: optimizer
    """

    # load the pre-trained model
    model, input_size = load_pretrained_models(model_name=arch)

    # find the name of last layer in model.
    # Pre-trained models can have classifier layer name as "fc" or "classifier"
    l = []
    [l.append(name) for name, param in model.named_parameters()]
    last_layer = l[-1]

    # freeze parameters - to prevent gradients and back prop
    for param in model.parameters():
        param.requires_grad = False

    # Create the custom classifier
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
