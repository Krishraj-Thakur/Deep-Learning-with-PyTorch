import torch
import torch.nn as nn
import numpy as np


class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size): # hidden size refers to no of neurons in the hidden layer
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # creates a fully connected dense layer and takes inputsize features and maps them to hidden size neurons
        self.relu = nn.ReLU() # activation function for rectified linear unit
        self.linear2 = nn.Linear(hidden_size, 1)   # seocnd linear layer that maps hiddensize neurons to single output neuron
    
    def forward(self, x):
        out = self.linear1(x) # passes x through the first layer
        out = self.relu(out) # introduces non linearity through the activation of ReLU
        out = self.linear2(out) # passes output throguh the second linear layer
        # sigmoid at the end
        y_pred = torch.sigmoid(out) # sigmoid to ensure values are between 0 and 1
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5) # Instantiates the neural network with:input_size = 28*28 → The input is an image with 28x28=784 pixels.
criterion = nn.BCELoss()                            # hidden_size = 5 → The hidden layer has 5 neurons.


