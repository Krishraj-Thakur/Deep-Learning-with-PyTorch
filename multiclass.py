import torch
import torch.nn as nn
import numpy as np  

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): # most of the working is similar to binary
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)   # seocnd linear layer that maps hiddensize neurons to multiple output neurons
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (applies Softmax)