# 1) Design model (input, output, size, forward pass)
# 2) Construct the loss and optimizer
# 3) Training loop
#   - forward pass: compute predictions
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)# creating synthetic dataset for linear regression
#creates a 100 data points, each point has 1 feature(input varible),noise for making problem harder,random state for reproducing same data each time
X = torch.from_numpy(X_numpy.astype(np.float32)) #converting arrays from numpy array to 32bit foat
y = torch.from_numpy(y_numpy.astype(np.float32)) # same as above
y = y.view(y.shape[0], 1) # reshapes y into a column vector for matrix multiplication

n_samples, n_features = X.shape
# 1)
input_size = n_features
output_size = 1 # model predicts one output
model = nn.Linear(input_size, output_size) # creates linear model

# 2)loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()# mean squared error loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)# using stochastic gradient descent to update weights
# model.parameters gets th weight and bias from the model
# 3) training loop
num_epoch = 100
for epoch in range(num_epoch):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted,y)

    #backward pass 
    loss.backward() # computes gradient of loss w.r.t to w and b


    #update
    optimizer.step() # updates weights using gradient descent 
    optimizer.zero_grad() # cleans previous gradient so that they dont accumulate

    if(epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach() # stops tracking gradients
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
    